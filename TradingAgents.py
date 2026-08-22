#!/usr/bin/env python3
"""
===============================================================
TRADINGAGENTS - FREE PIPELINE v3 (Groq + YFinance)  (FULLY PATCHED)
===============================================================
Based on: "TradingAgents: Multi-Agents LLM Financial Trading Framework"
(arXiv:2412.20138) & https://github.com/TauricResearch/TradingAgents

Architecture (5-Stage Firm Simulation from the paper):
I.   ANALYST TEAM   - Technical / News / Sentiment / Fundamentals / Geopolitical (parallel)
II.  RESEARCH TEAM  - Bull vs Bear multi-round debate + facilitator judge
III. TRADER         - Decision signal (BUY/SELL/HOLD) + reflection memory
IV.  RISK TEAM      - Aggressive / Neutral / Conservative review + judge
V.   FUND MANAGER   - Final approval & execution signal

PATCHES APPLIED:
  • Analyst Team upgraded 4 → 5 agents (added GEOPOLITICAL / Shadow Supply Chain)
  • New SENT_SYS, NEWS_SYS, GEOPOL_SYS prompts; TRADER_SYS interprets risk-premium scoring
  • Self-healing data layer (cache-clear + retry)
  • New --diagnose health check flag

Models (Groq Free Tier):
DEEP:     openai/gpt-oss-120b
FAST:     openai/gpt-oss-20b
FALLBACK: qwen/qwen3.6-27b

Install:
    pip install openai yfinance pandas rich

Run:
    python TradingAgents.py NVDA
    python TradingAgents.py --batch "simons"
    python TradingAgents.py --batch ALL
    python TradingAgents.py --list-baskets
    python TradingAgents.py --diagnose
===============================================================
"""
import sys, os, json, re, argparse, time, shutil, platform, tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import yfinance as yf
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.prompt import Prompt

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing dependency -> pip install openai")

console = Console()

# =============================================================
# CONFIG
# =============================================================
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
DIAGNOSE_MODE = "--diagnose" in sys.argv
AI_ENABLED    = False
client        = None

if GROQ_API_KEY:
    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    AI_ENABLED = True
elif not DIAGNOSE_MODE:
    console.print("[bold red]GROQ_API_KEY not set![/bold red]")
    console.print("[yellow]1. Go to https://console.groq.com/keys[/yellow]")
    console.print("[yellow]2. Create a free API key (starts with gsk_...)[/yellow]")
    console.print("[yellow]3. Run: export GROQ_API_KEY='gsk_YOUR_KEY_HERE'[/yellow]")
    sys.exit(1)

MODEL_DEEP   = os.getenv("TA_DEEP", "openai/gpt-oss-120b")
MODEL_FAST   = os.getenv("TA_FAST", "openai/gpt-oss-20b")
MODEL_BACKUP = "qwen/qwen3.6-27b"

DEBATE_ROUNDS = 2
MEMORY_FILE   = "trading_memory.jsonl"
MAX_RETRIES   = 3
RETRY_DELAY   = 5

# =============================================================
# LLM LAYER (retry on 429, fallback on 404, JSON-mode guard)
# =============================================================
def llm(system, user, model=None, temperature=0.7, force_json=False):
    if not AI_ENABLED or client is None:
        return ""
    target_model = model or MODEL_DEEP
    use_json = force_json
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(
                model=target_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=temperature,
                max_tokens=2048,
            )
            if use_json:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "response_format" in err or ("json" in err.lower() and "400" in err):
                use_json = False
                continue
            if "429" in err or "rate" in err.lower():
                wait = RETRY_DELAY * attempt
                console.print(f"[yellow]Rate limited. Waiting {wait}s (attempt {attempt}/{MAX_RETRIES})...[/yellow]")
                time.sleep(wait)
                continue
            if ("404" in err or "not_found" in err or "decommissioned" in err) and target_model != MODEL_BACKUP:
                console.print(f"[yellow]Model {target_model} unavailable -> falling back to {MODEL_BACKUP}[/yellow]")
                target_model = MODEL_BACKUP
                continue
            console.print(f"[red]LLM Error: {e}[/red]")
            return ""
    console.print(f"[red]LLM call failed after {MAX_RETRIES} retries[/red]")
    return ""

def extract_json(text):
    """Robustly extract the first JSON object from an LLM reply."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None

# =============================================================
# CACHE & CONNECTIVITY HEALING
# =============================================================
def clear_yf_cache():
    candidates = [
        os.path.expanduser("~/.cache/py-yfinance"),
        os.path.expanduser("~/Library/Caches/py-yfinance"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "py-yfinance"),
        os.path.join(tempfile.gettempdir(), "py-yfinance"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass
    try:
        yf.set_tz_cache_location(tempfile.gettempdir())
    except Exception:
        pass

def internet_ok(timeout=6):
    """Cheap connectivity probe. Reaching ANY server means internet works."""
    import urllib.error
    for url in ("https://www.google.com", "https://1.1.1.1", "https://api.groq.com"):
        try:
            urllib.request.urlopen(url, timeout=timeout)
            return True
        except urllib.error.HTTPError:
            # 403/404 means we successfully reached the server -> internet is OK
            return True
        except Exception:
            continue
    return False

# =============================================================
# DATA LAYER (YFinance - free, no keys, self-healing)
# =============================================================
def get_price_df(ticker):
    clear_yf_cache()
    for attempt in range(1, 4):
        try:
            df = yf.download(ticker, period="1y", progress=False,
                             auto_adjust=True, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                return df
        except Exception as e:
            clear_yf_cache()
            time.sleep(1.0 * attempt)
    console.print(f"[red]Data error ({ticker}) after retries[/red]")
    return pd.DataFrame()

def compute_indicators(df):
    """Technical indicators: SMA/RSI/MACD/Bollinger/ATR/returns."""
    if df.empty or len(df) < 50:
        return {"error": "Insufficient data"}
    c = df["Close"]
    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / loss))
    macd      = ema12 - ema26
    macd_sig  = macd.ewm(span=9).mean()
    macd_hist = macd - macd_sig
    bb_mid, bb_std = c.rolling(20).mean(), c.rolling(20).std()
    bb_up, bb_lo = bb_mid + 2 * bb_std, bb_mid - 2 * bb_std
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - c.shift()).abs(),
                    (df["Low"]  - c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()

    def safe(v):
        try:
            return round(float(v), 4) if pd.notna(v) else None
        except Exception:
            return None

    L = df.iloc[-1]
    return {
        "date":           str(df.index[-1].date()),
        "close":          safe(L["Close"]),
        "SMA20":          safe(sma20.iloc[-1]),
        "SMA50":          safe(sma50.iloc[-1]),
        "SMA200":         safe(sma200.iloc[-1]),
        "RSI14":          safe(rsi.iloc[-1]),
        "MACD":           safe(macd.iloc[-1]),
        "MACD_signal":    safe(macd_sig.iloc[-1]),
        "MACD_histogram": safe(macd_hist.iloc[-1]),
        "BB_upper":       safe(bb_up.iloc[-1]),
        "BB_lower":       safe(bb_lo.iloc[-1]),
        "ATR14":          safe(atr14.iloc[-1]),
        "return_5d_%":    safe(c.pct_change(5).iloc[-1] * 100),
        "return_20d_%":   safe(c.pct_change(20).iloc[-1] * 100),
        "return_60d_%":   safe(c.pct_change(60).iloc[-1] * 100) if len(c) > 60 else None,
        "52w_high":       safe(c.rolling(252).max().iloc[-1]),
        "52w_low":        safe(c.rolling(252).min().iloc[-1]),
        "volume":         safe(L["Volume"]) if "Volume" in df.columns else None,
    }

def get_news(ticker, limit=8):
    try:
        raw = yf.Ticker(ticker).news or []
        items = []
        for n in raw[:limit]:
            content = n.get("content", n) if isinstance(n, dict) else n
            if isinstance(content, dict):
                title = content.get("title", "")
                prov = content.get("provider", {})
                source = prov.get("displayName", "") if isinstance(prov, dict) else ""
                if title:
                    items.append({"title": title, "source": source})
        return items
    except Exception:
        return []

def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {}
    keep = ["marketCap", "enterpriseValue", "trailingPE", "forwardPE", "pegRatio",
            "priceToBook", "profitMargins", "grossMargins", "operatingMargins",
            "returnOnEquity", "returnOnAssets", "revenueGrowth", "earningsGrowth",
            "debtToEquity", "currentRatio", "quickRatio", "freeCashflow",
            "dividendYield", "targetMeanPrice", "targetHighPrice", "targetLowPrice",
            "recommendationKey", "numberOfAnalystOpinions", "sector", "industry",
            "shortPercentOfFloat", "heldPercentInsiders", "heldPercentInstitutions"]
    return {k: info[k] for k in keep if info.get(k) is not None}

# =============================================================
# AGENT ROLE PROMPTS (5-analyst team, Shadow Alpha tuned)
# =============================================================
TECH_SYS = """You are the MARKET (TECHNICAL) ANALYST at a quantitative trading firm.
Goal: Analyze market trends using technical indicators (SMA, RSI, MACD, Bollinger Bands, ATR).
Interpret momentum, trend strength, volatility regime, support/resistance.
End with a 'Key Points Summary' of 3-5 bullets. Cite actual numbers."""

NEWS_SYS = """You are the NEWS ANALYST at a quantitative trading firm. 
When analyzing headlines, cross-reference against these supply-chain signals:
- Congressional trading disclosures (Pelosi, bipartisan stock ban votes)
- 13F filing season (Burry, Ackman, Druckenmiller position changes)
- Semiconductor bottleneck keywords: CoWoS, ABF substrate, HBM3e
- Geopolitical disruption: sanctions designations, arms trafficking route shifts
- Physical constraints: transformer lead times, CDMO capacity, helium-3 shortage
Score news impact from -1.0 to +1.0. State score first.
End with a 'Key Points Summary' of 3-5 bullets."""

SENT_SYS = """You are the SOCIAL MEDIA / SENTIMENT ANALYST at a quantitative trading firm.
Goal: Gauge crowd and GEOPOLITICAL sentiment.
Look for 'Shadow Supply Chain' signals: 'port seizure', 'dark fleet', 'sanctions evasion', 'dual-use smuggling', 'precursor shortage', 'cartel disruption'.
Score sentiment from -1.0 (Supply Shock/Bearish) to +1.0 (Clear/Bullish).
State score first. Under 100 words."""

FUND_SYS = """You are the FUNDAMENTALS ANALYST at a quantitative trading firm.
Goal: Analyze and evaluate company financials and stock performance.
Assess profitability, growth, valuation, liquidity, leverage, analyst targets, insider/institutional holdings.
End with a 'Key Points Summary' stating under/overvalued assessment."""

GEOPOL_SYS = """You are the GEOPOLITICAL / SHADOW SUPPLY CHAIN ANALYST.
Goal: Assess how illicit supply chain disruptions, sanctions enforcement, 
and organized crime dynamics create pricing power for LEGITIMATE companies.

Signals to watch:
- Cocaine/narcotics route disruptions → precursor chemical demand shifts
- Arms trafficking corridor changes → defense prime contract acceleration  
- Sanctions evasion crackdowns → commodity rerouting → freight rate spikes
- Wildlife/timber trafficking enforcement → legal substitute demand

Score geopolitical risk premium from -1.0 (stable) to +1.0 (severe disruption).
State the score first. Under 200 words.

IMPORTANT INTERPRETATION: A HIGH score (+) means disruption is WORSENING,
which is BEARISH for broad markets but BULLISH for physical-bottleneck /
Shadow Alpha assets (they gain pricing power). Flag this explicitly."""

BULL_SYS = """You are the BULLISH RESEARCHER. Build the strongest evidence-based case FOR investing.
Directly counter the bear's points. Cite numbers from the analyst reports. Max 200 words."""
BEAR_SYS = """You are the BEARISH RESEARCHER. Build the strongest evidence-based case AGAINST investing.
Directly counter the bull's points. Cite numbers from the analyst reports. Max 200 words."""
JUDGE_SYS = """You are the DEBATE FACILITATOR. Review the debate transcript and select the prevailing perspective.
Return ONLY valid JSON: {"winner": "BULL" or "BEAR", "confidence": 0.0-1.0, "rationale": "..."}"""

TRADER_SYS = """You are the TRADER. Synthesize analyst reports, the debate outcome, and your past decisions
(reflect on what worked and what failed). Decide action, timing, and sizing.
Return ONLY valid JSON: {"action": "BUY" or "SELL" or "HOLD", "position_pct": 0-100, "confidence": 0.0-1.0, "rationale": "...", "stop_loss_pct": 0-100, "take_profit_pct": 0-100}

NOTE: The GEOPOLITICAL analyst's score is a RISK PREMIUM score, not a sentiment score.
A high geopolitical score (+) favors Shadow Alpha bottleneck assets (BUY signal for them)
even when general sentiment is bearish. Weigh this accordingly."""

RISK_AGGRESSIVE_SYS = """You are the RISKY ANALYST. Advocate high-reward, high-risk strategies; challenge excessive caution. Max 120 words."""
RISK_NEUTRAL_SYS    = """You are the NEUTRAL ANALYST. Provide a balanced perspective; suggest hedges, scaling, partial sizing. Max 120 words."""
RISK_SAFE_SYS       = """You are the SAFE ANALYST. Emphasize conservative strategy and risk mitigation; flag drawdown risk, suggest cuts or veto. Max 120 words."""
RISK_JUDGE_SYS = """You are the RISK REPORT MANAGER. Weigh the three risk perspectives and adjust the trader's plan within prudent risk constraints.
Return ONLY valid JSON: {"adjusted_action": "BUY" or "SELL" or "HOLD", "adjusted_position_pct": 0-100, "risk_score": 1-10, "risk_notes": "..."}"""
MANAGER_SYS = """You are the FUND MANAGER with final authority. Review the trader's plan and risk-team adjustments; approve, modify, or veto.
Return ONLY valid JSON: {"approved": true or false, "final_action": "BUY" or "SELL" or "HOLD", "final_position_pct": 0-100, "notes": "..."}"""

# =============================================================
# STAGE I: ANALYST TEAM (parallel, per paper Fig.1) — 5 agents
# =============================================================
def run_analysts(ticker):
    df = get_price_df(ticker)
    tech_data = compute_indicators(df) if not df.empty else {"error": "No price data"}
    news_data = get_news(ticker)
    fund_data = get_fundamentals(ticker)
    news_txt = "\n".join("- " + n["title"] + " (" + n.get("source", "") + ")" for n in news_data) or "No recent news found."
    headlines = [n["title"] for n in news_data] if news_data else ["No headlines available"]

    def _tech():
        return llm(TECH_SYS, "Ticker: " + ticker + "\nTechnical data:\n" + json.dumps(tech_data, indent=2))

    def _news():
        return llm(NEWS_SYS, "Ticker: " + ticker + "\nRecent headlines:\n" + news_txt)

    def _sent():
        return llm(SENT_SYS, "Ticker: " + ticker + "\nHeadlines to score:\n" + json.dumps(headlines),
                    model=MODEL_FAST, temperature=0.3)

    def _fund():
        return llm(FUND_SYS, "Ticker: " + ticker + "\nFundamentals:\n" + json.dumps(fund_data, indent=2))

    def _geopol():
        return llm(GEOPOL_SYS,
                    "Ticker: " + ticker +
                    "\nSector: " + str(fund_data.get("sector", "N/A")) +
                    " / " + str(fund_data.get("industry", "N/A")) +
                    "\nRecent headlines:\n" + news_txt,
                    temperature=0.3)

    results = {}
    with ThreadPoolExecutor(max_workers=2) as ex:  # 2-at-a-time respects free-tier rate limits
        futures = {"technical": ex.submit(_tech), "news": ex.submit(_news),
                   "sentiment": ex.submit(_sent), "fundamentals": ex.submit(_fund),
                   "geopolitical": ex.submit(_geopol)}
        for name, fut in futures.items():
            try:
                results[name] = fut.result() or "(no output)"
            except Exception as e:
                results[name] = "Error: " + str(e)
    return results

# =============================================================
# STAGE II: BULL vs BEAR DEBATE (n rounds + facilitator)
# =============================================================
def run_debate(analyst_reports, ticker):
    reports_txt = "\n".join("### " + k.upper() + " REPORT ###\n" + v for k, v in analyst_reports.items())
    transcript, bull_arg, bear_arg = [], "", ""
    for r in range(1, DEBATE_ROUNDS + 1):
        bull_arg = llm(BULL_SYS,
            "Ticker: " + ticker + "\nANALYST REPORTS:\n" + reports_txt +
            "\nBEAR'S LAST ARGUMENT:\n" + (bear_arg or "(You open the debate.)") +
            "\nRound " + str(r) + "/" + str(DEBATE_ROUNDS) + " - make the bullish case.")
        transcript.append(("Bull", r, bull_arg))
        bear_arg = llm(BEAR_SYS,
            "Ticker: " + ticker + "\nANALYST REPORTS:\n" + reports_txt +
            "\nBULL'S LAST ARGUMENT:\n" + bull_arg +
            "\nRound " + str(r) + "/" + str(DEBATE_ROUNDS) + " - make the bearish rebuttal.")
        transcript.append(("Bear", r, bear_arg))
    full = "\n".join("[" + who + " - Round " + str(rd) + "]\n" + arg for who, rd, arg in transcript)
    verdict = extract_json(llm(JUDGE_SYS, "FULL DEBATE TRANSCRIPT:\n" + full,
                               temperature=0.2, force_json=True))
    if not verdict or "winner" not in verdict:
        verdict = {"winner": "BEAR", "confidence": 0.5,
                   "rationale": "Judge unavailable - defaulting to caution."}
    return transcript, verdict

# =============================================================
# REFLECTION MEMORY (local JSONL, FinMem-style)
# =============================================================
def load_memory(ticker, n=5):
    if not os.path.exists(MEMORY_FILE):
        return []
    rows = []
    with open(MEMORY_FILE) as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                if rec.get("ticker") == ticker:
                    rows.append(rec)
            except Exception:
                pass
    return rows[-n:]

def save_memory(ticker, decision):
    rec = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "ticker": ticker,
           "action": decision.get("final_action"),
           "position_pct": decision.get("final_position_pct"),
           "notes": str(decision.get("notes", ""))[:200]}
    with open(MEMORY_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")

# =============================================================
# STAGE III: TRADER
# =============================================================
def run_trader(state, ticker):
    mem = load_memory(ticker)
    if mem:
        mem_txt = "\n".join("- " + str(m.get("date")) + ": " + str(m.get("action")) +
                            " @ " + str(m.get("position_pct")) + "% | " +
                            str(m.get("notes", ""))[:90] for m in mem)
    else:
        mem_txt = "No past decisions - first analysis of this ticker."
    reports = "\n".join("### " + k.upper() + " ###\n" + v[:900] for k, v in state["analysts"].items())
    prompt = ("Ticker: " + ticker + "\nANALYST REPORTS:\n" + reports +
              "\nDEBATE WINNER: " + str(state["verdict"].get("winner")) +
              " (confidence " + str(state["verdict"].get("confidence")) + ")" +
              "\nJudge rationale: " + str(state["verdict"].get("rationale", "")) +
              "\nYOUR PAST DECISIONS (reflect & learn):\n" + mem_txt +
              "\nMake today's decision. Return ONLY JSON.")
    out = llm(TRADER_SYS, prompt, temperature=0.3, force_json=True)
    res = extract_json(out)
    if not res or "action" not in res:
        res = {"action": "HOLD", "position_pct": 0, "confidence": 0.0,
               "rationale": out or "Trader unavailable.", "stop_loss_pct": 0, "take_profit_pct": 0}
    return res

# =============================================================
# STAGE IV: RISK MANAGEMENT TEAM
# =============================================================
def run_risk(state, ticker):
    plan = json.dumps(state["trader"], indent=2)
    ctx = "\n".join("### " + k.upper() + " ###\n" + v[:400] for k, v in state["analysts"].items())
    base = "Ticker: " + ticker + "\nTRADER'S PLAN:\n" + plan + "\nCONTEXT:\n" + ctx
    opinions = {}
    for label, sysp in [("Aggressive", RISK_AGGRESSIVE_SYS),
                        ("Neutral", RISK_NEUTRAL_SYS),
                        ("Conservative", RISK_SAFE_SYS)]:
        opinions[label] = llm(sysp, base + "\nGive your risk assessment.")
    all_op = "\n".join("[" + lab + "]\n" + o for lab, o in opinions.items())
    adj = extract_json(llm(RISK_JUDGE_SYS,
              "TRADER'S PLAN:\n" + plan + "\nRISK OPINIONS:\n" + all_op + "\nReturn ONLY JSON.",
              temperature=0.2, force_json=True))
    if not adj or "adjusted_action" not in adj:
        adj = {"adjusted_action": state["trader"].get("action", "HOLD"),
               "adjusted_position_pct": state["trader"].get("position_pct", 0),
               "risk_score": 5, "risk_notes": "Risk judge unavailable."}
    return opinions, adj

# =============================================================
# STAGE V: FUND MANAGER
# =============================================================
def run_manager(state):
    prompt = ("TRADER PLAN:\n" + json.dumps(state["trader"], indent=2) +
              "\nRISK-ADJUSTED PLAN:\n" + json.dumps(state["risk_adjusted"], indent=2) +
              "\nDEBATE WINNER: " + str(state["verdict"].get("winner")) +
              " (confidence " + str(state["verdict"].get("confidence")) + ")" +
              "\nMake the final call. Return ONLY JSON.")
    out = llm(MANAGER_SYS, prompt, temperature=0.2, force_json=True)
    res = extract_json(out)
    if not res or "final_action" not in res:
        res = {"approved": True,
               "final_action": state["risk_adjusted"].get("adjusted_action", "HOLD"),
               "final_position_pct": state["risk_adjusted"].get("adjusted_position_pct", 0),
               "notes": out or "Fund manager unavailable."}
    return res

# =============================================================
# ORCHESTRATOR
# =============================================================
def run_pipeline(ticker):
    console.print(Panel.fit(
        "[bold magenta]TRADINGAGENTS: " + ticker + "[/bold magenta]\n"
        "[dim]" + datetime.now().strftime("%Y-%m-%d %H:%M") +
        " | deep: " + MODEL_DEEP + " | fast: " + MODEL_FAST + "[/dim]",
        box=box.DOUBLE, border_style="magenta"))
    state = {"ticker": ticker}
    t0 = time.time()

    with console.status("[cyan]I. Analyst Team (5 agents)...[/cyan]"):
        state["analysts"] = run_analysts(ticker)
        for k, v in state["analysts"].items():
            console.print("  [green]ok[/green] analyst:" + k + " (" + str(len(v)) + " chars)")

    with console.status("[cyan]II. Bull vs Bear debate...[/cyan]"):
        state["transcript"], state["verdict"] = run_debate(state["analysts"], ticker)
        console.print("  [green]ok[/green] debate winner: [bold]" + str(state["verdict"].get("winner")) +
                      "[/bold] (confidence " + str(state["verdict"].get("confidence")) + ")")

    with console.status("[cyan]III. Trader deciding...[/cyan]"):
        state["trader"] = run_trader(state, ticker)
        console.print("  [green]ok[/green] trader: [bold]" + str(state["trader"].get("action")) +
                      "[/bold] @ " + str(state["trader"].get("position_pct")) + "%")

    with console.status("[cyan]IV. Risk team deliberating...[/cyan]"):
        state["risk_opinions"], state["risk_adjusted"] = run_risk(state, ticker)
        console.print("  [green]ok[/green] risk-adjusted: [bold]" + str(state["risk_adjusted"].get("adjusted_action")) +
                      "[/bold] @ " + str(state["risk_adjusted"].get("adjusted_position_pct")) +
                      "% (risk " + str(state["risk_adjusted"].get("risk_score")) + "/10)")

    with console.status("[cyan]V. Fund manager approving...[/cyan]"):
        state["final"] = run_manager(state)
    elapsed = round(time.time() - t0, 1)

    # Explainability log (paper section 6.1.4)
    log = Table(title="Full Decision Log - " + ticker, box=box.SIMPLE_HEAVY, show_lines=True)
    log.add_column("Stage", style="bold cyan", width=22)
    log.add_column("Output", overflow="fold")
    for k, v in state["analysts"].items():
        log.add_row("Analyst: " + k, (v[:600] + "...") if len(v) > 600 else v)
    for who, rd, arg in state["transcript"]:
        log.add_row("Debate R" + str(rd) + ": " + who, (arg[:450] + "...") if len(arg) > 450 else arg)
    log.add_row("Verdict", json.dumps(state["verdict"], indent=1))
    log.add_row("Trader", json.dumps(state["trader"], indent=1))
    for lab, op in state["risk_opinions"].items():
        log.add_row("Risk: " + lab, (op[:300] + "...") if len(op) > 300 else op)
    log.add_row("Risk-Adjusted", json.dumps(state["risk_adjusted"], indent=1))
    log.add_row("Fund Manager", json.dumps(state["final"], indent=1))
    console.print(log)

    # Final signal panel
    action = state["final"].get("final_action", "HOLD")
    color  = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(action, "white")
    console.print(Panel.fit(
        "[bold " + color + "]FINAL SIGNAL: " + action + "[/bold " + color + "]\n"
        "Position: " + str(state["final"].get("final_position_pct", 0)) + "% | "
        "Approved: " + str(state["final"].get("approved")) + "\n"
        "[italic]" + str(state["final"].get("notes", ""))[:300] + "[/italic]\n"
        "[dim]completed in " + str(elapsed) + "s[/dim]",
        title="Fund Manager Verdict - " + ticker, box=box.DOUBLE, border_style=color))
    save_memory(ticker, state["final"])
    return state

# =============================================================
# SYSTEM DIAGNOSTICS
# =============================================================
def run_diagnostics():
    console.print(Panel.fit(
        "[bold cyan]🧪 TRADINGAGENTS SYSTEM DIAGNOSTICS[/bold cyan]",
        box=box.DOUBLE, border_style="cyan"))
    results = []

    deps_ok = True
    dep_detail = []
    for mod in ("yfinance", "pandas", "rich", "openai"):
        try:
            __import__(mod)
        except Exception:
            deps_ok = False
            dep_detail.append(f"{mod} MISSING")
    results.append(("Python dependencies", deps_ok,
                    "all present" if deps_ok else ", ".join(dep_detail)))

    net = internet_ok()
    results.append(("Internet connectivity", net,
                    "Yahoo/Groq reachable" if net else "no connection"))

    yf_ok, yf_detail = False, "no data"
    if net:
        try:
            clear_yf_cache()
            h = yf.Ticker("SPY").history(period="5d")
            if h is not None and not h.empty:
                yf_ok = True
                yf_detail = f"SPY close ${h['Close'].iloc[-1]:.2f} ({len(h)} bars)"
            else:
                yf_detail = "empty history"
        except Exception as e:
            yf_detail = f"error: {str(e)[:60]}"
    results.append(("Yahoo Finance data feed", yf_ok, yf_detail))

    key_ok = bool(GROQ_API_KEY)
    results.append(("GROQ_API_KEY present", key_ok,
                    "set" if key_ok else "NOT set"))

    def model_test(model):
        if not AI_ENABLED:
            return False, "AI disabled"
        try:
            r = llm("You are a test. Reply with the single word: OK", "ping",
                    model=model, temperature=0.0)
            return bool(r and r.strip()), (r[:30] if r else "empty response")
        except Exception as e:
            return False, f"error: {str(e)[:50]}"

    if AI_ENABLED:
        for label, mdl in [("DEEP model", MODEL_DEEP),
                           ("FAST model", MODEL_FAST),
                           ("BACKUP model", MODEL_BACKUP)]:
            ok, det = model_test(mdl)
            results.append((f"{label} ({mdl})", ok, det))
        jok, jdet = False, "n/a"
        try:
            out = llm("Return ONLY JSON: {\"status\":\"ok\"}", "test",
                      model=MODEL_FAST, temperature=0.0, force_json=True)
            jok = extract_json(out) is not None
            jdet = "parses" if jok else "JSON parse failed"
        except Exception as e:
            jdet = f"error: {str(e)[:50]}"
        results.append(("LLM JSON mode", jok, jdet))
    else:
        results.append(("AI models", False, "skipped — GROQ_API_KEY not set"))

    def writable(path):
        try:
            d = os.path.dirname(os.path.abspath(path)) or "."
            return os.access(d, os.W_OK)
        except Exception:
            return False
    results.append(("File write permissions", writable(MEMORY_FILE),
                    "memory file writable" if writable(MEMORY_FILE) else "check directory permissions"))

    btickers = sorted({t for h in BASKETS.values() for t in h})
    results.append(("Basket config", len(BASKETS) > 0,
                    f"{len(BASKETS)} baskets, {len(btickers)} unique tickers"))

    tbl = Table(title="Diagnostic Results", box=box.HEAVY_HEAD, show_lines=True)
    tbl.add_column("Check", style="bold", min_width=30)
    tbl.add_column("Status", justify="center", width=8)
    tbl.add_column("Detail", overflow="fold")
    passed = 0
    for name, ok, detail in results:
        if ok:
            passed += 1
            status = "[bold green]✅ PASS[/bold green]"
        else:
            status = "[bold red]❌ FAIL[/bold red]"
        tbl.add_row(name, status, detail)
    console.print(tbl)
    score_color = "green" if passed == len(results) else ("yellow" if passed >= len(results)-2 else "red")
    console.print(Panel.fit(
        f"[bold {score_color}]HEALTH SCORE: {passed}/{len(results)} checks passed[/bold {score_color}]",
        box=box.DOUBLE, border_style=score_color))
    if not key_ok:
        console.print("[yellow]💡 Set your key:  export GROQ_API_KEY='gsk_...'[/yellow]")

# =============================================================
# SFA2 BASKETS (full set from sfa2.py)
# =============================================================
BASKETS = {
    "JIM SIMONS (RenTech)": {"NVDA": 1.0, "PLTR": 1.0, "UTHR": 1.0, "META": 1.0, "AAPL": 1.0,
                              "VRSN": 1.0, "EXEL": 1.0, "KGC": 1.0, "SPOT": 1.0, "NFLX": 1.0,
                              "HOOD": 1.0, "GOOGL": 1.0},
    "PELOSI TRACKER": {"NVDA": 1.0, "MSFT": 1.0, "AAPL": 1.0, "GOOGL": 1.0, "AMZN": 1.0,
                        "PANW": 1.0, "CRM": 1.0, "TEM": 1.0},
    "BURRY TRACKER": {"AMZN": 1.0, "VIST": 1.0, "VST": 1.0, "AEP": 1.0, "DTE": 1.0},
    "BUFFETT TRACKER": {"AAPL": 1.0, "BAC": 1.0, "CVX": 1.0, "OXY": 1.0, "KO": 1.0,
                         "AXP": 1.0, "MMM": 1.0},
    "ACKMAN TRACKER": {"GOOGL": 1.0, "META": 1.0, "CMG": 1.0, "NFLX": 1.0, "QSR": 1.0,
                        "HLT": 1.0, "UBER": 1.0},
    "DRUCKENMILLER TRACKER": {"NVDA": 1.0, "MSFT": 1.0, "LLY": 1.0, "UNH": 1.0, "VRT": 1.0,
                               "GE": 1.0, "VRTX": 1.0},
    "CITADEL TRACKER": {"SPY": 1.0, "QQQ": 1.0, "IWM": 1.0, "AAPL": 1.0, "MSFT": 1.0},
    "INVERSE CRAMER (Bear ETFs)": {"SQQQ": 1.0, "SH": 1.0, "SPXU": 1.0},
    "AI WORLD WAR III": {"LMT": 1.0, "RTX": 1.0, "NOC": 1.0, "PLTR": 1.0, "NVDA": 1.0,
                          "GD": 1.0, "SMH": 1.0},
    "GENOME (Biotech Innovation)": {"AMGN": 2.0, "GILD": 2.0, "VRTX": 2.0, "REGN": 2.0,
                                     "MRNA": 1.5, "BNTX": 1.5, "ALNY": 1.5, "IONS": 1.5,
                                     "CRSP": 1.0, "NTLA": 1.0, "BEAM": 1.0, "EDIT": 1.0},
    "TERRA (Global Commodities)": {"XOM": 2.0, "CVX": 2.0, "OXY": 1.5, "FCX": 1.5, "NEM": 1.5,
                                    "GLD": 1.5, "MOS": 1.0, "CF": 1.0, "CORN": 1.0, "PHO": 1.5,
                                    "AWK": 1.0},
    "NEXUS (AI Tech Infrastructure)": {"NVDA": 3.0, "AMD": 2.0, "AVGO": 2.0, "TSM": 2.0,
                                        "MSFT": 2.0, "GOOGL": 2.0, "AMZN": 2.0, "META": 1.5,
                                        "PLTR": 1.5, "NOW": 1.0, "SNOW": 1.0, "VRT": 1.0},
    "WATER (AI Cooling + Scarcity)": {"PHO": 1.0, "FIW": 1.0, "AWK": 1.0, "XYL": 1.0,
                                       "ECL": 1.0, "VRT": 1.0},
    "FLUID & MEMBRANE BOTTLENECKS": {"FLS": 2.0, "ROP": 2.0, "DD": 1.5, "XYL": 1.5, "VRT": 1.5},
    "AGRO-CHEM MONOPOLIES (War Squeeze)": {"NTR": 2.5, "MOS": 2.0, "CF": 2.0, "YARIY": 1.5, "FMC": 1.0},
    "CROP YIELD SHORTAGE (Food Inflation)": {"DBA": 3.0, "WEAT": 2.0, "CORN": 2.0},
    "BULK CHEMICAL FREIGHT": {"ZIM": 2.0, "FLNG": 2.0, "INSW": 1.5},
    "BORDER & AML TECH": {"PLTR": 2.0, "RTX": 1.5, "NICE": 1.5, "GD": 1.0},
    "GOLD MINERS (Traditional Supply)": {"NEM": 2.0, "GOLD": 2.0, "AEM": 2.0, "GDX": 1.5, "GDXJ": 1.0},
    "GOLD ROYALTY (Pick & Shovel)": {"FNV": 2.5, "WPM": 2.5, "RGLD": 2.0, "OR": 1.5},
    "E-WASTE & URBAN MINING": {"UMICY": 2.5, "NDA.DE": 2.5, "BOL.ST": 2.0, "GLNCY": 1.5,
                                "5711.T": 2.0, "5713.T": 1.5, "5714.T": 1.5, "5857.T": 1.0,
                                "SMSMY": 1.5, "NUE": 1.0, "CLH": 1.0, "JMAT.L": 1.5,
                                "TOM.OL": 1.0, "ANDR.VI": 1.0, "AREC": 0.5},
    "GOLD BULLION & ETFs": {"GLD": 3.0, "IAU": 2.0, "PHYS": 1.5, "RING": 1.0},
    "MINING CAPEX (Equipment)": {"CAT": 2.5, "SDVKY": 2.0, "EPI-B.ST": 1.5, "FLIDY": 1.0},
    "EU CANNABIS INFRASTRUCTURE": {"MTRS.ST": 2.5, "TT": 1.5, "LIGHT.AS": 2.0, "SRT.DE": 2.0,
                                    "LIN": 1.5, "AI.PA": 1.5, "SAP.DE": 1.5, "ZBRA": 1.0},
    "DENTAL DISRUPTION (Watch to Short)": {"NVST": 2.0, "XRAY": 2.0, "HSIC": 1.5, "IDXX": 1.0},
    "DRONES & EDGE AI (New Primes)": {"AVAV": 2.5, "KTOS": 2.0, "RCAT": 1.0},
    "EDGE AI & SENSORS (Brains & Eyes)": {"AMBA": 2.0, "TDY": 2.0, "MRCY": 1.5},
    "C-UAS & AI SURVEILLANCE (Shield)": {"AXON": 2.5, "PLTR": 2.5, "LHX": 1.5},
    "RARE EARTH (Drone Motors)": {"MP": 3.0},
    "QUANTUM BOTTLENECKS": {"MKSI": 3.0, "COHR": 2.5, "OXIG.L": 2.0, "PFV.DE": 2.0, "LITE": 1.5},
    "SENSORS & PHOTODETECTORS": {"6965.T": 3.0, "TDY": 2.0, "AMSYF": 1.5, "IPGP": 1.5, "LASR": 1.0},
    "PURE-PLAY QUANTUM (High Risk)": {"IONQ": 2.5, "RGTI": 1.5, "QBTS": 1.5, "QUBT": 0.5, "ARQQ": 0.5},
    "QUANTUM GIANTS (Safe Anchors)": {"IBM": 2.0, "GOOGL": 2.0, "HON": 2.0, "MSFT": 1.5},
    "HELIUM & CRYOGENIC GASES": {"LIN": 3.0, "APD": 2.0},
    "RF & CONTROL ELECTRONICS": {"KEYS": 2.0, "ADI": 2.0, "AMD": 1.5},
    "BIOTECH FOUNDRIES (CDMOs)": {"TMO": 3.0, "DHR": 2.5, "LONN.SW": 2.0, "BIO": 1.5},
    "COLD CHAIN & GLASS": {"CYRX": 2.5, "SHTPY": 2.0, "GLW": 1.5},
    "CROs (Trial Managers)": {"IQV": 2.5, "MEDP": 2.0, "LH": 1.5},
    "LIFE SCIENCE TOOLS": {"ILMN": 2.0, "WAT": 2.0, "A": 1.5},
    "PHARMA GLASS MONOPOLY": {"SHTPY": 3.0, "STVN": 2.5, "GXI.DE": 2.0},
    "SPECIALTY GLASS INFRASTRUCTURE": {"GLW": 3.0, "7741.T": 2.0, "5201.T": 1.5},
    "VIAL ECOSYSTEM": {"WST": 3.0, "MTD": 1.5},
    "BORON RAW MATERIALS": {"RIO": 3.0, "MOS": 1.5, "NTR": 1.5, "LIN": 1.0},
    "BENCHMARKS": {"SPY": 1.0, "QQQ": 1.0, "XBI": 1.0, "GSG": 1.0, "SMH": 1.0, "GLD": 1.0},
}

def batch_run(basket_name):
    if basket_name.upper() == "ALL":
        all_tickers = sorted({t for h in BASKETS.values() for t in h})
        label = "ALL BASKETS"
    else:
        match = [k for k in BASKETS if basket_name.upper() in k.upper()]
        if not match:
            console.print("[red]Basket not found. Available:[/red]")
            for k in BASKETS:
                console.print("  - " + k)
            return
        label = match[0]
        all_tickers = list(BASKETS[label].keys())

    console.print("[bold]Batch: " + label + " | " + str(len(all_tickers)) + " tickers[/bold]")
    results = []
    for i, t in enumerate(all_tickers, 1):
        console.print("\n[bold magenta]=== [" + str(i) + "/" + str(len(all_tickers)) + "] " + t + " ===[/bold magenta]")
        try:
            res = run_pipeline(t)
            f = res.get("final", {})
            results.append({"ticker": t, "action": f.get("final_action", "?"),
                            "position_pct": f.get("final_position_pct", 0),
                            "approved": f.get("approved", False),
                            "debate": res.get("verdict", {}).get("winner", "?"),
                            "notes": str(f.get("notes", ""))[:100]})
        except Exception as e:
            console.print("[red]Failed " + t + ": " + str(e) + "[/red]")
            results.append({"ticker": t, "action": "ERROR", "position_pct": 0,
                            "approved": False, "debate": "?", "notes": str(e)[:100]})
        if i < len(all_tickers):
            time.sleep(2)

    tbl = Table(title="Batch Results - " + label, box=box.DOUBLE_EDGE, show_lines=True)
    tbl.add_column("#", width=4)
    tbl.add_column("Ticker", width=8)
    tbl.add_column("Signal", width=8)
    tbl.add_column("Size %", width=8)
    tbl.add_column("Debate", width=8)
    tbl.add_column("Notes", overflow="fold")
    buys = sells = holds = 0
    for i, r in enumerate(results, 1):
        a = r["action"]
        c = {"BUY": "green", "SELL": "red", "HOLD": "yellow", "ERROR": "red"}.get(a, "white")
        tbl.add_row(str(i), r["ticker"], "[" + c + "]" + a + "[/" + c + "]",
                    str(r["position_pct"]), r["debate"], r["notes"])
        if a == "BUY": buys += 1
        elif a == "SELL": sells += 1
        elif a == "HOLD": holds += 1
    console.print(tbl)
    console.print(Panel.fit("BUY: " + str(buys) + " | SELL: " + str(sells) +
                            " | HOLD: " + str(holds) + " | Total: " + str(len(results)),
                            title="Summary", box=box.ROUNDED))
    csv_file = "tradingagents_batch_" + datetime.now().strftime("%Y%m%d_%H%M") + ".csv"
    pd.DataFrame(results).to_csv(csv_file, index=False)
    console.print("[dim]Exported to " + csv_file + "[/dim]")
# =============================================================
# TRADING MEMORY VIEWER
# =============================================================
def view_memory():
    """Display the reflection memory log (trading_memory.jsonl)."""
    if not os.path.exists(MEMORY_FILE):
        console.print("[yellow]No trading memory found yet. Run some analyses first.[/yellow]")
        return
    rows = []
    with open(MEMORY_FILE) as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                rows.append(rec)
            except Exception:
                pass
    if not rows:
        console.print("[yellow]Memory file is empty.[/yellow]")
        return

    console.print(Panel.fit(
        "[bold cyan]🧠 TRADING MEMORY — Decision History[/bold cyan]\n"
        f"[dim]{len(rows)} recorded decisions[/dim]",
        box=box.DOUBLE, border_style="cyan"))

    tbl = Table(title="Past Decisions", box=box.HEAVY_HEAD, show_lines=True)
    tbl.add_column("#", justify="center", width=4)
    tbl.add_column("Date", width=17)
    tbl.add_column("Ticker", width=8)
    tbl.add_column("Action", width=8)
    tbl.add_column("Size %", justify="right", width=7)
    tbl.add_column("Notes", overflow="fold")

    for i, r in enumerate(rows[-30:], 1):  # show last 30
        a = r.get("action", "?")
        c = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(a, "white")
        tbl.add_row(
            str(i),
            str(r.get("date", "?")),
            str(r.get("ticker", "?")),
            f"[{c}]{a}[/{c}]",
            str(r.get("position_pct", 0)),
            str(r.get("notes", ""))[:120],
        )
    console.print(tbl)

    # Summary stats
    actions = [r.get("action") for r in rows]
    buys = actions.count("BUY")
    sells = actions.count("SELL")
    holds = actions.count("HOLD")
    console.print(Panel.fit(
        f"BUY: {buys} | SELL: {sells} | HOLD: {holds} | Total: {len(rows)}",
        title="Memory Summary", box=box.ROUNDED))

# =============================================================
# BASKET PICKER (helper for menu option 2)
# =============================================================
def basket_picker():
    """Show numbered basket list, return selected basket name."""
    names = list(BASKETS.keys())
    console.print("\n[bold cyan]═══ AVAILABLE BASKETS ═══[/bold cyan]")
    for i, name in enumerate(names, 1):
        count = len(BASKETS[name])
        console.print(f"  [{i:2}] {name} [dim]({count} tickers)[/dim]")
    console.print(f"  [{len(names)+1:2}] 🌐 ALL BASKETS")

    choice = Prompt.ask("\n[bold]Select basket number[/bold]", default="1")
    try:
        idx = int(choice) - 1
        if idx == len(names):
            return "ALL"
        elif 0 <= idx < len(names):
            return names[idx]
        else:
            console.print("[red]Invalid selection.[/red]")
            return None
    except ValueError:
        # Try matching by name
        matches = [n for n in names if choice.upper() in n.upper()]
        if matches:
            return matches[0]
        console.print("[red]Basket not found.[/red]")
        return None

# =============================================================
# INTERACTIVE MENU
# =============================================================
def main_menu():
    """Interactive menu — launched when no CLI arguments are provided."""
    console.print(Panel.fit(
        "[bold magenta]⚛️ TRADINGAGENTS — FREE PIPELINE v3[/bold magenta]\n"
        f"[dim]5-Stage LLM Trading Firm Simulation[/dim]\n"
        f"[dim]Deep: {MODEL_DEEP} | Fast: {MODEL_FAST}[/dim]",
        box=box.DOUBLE, border_style="magenta"))

    while True:
        console.print("\n[bold cyan]╔══════════ TRADINGAGENTS MENU ══════════╗[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [1] 🎯 Single Ticker Analysis           [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [2] 📦 Batch Run by Basket              [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [3] 📋 List All Baskets                 [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [4] 🧠 View Trading Memory              [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [5] ⚡ Quick Multi-Ticker               [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [0] 🚪 Exit                             [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]╚═══════════════════════════════════════════╝[/bold cyan]")

        choice = Prompt.ask(
            "\n[bold]Select option[/bold]",
            choices=["0", "1", "2", "3", "4", "5"],
            default="1")

        if choice == "1":
            ticker = Prompt.ask("[bold]Ticker to analyze[/bold]", default="NVDA").upper().strip()
            if ticker:
                run_pipeline(ticker)

        elif choice == "2":
            basket = basket_picker()
            if basket:
                batch_run(basket)

        elif choice == "3":
            console.print("\n[bold cyan]═══ ALL BASKETS ═══[/bold cyan]")
            for name, holds in BASKETS.items():
                console.print(f"\n[bold yellow]{name}[/bold yellow] [dim]({len(holds)} tickers)[/dim]")
                console.print("  " + ", ".join(holds))

        elif choice == "4":
            view_memory()

        elif choice == "5":
            raw = Prompt.ask(
                "[bold]Enter tickers (comma-separated)[/bold]",
                default="NVDA, AAPL, MSFT")
            tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
            if tickers:
                console.print(f"\n[bold]Running {len(tickers)} tickers...[/bold]")
                for i, t in enumerate(tickers, 1):
                    console.print(f"\n[bold magenta]=== [{i}/{len(tickers)}] {t} ===[/bold magenta]")
                    try:
                        run_pipeline(t)
                    except Exception as e:
                        console.print(f"[red]Failed {t}: {e}[/red]")
                    if i < len(tickers):
                        time.sleep(2)

        elif choice == "0":
            console.print("\n[bold red]👋 Goodbye, Trader.[/bold red]")
            break

        if choice != "0":
            console.input("\n[dim]Press Enter to return to menu...[/dim]")

# =============================================================
# CLI
# =============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradingAgents Free Pipeline")
    parser.add_argument("tickers", nargs="*", help="Ticker(s) to analyze")
    parser.add_argument("--batch", type=str, help="Batch run on an SFA2 basket name (or ALL)")
    parser.add_argument("--list-baskets", action="store_true", help="List baskets")
    parser.add_argument("--diagnose", action="store_true", help="Run system diagnostics")
    parser.add_argument("--rounds", type=int, default=2, help="Debate rounds")
    args = parser.parse_args()
    DEBATE_ROUNDS = args.rounds

    if args.diagnose:
        run_diagnostics()
        sys.exit(0)

    if args.list_baskets:
        for name, holds in BASKETS.items():
            console.print("[bold]" + name + "[/bold] -> " + ", ".join(holds))
        sys.exit(0)

    if not AI_ENABLED:
        console.print("[bold red]AI not available — set GROQ_API_KEY to run analyses.[/bold red]")
        sys.exit(1)

    if args.batch:
        batch_run(args.batch)
    elif args.tickers:
        for t in args.tickers:
            run_pipeline(t.upper().strip())
    if args.batch:
        batch_run(args.batch)
    elif args.tickers:
        for t in args.tickers:
            run_pipeline(t.upper().strip())
    else:
        main_menu()