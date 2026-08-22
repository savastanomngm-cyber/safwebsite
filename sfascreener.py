#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
🔍 SAF SCREENER + AI — Shadow Alpha Asset Discovery Engine  (FULLY PATCHED)
═══════════════════════════════════════════════════════════════
Uses the SAME Groq API setup as TradingAgents.py

PATCHES APPLIED:
  • DEEP_REPORT_SYS now includes a GEOPOLITICAL ANGLE section
  • Self-healing data layer (cache-clear + retry + single-ticker fallback)
  • New System Diagnostics health check (menu option 11)

Install:  pip install yfinance pandas rich numpy openai
Setup:    export GROQ_API_KEY='gsk_YOUR_KEY_HERE'
Run:      python sfascreener.py
═══════════════════════════════════════════════════════════════
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.markdown import Markdown
from rich import box
import os
import sys
import json
import shutil
import re
import time
import platform
import tempfile
import urllib.request

console = Console()

# ╔═══════════════════════════════════════════════════════════╗
# ║  ⚙️  GLOBAL CONFIG                                         ║
# ╚═══════════════════════════════════════════════════════════╝
LOOKBACK_DAYS    = 400
BENCHMARK        = "SPY"
EXPORT_FILE      = "saf_screener_results.csv"
UNIVERSE_FILE    = "saf_candidate_universe.json"
SCORE_THRESHOLD  = 55
BOTTLENECK_PASS  = 22
AI_LOG_FILE      = "saf_ai_reports.json"

# ╔═══════════════════════════════════════════════════════════╗
# ║  🤖 LLM LAYER (Same as TradingAgents.py)                  ║
# ╚═══════════════════════════════════════════════════════════╝
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
AI_ENABLED   = False
client       = None

if GROQ_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        AI_ENABLED = True
    except ImportError:
        console.print("[yellow]⚠️ openai package not installed. AI features disabled.[/yellow]")
        console.print("[yellow]   Run: pip install openai[/yellow]")
else:
    console.print("[yellow]⚠️ GROQ_API_KEY not set. AI features disabled.[/yellow]")
    console.print("[yellow]   Quantitative screening (options 1-7) still works.[/yellow]")
    console.print("[yellow]   For AI features: export GROQ_API_KEY='gsk_...'[/yellow]")

MODEL_DEEP   = os.getenv("TA_DEEP", "openai/gpt-oss-120b")
MODEL_FAST   = os.getenv("TA_FAST", "openai/gpt-oss-20b")
MODEL_BACKUP = "qwen/qwen3.6-27b"
MAX_RETRIES  = 3
RETRY_DELAY  = 5

def llm(system, user, model=None, temperature=0.7, force_json=False):
    """Same LLM call function as TradingAgents.py with retry/fallback."""
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
                max_tokens=3000,
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
                console.print(f"[yellow]Rate limited. Waiting {wait}s...[/yellow]")
                time.sleep(wait)
                continue
            if ("404" in err or "not_found" in err or "decommissioned" in err) and target_model != MODEL_BACKUP:
                console.print(f"[yellow]Model unavailable -> falling back to {MODEL_BACKUP}[/yellow]")
                target_model = MODEL_BACKUP
                continue
            console.print(f"[red]LLM Error: {e}[/red]")
            return ""
    return ""

def extract_json(text):
    """Same JSON extractor as TradingAgents.py."""
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

# ╔═══════════════════════════════════════════════════════════╗
# ║  🤖 AI AGENT PROMPTS (Shadow Alpha Specialized)            ║
# ╚═══════════════════════════════════════════════════════════╝
SUPPLY_CHAIN_SYS = """You are a SHADOW ALPHA SUPPLY CHAIN ANALYST at a macro hedge fund.
Your philosophy: Don't buy the end product. Buy the physical bottleneck that enables it.
You identify oligopolies, monopolies, and choke points in supply chains.
You think in second-order effects: "If X scales 100x, what physically breaks first?"
Given a trend or demand signal, you must:
1. Map the complete physical supply chain (raw materials → components → equipment → logistics)
2. Identify the TOP 3 choke points (who controls the bottleneck?)
3. Suggest specific publicly-traded tickers for each bottleneck
4. Explain WHY each is a bottleneck (market concentration, no substitutes, high capex barriers)
Return ONLY valid JSON:
{
  "trend": "...",
  "supply_chain": ["layer1", "layer2", ...],
  "bottlenecks": [
    {
      "name": "...",
      "why_bottleneck": "...",
      "tickers": ["TICK1", "TICK2"],
      "market_concentration": "high/medium/low",
      "substitutability": "none/limited/many"
    }
  ],
  "top_pick": "TICKER",
  "thesis_summary": "..."
}"""

BOTTLENECK_ANALYST_SYS = """You are a SHADOW ALPHA BOTTLENECK ANALYST at a macro hedge fund.
You evaluate whether a company is a TRUE bottleneck/monopoly using 6 criteria.
Score each criterion 1-5 based on what you know about the company:
1. Market Concentration: Do ≤5 companies control >70% of supply?
2. Substitutability: Is there NO viable alternative?
3. Capital Intensity: Would it take >$1B and >5 years to build a competitor?
4. Regulatory Moat: Are there permits/patents blocking new entrants?
5. Demand Inelasticity: Will buyers pay ANY price?
6. Cross-Sector Demand: Needed by MULTIPLE growing industries?
Return ONLY valid JSON:
{
  "ticker": "...",
  "company_name": "...",
  "what_they_do": "...",
  "scores": {
    "market_concentration": N,
    "substitutability": N,
    "capital_intensity": N,
    "regulatory_moat": N,
    "demand_inelasticity": N,
    "cross_sector_demand": N
  },
  "total": N,
  "reasoning": {
    "market_concentration": "...",
    "substitutability": "...",
    "capital_intensity": "...",
    "regulatory_moat": "...",
    "demand_inelasticity": "...",
    "cross_sector_demand": "..."
  },
  "verdict": "TRUE BOTTLENECK" or "NOT A BOTTLENECK",
  "key_risk": "..."
}"""

DEEP_REPORT_SYS = """You are the FUND MANAGER at the Skia Alpha Fund (SAF).
You write the final investment memo by synthesizing:
- Quantitative screening data (trend, correlation, relative strength, stability)
- Fundamental metrics (margins, market cap, valuation)
- Shadow Alpha bottleneck assessment
Write a concise investment memo (max 300 words) with:
1. VERDICT: BUY / WATCH / SKIP
2. THESIS: One paragraph on WHY this is a Shadow Alpha play
3. RISK: The single biggest risk
4. CATALYST: What event would confirm the thesis
5. SIZING: Suggested weight (0.5x / 1x / 2x / 3x)
6. GEOPOLITICAL ANGLE: Does this asset BENEFIT from supply-chain disruption?
   (If yes, higher disruption = stronger thesis.)
Be decisive. Be specific. Cite numbers."""

CANDIDATE_GEN_SYS = """You are a SHADOW ALPHA DISCOVERY ENGINE.
Given a theme or sector, generate 10 candidate tickers that represent
physical bottlenecks, oligopolies, or "pick and shovel" plays.
Avoid obvious mega-cap picks. Focus on hidden, boring, essential suppliers.
Return ONLY valid JSON:
{
  "theme": "...",
  "candidates": [
    {"ticker": "...", "name": "...", "why": "...", "sector": "..."}
  ]
}"""

# ╔═══════════════════════════════════════════════════════════╗
# ║  🌍 DEFAULT CANDIDATE UNIVERSE                            ║
# ╚═══════════════════════════════════════════════════════════╝
DEFAULT_UNIVERSE = {
    "Specialty Chemicals": [
        "LIN", "APD", "SHW", "PPG", "ECL", "DD", "ALB", "FMC",
        "MOS", "CF", "NTR", "CE", "ASH", "OLN", "WLK", "TROX",
    ],
    "Industrial Equipment": [
        "CAT", "PH", "ITT", "FLS", "ROP", "IEX", "CMI", "DOV",
        "ETN", "EMR", "HON", "MMM", "MIDD", "GTLS", "XMTR",
    ],
    "Life Science Tools": [
        "TMO", "DHR", "A", "WAT", "MTD", "ILMN", "PKI", "BRKR",
        "MKSI", "KEYS", "TECH", "RVTY", "AZTA",
    ],
    "Semicon Equipment": [
        "AMAT", "LRCX", "KLAC", "ASML", "TER", "ONTO", "COHU",
        "ENTG", "UCTT",
    ],
    "Specialty Glass": [
        "GLW", "STVN", "SHTPY", "GXI.DE", "IP", "PKG", "AMCR",
        "SEE", "ATR", "BERY",
    ],
    "Critical Minerals": [
        "FCX", "NEM", "GOLD", "AEM", "RIO", "BHP", "SCCO", "MP",
        "UUUU", "CCJ", "TMC", "LAC",
    ],
    "Shipping & Freight": [
        "ZIM", "INSW", "FLNG", "STNG", "TNK", "DHT", "OET", "TGS",
        "FRO", "EURN",
    ],
    "Water & Environmental": [
        "XYL", "AWK", "WMS", "AOS", "PENT", "TTEK", "DAR",
    ],
    "Defense & Aerospace": [
        "LMT", "RTX", "NOC", "GD", "LHX", "HII", "AVAV", "KTOS",
        "MRCY", "HEI", "HEIA",
    ],
    "Sensors & Connectors": [
        "TDY", "TEL", "APH", "ADI", "TXN", "MCHP", "ON", "AMBA",
        "IPGP", "LASR", "COHR", "LITE",
    ],
    "Energy Infrastructure": [
        "OXY", "XOM", "CVX", "VST", "CEG", "NRG", "ET", "KMI",
    ],
    "MedTech & Diagnostics": [
        "ISRG", "DXCM", "PODD", "BSX", "ZBH", "WST", "COO", "STE",
    ],
}

SAF_BASKET_TICKERS = {
    "💧 Water": ["PHO", "FIW", "AWK", "XYL", "ECL", "VRT", "FLS", "ROP", "DD"],
    "⚔️ Agro-Chem": ["NTR", "MOS", "CF", "YARIY", "FMC"],
    "🥇 Gold/E-Waste": ["NEM", "GOLD", "AEM", "FNV", "WPM", "RGLD", "OR",
                         "UMICY", "NDA.DE", "RIO", "MP"],
    "🌿 EU Cannabis": ["MTRS.ST", "TT", "LIGHT.AS", "SRT.DE", "LIN", "AI.PA"],
    "🛡️ Warfare": ["AVAV", "KTOS", "AMBA", "TDY", "MRCY", "AXON", "PLTR", "LHX", "MP"],
    "⚛️ Quantum": ["MKSI", "COHR", "LITE", "6965.T", "IPGP", "LASR",
                    "IONQ", "RGTI", "QBTS", "IBM", "HON", "LIN", "APD",
                    "KEYS", "ADI", "AMD"],
    "🧬 Biotech Infra": ["TMO", "DHR", "BIO", "CYRX", "SHTPY", "GLW",
                          "IQV", "MEDP", "LH", "ILMN", "WAT", "A"],
    "🔬 Borosilicate": ["SHTPY", "STVN", "GXI.DE", "GLW", "7741.T", "5201.T",
                         "WST", "MTD", "RIO", "MOS", "NTR", "LIN"],
}

# ╔═══════════════════════════════════════════════════════════╗
# ║  🌐 UNIVERSE MANAGEMENT                                   ║
# ╚═══════════════════════════════════════════════════════════╝
def load_universe():
    if os.path.exists(UNIVERSE_FILE):
        try:
            with open(UNIVERSE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_UNIVERSE

def save_universe(universe):
    with open(UNIVERSE_FILE, "w") as f:
        json.dump(universe, f, indent=2)

def get_all_candidates(universe):
    tickers = set()
    for group in universe.values():
        tickers.update(group)
    return sorted(tickers)

# ╔═══════════════════════════════════════════════════════════╗
# ║  🩹 CACHE & CONNECTIVITY HEALING LAYER                     ║
# ╚═══════════════════════════════════════════════════════════╝
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

def _download_batch(tickers, start, end):
    out = {}
    try:
        data = yf.download(tickers, start=start, end=end,
                           progress=False, auto_adjust=True, threads=False)
        if data is None or data.empty:
            return out
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers:
                if ('Close', t) in data.columns:
                    s = data[('Close', t)]
                    if s.dropna().shape[0] > 0:
                        out[t] = s
        else:
            if len(tickers) == 1 and 'Close' in data.columns:
                s = data['Close']
                if s.dropna().shape[0] > 0:
                    out[tickers[0]] = s
    except Exception:
        pass
    return out

# ╔═══════════════════════════════════════════════════════════╗
# ║  📥 DATA FETCHING (self-healing)                          ║
# ╚═══════════════════════════════════════════════════════════╝
def fetch_prices(tickers):
    console.print(f"[dim]📡 Fetching data for {len(tickers)} tickers (self-healing)...[/dim]")
    clear_yf_cache()

    if not internet_ok():
        console.print("[bold red]❌ No internet connection detected.[/bold red]")
        return pd.DataFrame()

    end   = datetime.now()
    start = (end - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    all_data = {}
    chunk_size = 25
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        batch = (i // chunk_size) + 1
        console.print(f"[dim]  -> Batch {batch} ({len(chunk)} tickers)...[/dim]")
        got = {}
        for attempt in range(1, 4):
            got = _download_batch(chunk, start, end_s)
            if got:
                break
            console.print(f"[yellow]⚠️ Batch {batch} failed (attempt {attempt}/3) — clearing cache...[/yellow]")
            clear_yf_cache()
            time.sleep(1.5 * attempt)
        all_data.update(got)

        missing = [t for t in chunk if t not in all_data]
        for t in missing:
            single = _download_batch([t], start, end_s)
            if single:
                all_data.update(single)

    if not all_data:
        return pd.DataFrame()
    console.print(f"[green]✅ Fetched {len(all_data)}/{len(tickers)} tickers.[/green]")
    if len(all_data) < len(tickers):
        miss = sorted(set(tickers) - set(all_data.keys()))
        console.print(f"[yellow]⚠️ {len(miss)} unavailable: {', '.join(miss[:15])}{'...' if len(miss)>15 else ''}[/yellow]")
    return pd.DataFrame(all_data)

def fetch_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":         info.get("longName", ticker),
            "market_cap":   info.get("marketCap"),
            "gross_margin": info.get("grossMargins"),
            "oper_margin":  info.get("operatingMargins"),
            "sector":       info.get("sector", "N/A"),
            "industry":     info.get("industry", "N/A"),
            "employees":    info.get("fullTimeEmployees"),
            "forward_pe":   info.get("forwardPE"),
        }
    except Exception:
        return {}

# ╔═══════════════════════════════════════════════════════════╗
# ║  📊 QUANTITATIVE ANALYSIS                                 ║
# ╚═══════════════════════════════════════════════════════════╝
def period_return(prices, days):
    prices = prices.dropna()
    if len(prices) < 2:
        return None
    if days == "YTD":
        yr = prices.index[-1].year
        yp = prices[prices.index.year == yr]
        if len(yp) < 2:
            return None
        base = yp.iloc[0]
    else:
        cutoff = prices.index[-1] - timedelta(days=days)
        valid  = prices[prices.index >= cutoff]
        if len(valid) < 2:
            return None
        base = valid.iloc[0]
    return ((prices.iloc[-1] - base) / base) * 100

def max_drawdown(prices):
    prices = prices.dropna()
    if len(prices) < 2:
        return None
    peak = prices.cummax()
    dd   = (prices - peak) / peak
    return dd.min() * 100

def compute_correlation(ticker_prices, spy_prices):
    common = ticker_prices.index.intersection(spy_prices.index)
    if len(common) < 60:
        return None
    t_ret = ticker_prices[common].pct_change().dropna()
    s_ret = spy_prices[common].pct_change().dropna()
    common2 = t_ret.index.intersection(s_ret.index)
    if len(common2) < 60:
        return None
    return t_ret[common2].corr(s_ret[common2])

def compute_shadow_alpha_score(pdf, ticker, spy):
    if ticker not in pdf.columns:
        return None
    prices = pdf[ticker].dropna()
    if len(prices) < 100:
        return None
    spy_clean = spy.dropna()

    ytd = period_return(prices, "YTD") or 0
    m1  = period_return(prices, 30) or 0
    sma50 = prices.rolling(50).mean().iloc[-1]
    above_sma = 1 if prices.iloc[-1] > sma50 else 0
    trend_score = min(25, max(0, (ytd * 0.25) + (m1 * 0.25) + (above_sma * 8)))

    corr = compute_correlation(prices, spy_clean)
    if corr is None:
        corr = 0.5
    indep_score = max(0, min(25, (1 - abs(corr)) * 30))

    spy_ytd = period_return(spy_clean, "YTD") or 0
    rel_str = ytd - spy_ytd
    rs_score = min(25, max(0, (rel_str * 0.4) + 12))

    returns = prices.pct_change().dropna()
    ann_vol = returns.std() * np.sqrt(252)
    mdd = max_drawdown(prices) or 0
    stab_score = max(0, min(25, 25 - (ann_vol * 25) - (abs(mdd) * 0.4)))

    total = trend_score + indep_score + rs_score + stab_score
    return {
        "ticker":       ticker,
        "price":        round(prices.iloc[-1], 2),
        "ytd":          round(ytd, 2),
        "correlation":  round(corr, 3),
        "max_dd":       round(mdd, 2),
        "trend":        round(trend_score, 1),
        "independence": round(indep_score, 1),
        "rel_strength": round(rs_score, 1),
        "stability":    round(stab_score, 1),
        "total":        round(total, 1),
    }

def check_investability(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        mc   = info.get("marketCap", 0) or 0
        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        vol  = info.get("averageVolume", 0) or 0
        checks = {
            "market_cap_ok":  mc > 500_000_000,
            "price_ok":       price > 5.0,
            "volume_ok":      vol > 100_000,
            "not_penny":      price > 1.0,
        }
        passed = sum(checks.values())
        return {
            "ticker": ticker, "market_cap": mc, "price": price,
            "avg_volume": vol, "checks": checks, "passed": passed,
            "status": "✅ PASS" if passed >= 3 else "⚠️ REVIEW",
        }
    except Exception:
        return {
            "ticker": ticker, "market_cap": None, "price": None,
            "avg_volume": None, "checks": {}, "passed": 0,
            "status": "❌ NO DATA",
        }

# ╔═══════════════════════════════════════════════════════════╗
# ║  🖥️  DISPLAY HELPERS                                       ║
# ╚═══════════════════════════════════════════════════════════╝
def fmt(val, suffix="%"):
    if val is None:
        return "[dim]—[/dim]"
    c = "green" if val >= 0 else "red"
    a = "▲" if val >= 0 else "▼"
    return f"[{c}]{a} {val:+.2f}{suffix}[/{c}]"

def fmt_mc(mc):
    if mc is None:
        return "—"
    if mc >= 1e12:
        return f"${mc/1e12:.2f}T"
    if mc >= 1e9:
        return f"${mc/1e9:.2f}B"
    if mc >= 1e6:
        return f"${mc/1e6:.1f}M"
    return f"${mc:,.0f}"

def save_ai_report(report_type, data):
    """Log AI reports for audit trail."""
    log = []
    if os.path.exists(AI_LOG_FILE):
        try:
            with open(AI_LOG_FILE, "r") as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append({
        "type": report_type,
        "date": datetime.now().isoformat(),
        "data": data,
    })
    log = log[-100:]
    with open(AI_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2, default=str)

# ╔═══════════════════════════════════════════════════════════╗
# ║  📡 OPTION 1: FULL SCREEN                                 ║
# ╚═══════════════════════════════════════════════════════════╝
def run_full_screen(universe):
    console.print(Panel.fit(
        "[bold cyan]📡 SHADOW ALPHA SCREEN — Full Universe Scan[/bold cyan]\n"
        "[dim]Stages 4-6: Investability + Quant Validation + Scoring[/dim]",
        box=box.DOUBLE, border_style="cyan",
    ))
    all_tickers = get_all_candidates(universe)
    all_tickers.append(BENCHMARK)
    all_tickers = list(set(all_tickers))
    pdf = fetch_prices(all_tickers)
    if pdf.empty or BENCHMARK not in pdf.columns:
        console.print("[red]❌ No data fetched. Check internet / run Diagnostics.[/red]")
        return
    spy = pdf[BENCHMARK]
    results = []
    console.print(f"\n[dim]Scoring {len(all_tickers)-1} candidates...[/dim]")
    for ticker in all_tickers:
        if ticker == BENCHMARK:
            continue
        score = compute_shadow_alpha_score(pdf, ticker, spy)
        if score:
            results.append(score)
    results.sort(key=lambda x: x["total"], reverse=True)

    tbl = Table(title="🔍 Shadow Alpha Screen Results", box=box.HEAVY_HEAD, show_lines=True)
    tbl.add_column("#", justify="center", width=4)
    tbl.add_column("Ticker", style="bold cyan", width=8)
    tbl.add_column("Price", justify="right", width=9)
    tbl.add_column("YTD", justify="right", width=10)
    tbl.add_column("Corr", justify="right", width=7)
    tbl.add_column("MaxDD", justify="right", width=8)
    tbl.add_column("Trend", justify="right", width=6)
    tbl.add_column("Indep", justify="right", width=6)
    tbl.add_column("RelStr", justify="right", width=6)
    tbl.add_column("Stab", justify="right", width=6)
    tbl.add_column("TOTAL", justify="right", width=8, style="bold")
    tbl.add_column("Signal", justify="center", width=12)
    for i, r in enumerate(results, 1):
        if r["total"] >= SCORE_THRESHOLD:
            signal = "[bold green]🎯 CANDIDATE[/bold green]"
        elif r["total"] >= SCORE_THRESHOLD - 10:
            signal = "[yellow]👀 WATCH[/yellow]"
        else:
            signal = "[dim]—[/dim]"
        tbl.add_row(
            str(i), r["ticker"], f"${r['price']:.2f}",
            fmt(r["ytd"]), f"{r['correlation']:.2f}",
            f"{r['max_dd']:.1f}%",
            f"{r['trend']:.0f}", f"{r['independence']:.0f}",
            f"{r['rel_strength']:.0f}", f"{r['stability']:.0f}",
            f"[bold]{r['total']:.1f}[/bold]", signal,
        )
    console.print(tbl)
    candidates = [r for r in results if r["total"] >= SCORE_THRESHOLD]
    watchlist  = [r for r in results if SCORE_THRESHOLD - 10 <= r["total"] < SCORE_THRESHOLD]
    console.print(f"\n[bold green]🎯 {len(candidates)} CANDIDATES[/bold green] (score ≥ {SCORE_THRESHOLD})")
    console.print(f"[yellow]👀 {len(watchlist)} ON WATCHLIST[/yellow]")
    df = pd.DataFrame(results)
    df.to_csv(EXPORT_FILE, index=False)
    console.print(f"\n[dim]💾 Saved to {EXPORT_FILE}[/dim]")
    return results

# ╔═══════════════════════════════════════════════════════════╗
# ║  🔍 OPTION 2: DEEP ANALYSIS                               ║
# ╚═══════════════════════════════════════════════════════════╝
def deep_analysis(universe):
    ticker = Prompt.ask("\n[bold]Enter ticker for deep analysis[/bold]", default="MKSI").upper().strip()
    console.print(f"\n[dim]🔍 Running deep analysis on {ticker}...[/dim]")
    pdf = fetch_prices([ticker, BENCHMARK])
    if pdf.empty or ticker not in pdf.columns:
        console.print(f"[red]❌ Could not fetch data for {ticker}.[/red]")
        return
    spy = pdf[BENCHMARK]

    console.print("\n[bold cyan]═══ STAGE 4: INVESTABILITY FILTER ═══[/bold cyan]")
    inv = check_investability(ticker)
    inv_tbl = Table(box=box.SIMPLE)
    inv_tbl.add_column("Check", style="bold")
    inv_tbl.add_column("Value")
    inv_tbl.add_column("Status")
    inv_tbl.add_row("Market Cap > $500M", fmt_mc(inv["market_cap"]),
                    "✅" if inv["checks"].get("market_cap_ok") else "❌")
    inv_tbl.add_row("Price > $5", f"${inv['price']:.2f}" if inv["price"] else "—",
                    "✅" if inv["checks"].get("price_ok") else "❌")
    inv_tbl.add_row("Avg Volume > 100K", f"{inv['avg_volume']:,.0f}" if inv["avg_volume"] else "—",
                    "✅" if inv["checks"].get("volume_ok") else "❌")
    inv_tbl.add_row("Overall", inv["status"], f"{inv['passed']}/4")
    console.print(inv_tbl)

    console.print("\n[bold cyan]═══ STAGE 5: QUANTITATIVE VALIDATION ═══[/bold cyan]")
    score = compute_shadow_alpha_score(pdf, ticker, spy)
    if not score:
        console.print("[red]❌ Insufficient data.[/red]")
        return
    q_tbl = Table(box=box.SIMPLE)
    q_tbl.add_column("Metric", style="bold")
    q_tbl.add_column("Value", justify="right")
    q_tbl.add_column("Score", justify="right")
    q_tbl.add_column("Max", justify="right")
    q_tbl.add_row("YTD Return", fmt(score["ytd"]), "", "")
    q_tbl.add_row("SPY Correlation", f"{score['correlation']:.3f}", "", "")
    q_tbl.add_row("Max Drawdown", f"{score['max_dd']:.1f}%", "", "")
    q_tbl.add_row("Trend Strength", "", f"{score['trend']:.1f}", "25")
    q_tbl.add_row("Independence", "", f"{score['independence']:.1f}", "25")
    q_tbl.add_row("Relative Strength", "", f"{score['rel_strength']:.1f}", "25")
    q_tbl.add_row("Stability", "", f"{score['stability']:.1f}", "25")
    q_tbl.add_row("[bold]COMPOSITE[/bold]", "", f"[bold]{score['total']:.1f}[/bold]", "[bold]100[/bold]")
    console.print(q_tbl)

    if score["correlation"] < 0.5:
        console.print("[green]✅ LOW SPY correlation → Physical supply/demand driver (true bottleneck signal)[/green]")
    elif score["correlation"] < 0.75:
        console.print("[yellow]⚠️ MODERATE SPY correlation → Mixed drivers[/yellow]")
    else:
        console.print("[red]❌ HIGH SPY correlation → Beta play, NOT a true bottleneck[/red]")

    console.print("\n[bold cyan]═══ FUNDAMENTALS ═══[/bold cyan]")
    fund = fetch_fundamentals(ticker)
    if fund:
        f_tbl = Table(box=box.SIMPLE)
        f_tbl.add_column("Metric", style="bold")
        f_tbl.add_column("Value")
        f_tbl.add_row("Company", fund.get("name", "—"))
        f_tbl.add_row("Sector", fund.get("sector", "—"))
        f_tbl.add_row("Industry", fund.get("industry", "—"))
        f_tbl.add_row("Market Cap", fmt_mc(fund.get("market_cap")))
        gm = fund.get("gross_margin")
        f_tbl.add_row("Gross Margin", f"{gm*100:.1f}%" if gm else "—")
        om = fund.get("oper_margin")
        f_tbl.add_row("Operating Margin", f"{om*100:.1f}%" if om else "—")
        f_tbl.add_row("Forward P/E", f"{fund['forward_pe']:.1f}" if fund.get("forward_pe") else "—")
        console.print(f_tbl)
        if gm and gm > 0.40:
            console.print(f"[green]✅ Gross margin {gm*100:.1f}% > 40% → Pricing power (bottleneck signal)[/green]")

    console.print("\n[bold cyan]═══ SAF CROSS-REFERENCE ═══[/bold cyan]")
    found_in = [b for b, ts in SAF_BASKET_TICKERS.items() if ticker in ts]
    if found_in:
        console.print(f"[yellow]⚠️ {ticker} already in SAF baskets:[/yellow]")
        for b in found_in:
            console.print(f"   • {b}")
    else:
        console.print(f"[green]✅ {ticker} NOT yet in SAF → Potential new addition[/green]")

    if AI_ENABLED:
        console.print("\n[bold magenta]═══ 🤖 AI FUND MANAGER VERDICT ═══[/bold magenta]")
        with console.status("[magenta]AI synthesizing report...[/magenta]"):
            ai_prompt = (
                f"Ticker: {ticker}\n"
                f"QUANTITATIVE SCORE:\n{json.dumps(score, indent=2)}\n"
                f"FUNDAMENTALS:\n{json.dumps(fund, indent=2, default=str)}\n"
                f"INVESTABILITY: {inv['status']} ({inv['passed']}/4 checks passed)\n"
                f"ALREADY IN SAF: {'Yes - ' + ', '.join(found_in) if found_in else 'No'}\n"
                f"Write the final investment memo."
            )
            memo = llm(DEEP_REPORT_SYS, ai_prompt, temperature=0.4)
            if memo:
                console.print(Panel(memo, title=f"🤖 AI Verdict — {ticker}",
                                    border_style="magenta", box=box.ROUNDED))
                save_ai_report("deep_analysis", {"ticker": ticker, "memo": memo, "score": score})

# ╔═══════════════════════════════════════════════════════════╗
# ║  🧮 OPTION 3: INTERACTIVE BOTTLENECK SCORING              ║
# ╚═══════════════════════════════════════════════════════════╝
BOTTLENECK_CRITERIA = [
    {"name": "Market Concentration",
     "question": "Do ≤5 companies control >70% of supply?",
     "hint": "5=Dominant oligopoly | 3=Moderate | 1=Fragmented"},
    {"name": "Substitutability",
     "question": "Is there NO viable alternative material/process?",
     "hint": "5=Zero substitutes | 3=Some alternatives | 1=Many substitutes"},
    {"name": "Capital Intensity",
     "question": "Would it take >$1B and >5 years to build a competitor?",
     "hint": "5=Massive barriers | 3=Moderate | 1=Easy to enter"},
    {"name": "Regulatory Moat",
     "question": "Are there permits/patents blocking new entrants?",
     "hint": "5=Strong moat | 3=Some protection | 1=No barriers"},
    {"name": "Demand Inelasticity",
     "question": "Will buyers pay ANY price because there's no substitute?",
     "hint": "5=Must-have | 3=Important | 1=Nice-to-have"},
    {"name": "Cross-Sector Demand",
     "question": "Needed by MULTIPLE growing industries simultaneously?",
     "hint": "5=Many sectors | 3=Few sectors | 1=Single sector"},
]

def interactive_bottleneck_score():
    ticker = Prompt.ask("\n[bold]Enter ticker to score[/bold]", default="MKSI").upper().strip()

    ai_scores = None
    if AI_ENABLED:
        console.print(f"\n[dim]🤖 AI pre-scoring {ticker}...[/dim]")
        with console.status("[magenta]AI researching bottleneck profile...[/magenta]"):
            ai_out = llm(BOTTLENECK_ANALYST_SYS,
                         f"Ticker: {ticker}\nAnalyze this company as a potential Shadow Alpha bottleneck.",
                         temperature=0.3, force_json=True)
            ai_scores = extract_json(ai_out)

    console.print(Panel.fit(
        f"[bold magenta]🧮 BOTTLENECK SCORING — {ticker}[/bold magenta]\n"
        f"[dim]Score each criterion 1-5. Pass: ≥{BOTTLENECK_PASS}/30[/dim]",
        box=box.DOUBLE, border_style="magenta",
    ))

    if ai_scores and "scores" in ai_scores:
        console.print(f"\n[bold yellow]🤖 AI SUGGESTION:[/bold yellow]")
        console.print(f"   Company: {ai_scores.get('company_name', '?')}")
        console.print(f"   What: {ai_scores.get('what_they_do', '?')}")
        ai_tbl = Table(box=box.SIMPLE, show_edge=False)
        ai_tbl.add_column("Criterion", style="bold")
        ai_tbl.add_column("AI Score", justify="center")
        ai_tbl.add_column("AI Reasoning", overflow="fold")
        s = ai_scores.get("scores", {})
        r = ai_scores.get("reasoning", {})
        for c in BOTTLENECK_CRITERIA:
            key = c["name"].lower().replace(" ", "_")
            ai_tbl.add_row(c["name"],
                           str(s.get(key, "?")),
                           str(r.get(key, ""))[:80])
        ai_tbl.add_row("[bold]TOTAL[/bold]", f"[bold]{ai_scores.get('total', '?')}[/bold]", "")
        console.print(ai_tbl)
        console.print(f"   AI Verdict: [bold]{ai_scores.get('verdict', '?')}[/bold]")
        console.print(f"   Key Risk: {ai_scores.get('key_risk', '?')}")
        save_ai_report("bottleneck_analysis", {"ticker": ticker, "ai": ai_scores})

    console.print(f"\n[bold]Now score it yourself (AI scores shown as reference):[/bold]")
    scores = {}
    total = 0
    for i, c in enumerate(BOTTLENECK_CRITERIA, 1):
        console.print(f"\n[bold cyan]{i}/6: {c['name']}[/bold cyan]")
        console.print(f"  [dim]{c['question']}[/dim]")
        console.print(f"  [dim]{c['hint']}[/dim]")
        while True:
            try:
                s = IntPrompt.ask(f"  Score for {c['name']}", default=3)
                if 1 <= s <= 5:
                    scores[c["name"]] = s
                    total += s
                    break
                console.print("[red]Enter 1-5.[/red]")
            except Exception:
                console.print("[red]Invalid input.[/red]")

    console.print("\n[bold cyan]═══ YOUR SCORING ═══[/bold cyan]")
    r_tbl = Table(box=box.SIMPLE_HEAVY)
    r_tbl.add_column("Criterion", style="bold")
    r_tbl.add_column("You", justify="center", width=5)
    r_tbl.add_column("AI", justify="center", width=5)
    r_tbl.add_column("Bar", width=25)
    ai_s = ai_scores.get("scores", {}) if ai_scores else {}
    for c in BOTTLENECK_CRITERIA:
        key = c["name"].lower().replace(" ", "_")
        your_s = scores[c["name"]]
        ai_val = ai_s.get(key, "—")
        bar = "█" * your_s + "░" * (5 - your_s)
        color = "green" if your_s >= 4 else ("yellow" if your_s >= 3 else "red")
        r_tbl.add_row(c["name"], f"[{color}]{your_s}[/{color}]", str(ai_val),
                      f"[{color}]{bar}[/{color}]")
    r_tbl.add_row("[bold]TOTAL[/bold]", f"[bold]{total}[/bold]",
                  str(ai_scores.get("total", "—")) if ai_scores else "—", "out of 30")
    console.print(r_tbl)

    if total >= BOTTLENECK_PASS:
        console.print(f"\n[bold green]🎯 TRUE SHADOW ALPHA BOTTLENECK ({total}/30 ≥ {BOTTLENECK_PASS})[/bold green]")
        console.print("[green]   → Qualifies for SAF integration.[/green]")
    else:
        console.print(f"\n[red]❌ Does NOT qualify ({total}/30 < {BOTTLENECK_PASS})[/red]")

    log_entry = {"ticker": ticker, "date": datetime.now().isoformat(),
                 "your_scores": scores, "your_total": total,
                 "ai_scores": ai_scores, "qualified": total >= BOTTLENECK_PASS}
    save_ai_report("bottleneck_scoring", log_entry)

# ╔═══════════════════════════════════════════════════════════╗
# ║  📋 OPTION 4: VIEW UNIVERSE                               ║
# ╚═══════════════════════════════════════════════════════════╝
def view_universe(universe):
    console.print(Panel.fit("[bold cyan]🌍 CANDIDATE UNIVERSE[/bold cyan]",
                            box=box.DOUBLE, border_style="cyan"))
    total = 0
    for sector, tickers in universe.items():
        console.print(f"\n[bold yellow]{sector}[/bold yellow] [dim]({len(tickers)})[/dim]")
        console.print("  " + ", ".join(tickers))
        total += len(tickers)
    console.print(f"\n[bold]Total: {total} candidates[/bold]")

# ╔═══════════════════════════════════════════════════════════╗
# ║  ➕ OPTION 5: ADD TICKER                                   ║
# ╚═══════════════════════════════════════════════════════════╝
def add_to_universe(universe):
    console.print("\n[bold]Available sectors:[/bold]")
    sectors = list(universe.keys())
    for i, s in enumerate(sectors, 1):
        console.print(f"  [{i}] {s}")
    console.print(f"  [{len(sectors)+1}] ➕ Create new sector")
    choice = Prompt.ask("\n[bold]Select sector[/bold]", default="1")
    try:
        idx = int(choice) - 1
        if idx == len(sectors):
            new_sector = Prompt.ask("[bold]New sector name[/bold]")
            universe[new_sector] = []
            sector = new_sector
        elif 0 <= idx < len(sectors):
            sector = sectors[idx]
        else:
            return
    except ValueError:
        return
    ticker = Prompt.ask("[bold]Ticker to add[/bold]").upper().strip()
    if ticker not in universe[sector]:
        universe[sector].append(ticker)
        save_universe(universe)
        console.print(f"[green]✅ Added {ticker} to '{sector}'.[/green]")
    else:
        console.print(f"[yellow]⚠️ {ticker} already in '{sector}'.[/yellow]")

# ╔═══════════════════════════════════════════════════════════╗
# ║  🗑️  OPTION 6: REMOVE TICKER                              ║
# ╚═══════════════════════════════════════════════════════════╝
def remove_from_universe(universe):
    ticker = Prompt.ask("\n[bold]Ticker to remove[/bold]").upper().strip()
    removed = False
    for sector in universe:
        if ticker in universe[sector]:
            universe[sector].remove(ticker)
            removed = True
            console.print(f"[green]✅ Removed {ticker} from '{sector}'.[/green]")
    if not removed:
        console.print(f"[red]❌ {ticker} not found.[/red]")
    else:
        save_universe(universe)

# ╔═══════════════════════════════════════════════════════════╗
# ║  📊 OPTION 7: COMPARE TOP CANDIDATES                      ║
# ╚═══════════════════════════════════════════════════════════╝
def compare_candidates():
    if not os.path.exists(EXPORT_FILE):
        console.print("[yellow]⚠️ No results. Run Full Screen first.[/yellow]")
        return
    df = pd.read_csv(EXPORT_FILE)
    if df.empty:
        return
    top_n = IntPrompt.ask("\n[bold]How many top candidates?[/bold]", default=10)
    top = df.head(top_n)
    tbl = Table(title=f"🏆 Top {top_n} Candidates", box=box.HEAVY_HEAD, show_lines=True)
    tbl.add_column("Ticker", style="bold cyan")
    tbl.add_column("Price", justify="right")
    tbl.add_column("YTD", justify="right")
    tbl.add_column("Corr", justify="right")
    tbl.add_column("Score", justify="right", style="bold")
    for _, row in top.iterrows():
        t = row["ticker"]
        in_saf = any(t in ts for ts in SAF_BASKET_TICKERS.values())
        tag = " [yellow]●[/yellow]" if in_saf else ""
        tbl.add_row(t + tag, f"${row['price']:.2f}", f"{row['ytd']:+.2f}%",
                    f"{row['correlation']:.2f}", f"[bold]{row['total']:.1f}[/bold]")
    console.print(tbl)
    console.print("[dim]● = Already in SAF baskets[/dim]")

# ╔═══════════════════════════════════════════════════════════╗
# ║  🤖 OPTION 8: AI SUPPLY CHAIN DISCOVERY (Stages 1-2)      ║
# ╚═══════════════════════════════════════════════════════════╝
def ai_supply_chain_discovery():
    """AI maps a supply chain for a given trend and finds bottlenecks."""
    if not AI_ENABLED:
        console.print("[red]❌ AI not available. Set GROQ_API_KEY.[/red]")
        return
    trend = Prompt.ask(
        "\n[bold]Enter a demand signal / trend[/bold]",
        default="solid-state batteries for electric vehicles",
    )
    console.print(Panel.fit(
        f"[bold magenta]🤖 AI SUPPLY CHAIN DISCOVERY[/bold magenta]\n"
        f"[dim]Trend: {trend}[/dim]",
        box=box.DOUBLE, border_style="magenta",
    ))
    with console.status("[magenta]AI mapping supply chain & identifying bottlenecks...[/magenta]"):
        out = llm(SUPPLY_CHAIN_SYS,
                  f"Map the supply chain for this trend and find the Shadow Alpha bottlenecks:\nTREND: {trend}",
                  temperature=0.5, force_json=True)
    result = extract_json(out)
    if not result:
        console.print("[red]❌ AI returned invalid output. Try again.[/red]")
        if out:
            console.print(Panel(out, title="Raw AI Output", border_style="yellow"))
        return

    console.print("\n[bold cyan]═══ SUPPLY CHAIN MAP ═══[/bold cyan]")
    for i, layer in enumerate(result.get("supply_chain", []), 1):
        console.print(f"  {'  ' * (i-1)}└─ {layer}")

    console.print("\n[bold cyan]═══ BOTTLENECKS IDENTIFIED ═══[/bold cyan]")
    for i, b in enumerate(result.get("bottlenecks", []), 1):
        b_tbl = Table(title=f"Bottleneck #{i}: {b.get('name', '?')}", box=box.ROUNDED)
        b_tbl.add_column("Attribute", style="bold")
        b_tbl.add_column("Value", overflow="fold")
        b_tbl.add_row("Why Bottleneck", b.get("why_bottleneck", "?"))
        b_tbl.add_row("Tickers", ", ".join(b.get("tickers", [])))
        b_tbl.add_row("Market Concentration", b.get("market_concentration", "?"))
        b_tbl.add_row("Substitutability", b.get("substitutability", "?"))
        console.print(b_tbl)

    top = result.get("top_pick", "?")
    console.print(f"\n[bold green]🎯 AI TOP PICK: {top}[/bold green]")
    console.print(f"[dim]Thesis: {result.get('thesis_summary', '')}[/dim]")

    all_tickers = set()
    for b in result.get("bottlenecks", []):
        all_tickers.update(b.get("tickers", []))
    if all_tickers and Confirm.ask(f"\n[bold]Add {len(all_tickers)} candidates to universe?[/bold]", default=True):
        universe = load_universe()
        sector_name = f"AI: {trend[:30]}"
        if sector_name not in universe:
            universe[sector_name] = []
        for t in all_tickers:
            if t not in universe[sector_name]:
                universe[sector_name].append(t.upper().strip())
        save_universe(universe)
        console.print(f"[green]✅ Added to sector '{sector_name}'[/green]")
    save_ai_report("supply_chain_discovery", {"trend": trend, "result": result})

# ╔═══════════════════════════════════════════════════════════╗
# ║  🤖 OPTION 9: AI BOTTLENECK ANALYSIS (Stage 3)            ║
# ╚═══════════════════════════════════════════════════════════╝
def ai_bottleneck_analysis():
    """AI researches a ticker and scores the 6-criteria rubric."""
    if not AI_ENABLED:
        console.print("[red]❌ AI not available. Set GROQ_API_KEY.[/red]")
        return
    ticker = Prompt.ask("\n[bold]Enter ticker for AI bottleneck analysis[/bold]", default="MKSI").upper().strip()
    console.print(Panel.fit(
        f"[bold magenta]🤖 AI BOTTLENECK ANALYSIS — {ticker}[/bold magenta]\n"
        f"[dim]6-Criteria Shadow Alpha Rubric[/dim]",
        box=box.DOUBLE, border_style="magenta",
    ))
    with console.status("[magenta]AI researching and scoring...[/magenta]"):
        out = llm(BOTTLENECK_ANALYST_SYS,
                  f"Ticker: {ticker}\nPerform full Shadow Alpha bottleneck analysis.",
                  temperature=0.3, force_json=True)
    result = extract_json(out)
    if not result:
        console.print("[red]❌ AI returned invalid output.[/red]")
        if out:
            console.print(Panel(out, title="Raw Output", border_style="yellow"))
        return

    console.print(f"\n[bold]Company:[/bold] {result.get('company_name', '?')}")
    console.print(f"[bold]What they do:[/bold] {result.get('what_they_do', '?')}")
    scores = result.get("scores", {})
    reasoning = result.get("reasoning", {})
    total = result.get("total", sum(scores.values()))
    s_tbl = Table(box=box.SIMPLE_HEAVY)
    s_tbl.add_column("Criterion", style="bold")
    s_tbl.add_column("Score", justify="center", width=6)
    s_tbl.add_column("Bar", width=20)
    s_tbl.add_column("AI Reasoning", overflow="fold")
    for c in BOTTLENECK_CRITERIA:
        key = c["name"].lower().replace(" ", "_")
        s = scores.get(key, 0)
        bar = "█" * s + "░" * (5 - s)
        color = "green" if s >= 4 else ("yellow" if s >= 3 else "red")
        s_tbl.add_row(c["name"], f"[{color}]{s}[/{color}]",
                      f"[{color}]{bar}[/{color}]",
                      str(reasoning.get(key, ""))[:100])
    s_tbl.add_row("[bold]TOTAL[/bold]", f"[bold]{total}[/bold]", "", f"Pass threshold: {BOTTLENECK_PASS}")
    console.print(s_tbl)

    verdict = result.get("verdict", "?")
    v_color = "green" if "TRUE" in verdict.upper() else "red"
    console.print(f"\n[bold {v_color}]🏆 AI VERDICT: {verdict}[/bold {v_color}]")
    console.print(f"[dim]Key Risk: {result.get('key_risk', '?')}[/dim]")
    if total >= BOTTLENECK_PASS:
        console.print(f"\n[bold green]✅ PASSES ({total}/30 ≥ {BOTTLENECK_PASS}) → Run Deep Analysis (Option 2) next[/bold green]")
    else:
        console.print(f"\n[red]❌ BELOW THRESHOLD ({total}/30 < {BOTTLENECK_PASS})[/red]")
    save_ai_report("ai_bottleneck_analysis", {"ticker": ticker, "result": result})

# ╔═══════════════════════════════════════════════════════════╗
# ║  🤖 OPTION 10: AI DEEP REPORT (Stages 4-6 Combined)       ║
# ╚═══════════════════════════════════════════════════════════╝
def ai_deep_report():
    """Full AI-synthesized investment memo combining quant + fundamentals."""
    if not AI_ENABLED:
        console.print("[red]❌ AI not available. Set GROQ_API_KEY.[/red]")
        return
    ticker = Prompt.ask("\n[bold]Enter ticker for AI Deep Report[/bold]", default="MKSI").upper().strip()
    console.print(f"\n[dim]🤖 Generating AI Deep Report for {ticker}...[/dim]")

    pdf = fetch_prices([ticker, BENCHMARK])
    if pdf.empty or ticker not in pdf.columns:
        console.print(f"[red]❌ No price data for {ticker}.[/red]")
        return
    spy = pdf[BENCHMARK]
    score = compute_shadow_alpha_score(pdf, ticker, spy)
    fund = fetch_fundamentals(ticker)
    inv = check_investability(ticker)
    found_in = [b for b, ts in SAF_BASKET_TICKERS.items() if ticker in ts]

    with console.status("[magenta]AI writing investment memo...[/magenta]"):
        ai_prompt = (
            f"Ticker: {ticker}\n"
            f"QUANTITATIVE SHADOW ALPHA SCORE:\n{json.dumps(score, indent=2)}\n"
            f"FUNDAMENTALS:\n{json.dumps(fund, indent=2, default=str)}\n"
            f"INVESTABILITY: {inv['status']} ({inv['passed']}/4 checks)\n"
            f"Market Cap: {fmt_mc(inv['market_cap'])}\n"
            f"Avg Volume: {inv['avg_volume']:,}\n"
            f"ALREADY IN SAF BASKETS: {'Yes: ' + ', '.join(found_in) if found_in else 'No'}\n"
            f"Write the final Shadow Alpha investment memo. Be decisive."
        )
        memo = llm(DEEP_REPORT_SYS, ai_prompt, temperature=0.4)
        if memo:
            console.print(Panel(memo, title=f"🤖 FUND MANAGER MEMO — {ticker}",
                                border_style="magenta", box=box.DOUBLE))
            save_ai_report("ai_deep_report", {"ticker": ticker, "memo": memo,
                                              "score": score, "fundamentals": fund})
        else:
            console.print("[red]❌ AI could not generate report.[/red]")

# ╔═══════════════════════════════════════════════════════════╗
# ║  🧪 SYSTEM DIAGNOSTICS                                    ║
# ╚═══════════════════════════════════════════════════════════╝
def run_diagnostics():
    console.print(Panel.fit(
        "[bold cyan]🧪 SAF SCREENER SYSTEM DIAGNOSTICS[/bold cyan]",
        box=box.DOUBLE, border_style="cyan"))
    results = []

    deps_ok = True
    dep_detail = []
    for mod in ("yfinance", "pandas", "numpy", "rich"):
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
            h = yf.Ticker(BENCHMARK).history(period="5d")
            if h is not None and not h.empty:
                yf_ok = True
                yf_detail = f"{BENCHMARK} close ${h['Close'].iloc[-1]:.2f} ({len(h)} bars)"
            else:
                yf_detail = "empty history"
        except Exception as e:
            yf_detail = f"error: {str(e)[:60]}"
    results.append(("Yahoo Finance data feed", yf_ok, yf_detail))

    cache_ok = True
    try:
        clear_yf_cache()
        _ = yf.Ticker("AAPL").history(period="1d")
    except Exception:
        cache_ok = False
    results.append(("yfinance cache health", cache_ok,
                    "cleared & re-writable" if cache_ok else "still failing — check disk permissions"))

    key_ok = bool(GROQ_API_KEY)
    results.append(("GROQ_API_KEY present", key_ok,
                    "set" if key_ok else "NOT set — AI features disabled"))

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
    fio_ok = writable(EXPORT_FILE) and writable(UNIVERSE_FILE) and writable(AI_LOG_FILE)
    results.append(("File write permissions", fio_ok,
                    "export/universe/log writable" if fio_ok else "check directory permissions"))

    uni = load_universe()
    ucand = get_all_candidates(uni)
    results.append(("Candidate universe", len(ucand) > 0,
                    f"{len(uni)} sectors, {len(ucand)} candidates"))

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
    if not yf_ok and net:
        console.print("[yellow]💡 Data feed failing despite internet → run terminal cache-clear, then retry.[/yellow]")
    if not key_ok:
        console.print("[yellow]💡 Set your key:  export GROQ_API_KEY='gsk_...'[/yellow]")
    if AI_ENABLED and passed == len(results):
        console.print("[bold green]🎉 Full pipeline operational — AI + data + files all green.[/bold green]")

# ╔═══════════════════════════════════════════════════════════╗
# ║  🎛️  MAIN MENU                                             ║
# ╚═══════════════════════════════════════════════════════════╝
def main_menu():
    ai_status = "[green]ENABLED[/green]" if AI_ENABLED else "[red]DISABLED[/red]"
    console.print(Panel.fit(
        "[bold magenta]🔍 SAF SCREENER + AI[/bold magenta]\n"
        f"[dim]Shadow Alpha Asset Discovery Engine · AI: {ai_status}[/dim]",
        box=box.DOUBLE, border_style="magenta",
    ))

    if not internet_ok():
        console.print("[bold red]⚠️ No internet detected — data features will fail. Run [11] Diagnostics.[/bold red]")

    universe = load_universe()

    while True:
        console.print("\n[bold cyan]╔════════════ SAF SCREENER + AI ════════════╗[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [bold]QUANTITATIVE (No AI needed)[/bold]             [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [1] 📡 Full Screen (All Candidates)       [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [2] 🔍 Deep Analysis (Single Ticker)      [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [3] 🧮 Interactive Bottleneck Scoring     [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [4] 📋 View Candidate Universe            [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [5] ➕ Add Ticker to Universe             [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [6] 🗑️  Remove Ticker from Universe        [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [7] 📊 Compare Top Candidates             [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [bold]AI-POWERED (Requires GROQ_API_KEY)[/bold]      [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [8] 🤖 AI Supply Chain Discovery          [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [9] 🤖 AI Bottleneck Analysis             [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [10] 🤖 AI Deep Report (Full Memo)        [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [bold]SYSTEM[/bold]                                  [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [11] 🧪 Run Diagnostics (Health Check)    [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [0] 🚪 Exit                               [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]╚════════════════════════════════════════════╝[/bold cyan]")
        choice = Prompt.ask(
            "\n[bold]Select option[/bold]",
            choices=["0","1","2","3","4","5","6","7","8","9","10","11"],
            default="1",
        )
        if choice == "1":
            run_full_screen(universe)
        elif choice == "2":
            deep_analysis(universe)
        elif choice == "3":
            interactive_bottleneck_score()
        elif choice == "4":
            view_universe(universe)
        elif choice == "5":
            add_to_universe(universe)
        elif choice == "6":
            remove_from_universe(universe)
        elif choice == "7":
            compare_candidates()
        elif choice == "8":
            ai_supply_chain_discovery()
        elif choice == "9":
            ai_bottleneck_analysis()
        elif choice == "10":
            ai_deep_report()
        elif choice == "11":
            run_diagnostics()
        elif choice == "0":
            console.print("\n[bold red]👋 Goodbye, Shadow Alpha Analyst.[/bold red]")
            break
        if choice != "0":
            console.input("\n[dim]Press Enter to return to menu...[/dim]")

if __name__ == "__main__":
    main_menu()