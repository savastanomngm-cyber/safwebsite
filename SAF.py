#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
🦅 SKIA ALPHA FUND v3 — Unified AI Pipeline  (FULLY PATCHED)
═══════════════════════════════════════════════════════════════
Combines: sfa2.py (baskets) + TradingAgents.py (5-stage agents)
          + AI correlation + auto-add to SAF

PATCHES APPLIED:
  • Analyst Team upgraded 4 → 5 agents (added GEOPOLITICAL / Shadow Supply Chain)
  • New SENT_SYS, NEWS_SYS, GEOPOL_SYS prompts; TRADER_SYS interprets risk-premium scoring
  • Self-healing data layer (cache-clear + retry + single-ticker fallback)
  • New System Diagnostics health check (menu option 9)

Install:  pip install yfinance pandas rich numpy openai
Setup:    export GROQ_API_KEY='gsk_YOUR_KEY_HERE'
Run:      python SAF.py
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
from rich import box
from concurrent.futures import ThreadPoolExecutor
import os, sys, json, re, time, shutil, platform, tempfile
import urllib.request

console = Console()

# ╔═══════════════════════════════════════════════════════════╗
# ║  ⚙️  GLOBAL CONFIG                                         ║
# ╚═══════════════════════════════════════════════════════════╝
LOOKBACK_DAYS    = 400
BENCHMARK        = "SPY"
EXPORT_FILE      = "skia_alpha_tickers.csv"
SAVINGS_FILE     = "saf_baskets_custom.json"   # Persist added assets
MEMORY_FILE      = "trading_memory.jsonl"
DEBATE_ROUNDS    = 2
MAX_RETRIES      = 3
RETRY_DELAY      = 5

# ╔═══════════════════════════════════════════════════════════╗
# ║  🤖 LLM LAYER (Same as TradingAgents.py — Groq)           ║
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
        console.print("[yellow]⚠️ pip install openai — AI features disabled.[/yellow]")
else:
    console.print("[yellow]⚠️ GROQ_API_KEY not set. AI pipeline disabled.[/yellow]")
    console.print("[yellow]   Quantitative tracking (options 1-6) still works.[/yellow]")

MODEL_DEEP   = os.getenv("TA_DEEP", "openai/gpt-oss-120b")
MODEL_FAST   = os.getenv("TA_FAST", "openai/gpt-oss-20b")
MODEL_BACKUP = "qwen/qwen3.6-27b"

def llm(system, user, model=None, temperature=0.7, force_json=False):
    """Same LLM function as TradingAgents.py."""
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
                console.print(f"[yellow]Rate limited. Waiting {wait}s...[/yellow]")
                time.sleep(wait)
                continue
            if ("404" in err or "not_found" in err) and target_model != MODEL_BACKUP:
                target_model = MODEL_BACKUP
                continue
            console.print(f"[red]LLM Error: {e}[/red]")
            return ""
    return ""

def extract_json(text):
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
# ║  🧺 ALL SAF BASKETS                                       ║
# ╚═══════════════════════════════════════════════════════════╝
BASKETS = {
    "🧠 JIM SIMONS (RenTech)": {
        "NVDA": 1.0, "PLTR": 1.0, "UTHR": 1.0, "META": 1.0,
        "AAPL": 1.0, "VRSN": 1.0, "EXEL": 1.0, "KGC": 1.0,
        "SPOT": 1.0, "NFLX": 1.0, "HOOD": 1.0, "GOOGL": 1.0,
    },
    "🏛️ PELOSI TRACKER": {
        "NVDA": 1.0, "MSFT": 1.0, "AAPL": 1.0, "GOOGL": 1.0,
        "AMZN": 1.0, "PANW": 1.0, "CRM": 1.0, "TEM": 1.0,
    },
    "🐻 BURRY TRACKER": {
        "AMZN": 1.0, "VIST": 1.0, "VST": 1.0, "AEP": 1.0, "DTE": 1.0,
    },
    "🦉 BUFFETT TRACKER": {
        "AAPL": 1.0, "BAC": 1.0, "CVX": 1.0, "OXY": 1.0,
        "KO": 1.0, "AXP": 1.0, "MMM": 1.0,
    },
    "🦁 ACKMAN TRACKER": {
        "GOOGL": 1.0, "META": 1.0, "CMG": 1.0, "NFLX": 1.0,
        "QSR": 1.0, "HLT": 1.0, "UBER": 1.0,
    },
    "🧠 DRUCKENMILLER TRACKER": {
        "NVDA": 1.0, "MSFT": 1.0, "LLY": 1.0, "UNH": 1.0,
        "VRT": 1.0, "GE": 1.0, "VRTX": 1.0,
    },
    "🏭 CITADEL TRACKER": {
        "SPY": 1.0, "QQQ": 1.0, "IWM": 1.0, "AAPL": 1.0, "MSFT": 1.0,
    },
    "🎯 INVERSE CRAMER (Bear ETFs)": {"SQQQ": 1.0, "SH": 1.0, "SPXU": 1.0},
    "🤖 AI WORLD WAR III": {
        "LMT": 1.0, "RTX": 1.0, "NOC": 1.0, "PLTR": 1.0,
        "NVDA": 1.0, "GD": 1.0, "SMH": 1.0,
    },
    "🧬 GENOME (Biotech Innovation)": {
        "AMGN": 2.0, "GILD": 2.0, "VRTX": 2.0, "REGN": 2.0,
        "MRNA": 1.5, "BNTX": 1.5, "ALNY": 1.5, "IONS": 1.5,
        "CRSP": 1.0, "NTLA": 1.0, "BEAM": 1.0, "EDIT": 1.0,
    },
    "⛏️ TERRA (Global Commodities)": {
        "XOM": 2.0, "CVX": 2.0, "OXY": 1.5, "FCX": 1.5, "NEM": 1.5,
        "GLD": 1.5, "MOS": 1.0, "CF": 1.0, "CORN": 1.0, "PHO": 1.5, "AWK": 1.0,
    },
    "🤖 NEXUS (AI Tech Infrastructure)": {
        "NVDA": 3.0, "AMD": 2.0, "AVGO": 2.0, "TSM": 2.0,
        "MSFT": 2.0, "GOOGL": 2.0, "AMZN": 2.0, "META": 1.5,
        "PLTR": 1.5, "NOW": 1.0, "SNOW": 1.0, "VRT": 1.0,
    },
    "💧 WATER (AI Cooling + Scarcity)": {
        "PHO": 1.0, "FIW": 1.0, "AWK": 1.0, "XYL": 1.0, "ECL": 1.0, "VRT": 1.0,
    },
    "⚙️ FLUID & MEMBRANE BOTTLENECKS": {
        "FLS": 2.0, "ROP": 2.0, "DD": 1.5, "XYL": 1.5, "VRT": 1.5,
    },
    "⚔️ AGRO-CHEM MONOPOLIES (War Squeeze)": {
        "NTR": 2.5, "MOS": 2.0, "CF": 2.0, "YARIY": 1.5, "FMC": 1.0,
    },
    "🌾 CROP YIELD SHORTAGE": {"DBA": 3.0, "WEAT": 2.0, "CORN": 2.0},
    "🚢 BULK CHEMICAL FREIGHT": {"ZIM": 2.0, "FLNG": 2.0, "INSW": 1.5},
    "🕵️ BORDER & AML TECH": {"PLTR": 2.0, "RTX": 1.5, "NICE": 1.5, "GD": 1.0},
    "🥇 GOLD MINERS (Traditional)": {
        "NEM": 2.0, "GOLD": 2.0, "AEM": 2.0, "GDX": 1.5, "GDXJ": 1.0,
    },
    "👑 GOLD ROYALTY (Pick & Shovel)": {"FNV": 2.5, "WPM": 2.5, "RGLD": 2.0, "OR": 1.5},
    "♻️ E-WASTE & URBAN MINING": {
        "UMICY": 2.5, "NDA.DE": 2.5, "BOL.ST": 2.0, "GLNCY": 1.5,
        "5711.T": 2.0, "5713.T": 1.5, "5714.T": 1.5, "5857.T": 1.0,
        "SMSMY": 1.5, "NUE": 1.0, "CLH": 1.0, "JMAT.L": 1.5,
        "TOM.OL": 1.0, "ANDR.VI": 1.0, "AREC": 0.5,
    },
    "🏦 GOLD BULLION & ETFs": {"GLD": 3.0, "IAU": 2.0, "PHYS": 1.5, "RING": 1.0},
    "🔧 MINING CAPEX (Equipment)": {"CAT": 2.5, "SDVKY": 2.0, "EPI-B.ST": 1.5, "FLIDY": 1.0},
    "🌿 EU CANNABIS INFRASTRUCTURE": {
        "MTRS.ST": 2.5, "TT": 1.5, "LIGHT.AS": 2.0, "SRT.DE": 2.0,
        "LIN": 1.5, "AI.PA": 1.5, "SAP.DE": 1.5, "ZBRA": 1.0,
    },
    "🦷 DENTAL DISRUPTION (Watch to Short)": {
        "NVST": 2.0, "XRAY": 2.0, "HSIC": 1.5, "IDXX": 1.0,
    },
    "🛡️ DRONES & EDGE AI (New Primes)": {"AVAV": 2.5, "KTOS": 2.0, "RCAT": 1.0},
    "🧠 EDGE AI & SENSORS": {"AMBA": 2.0, "TDY": 2.0, "MRCY": 1.5},
    "🛡️ C-UAS & AI SURVEILLANCE": {"AXON": 2.5, "PLTR": 2.5, "LHX": 1.5},
    "🧲 RARE EARTH (Drone Motors)": {"MP": 3.0},
    "⚛️ QUANTUM BOTTLENECKS": {
        "MKSI": 3.0, "COHR": 2.5, "OXIG.L": 2.0, "PFV.DE": 2.0, "LITE": 1.5,
    },
    "🔦 SENSORS & PHOTODETECTORS": {
        "6965.T": 3.0, "TDY": 2.0, "AMSYF": 1.5, "IPGP": 1.5, "LASR": 1.0,
    },
    "🎲 PURE-PLAY QUANTUM (High Risk)": {
        "IONQ": 2.5, "RGTI": 1.5, "QBTS": 1.5, "QUBT": 0.5, "ARQQ": 0.5,
    },
    "🏛️ QUANTUM GIANTS (Safe Anchors)": {"IBM": 2.0, "GOOGL": 2.0, "HON": 2.0, "MSFT": 1.5},
    "❄️ HELIUM & CRYOGENIC GASES": {"LIN": 3.0, "APD": 2.0},
    "📡 RF & CONTROL ELECTRONICS": {"KEYS": 2.0, "ADI": 2.0, "AMD": 1.5},
    "🏭 BIOTECH FOUNDRIES (CDMOs)": {"TMO": 3.0, "DHR": 2.5, "LONN.SW": 2.0, "BIO": 1.5},
    "❄️ COLD CHAIN & GLASS": {"CYRX": 2.5, "SHTPY": 2.0, "GLW": 1.5},
    "📋 CROs (Trial Managers)": {"IQV": 2.5, "MEDP": 2.0, "LH": 1.5},
    "🔬 LIFE SCIENCE TOOLS": {"ILMN": 2.0, "WAT": 2.0, "A": 1.5},
    "🔬 PHARMA GLASS MONOPOLY": {"SHTPY": 3.0, "STVN": 2.5, "GXI.DE": 2.0},
    "🪟 SPECIALTY GLASS INFRASTRUCTURE": {"GLW": 3.0, "7741.T": 2.0, "5201.T": 1.5},
    "🧩 VIAL ECOSYSTEM": {"WST": 3.0, "MTD": 1.5},
    "⛏️ BORON RAW MATERIALS": {"RIO": 3.0, "MOS": 1.5, "NTR": 1.5, "LIN": 1.0},
    "📏 SPY (S&P 500)": {"SPY": 1.0},
    "📏 QQQ (Nasdaq 100)": {"QQQ": 1.0},
    "📏 XBI (Biotech ETF)": {"XBI": 1.0},
    "📏 GSG (Commodity ETF)": {"GSG": 1.0},
    "📏 SMH (Chip ETF)": {"SMH": 1.0},
    "📏 GLD (Gold ETF)": {"GLD": 1.0},
}

SECTIONS = {
    "🎮 AUTOPILOT TRACKERS": [
        "🧠 JIM SIMONS (RenTech)", "🏛️ PELOSI TRACKER", "🐻 BURRY TRACKER",
        "🦉 BUFFETT TRACKER", "🦁 ACKMAN TRACKER", "🧠 DRUCKENMILLER TRACKER",
        "🏭 CITADEL TRACKER", "🎯 INVERSE CRAMER (Bear ETFs)", "🤖 AI WORLD WAR III",
    ],
    "📊 CUSTOM INDICES": [
        "🧬 GENOME (Biotech Innovation)", "⛏️ TERRA (Global Commodities)",
        "🤖 NEXUS (AI Tech Infrastructure)",
    ],
    "💧 WATER": ["💧 WATER (AI Cooling + Scarcity)", "⚙️ FLUID & MEMBRANE BOTTLENECKS"],
    "⚔️ FERTILIZER / WAR SQUEEZE": [
        "⚔️ AGRO-CHEM MONOPOLIES (War Squeeze)", "🌾 CROP YIELD SHORTAGE",
        "🚢 BULK CHEMICAL FREIGHT", "🕵️ BORDER & AML TECH",
    ],
    "🥇 GOLD & E-WASTE": [
        "🥇 GOLD MINERS (Traditional)", "👑 GOLD ROYALTY (Pick & Shovel)",
        "♻️ E-WASTE & URBAN MINING", "🏦 GOLD BULLION & ETFs", "🔧 MINING CAPEX (Equipment)",
    ],
    "🌿 EU CANNABIS INFRA": ["🌿 EU CANNABIS INFRASTRUCTURE"],
    "🦷 DENTAL DISRUPTION": ["🦷 DENTAL DISRUPTION (Watch to Short)"],
    "🛡️ NEXT-GEN WARFARE": [
        "🛡️ DRONES & EDGE AI (New Primes)", "🧠 EDGE AI & SENSORS",
        "🛡️ C-UAS & AI SURVEILLANCE", "🧲 RARE EARTH (Drone Motors)",
    ],
    "⚛️ QUANTUM COMPUTING": [
        "⚛️ QUANTUM BOTTLENECKS", "🔦 SENSORS & PHOTODETECTORS",
        "🎲 PURE-PLAY QUANTUM (High Risk)", "🏛️ QUANTUM GIANTS (Safe Anchors)",
        "❄️ HELIUM & CRYOGENIC GASES", "📡 RF & CONTROL ELECTRONICS",
    ],
    "🧬 BIOTECH INFRASTRUCTURE": [
        "🏭 BIOTECH FOUNDRIES (CDMOs)", "❄️ COLD CHAIN & GLASS",
        "📋 CROs (Trial Managers)", "🔬 LIFE SCIENCE TOOLS",
    ],
    "🔬 BOROSILICATE & BORON": [
        "🔬 PHARMA GLASS MONOPOLY", "🪟 SPECIALTY GLASS INFRASTRUCTURE",
        "🧩 VIAL ECOSYSTEM", "⛏️ BORON RAW MATERIALS",
    ],
    "📏 BENCHMARKS": [
        "📏 SPY (S&P 500)", "📏 QQQ (Nasdaq 100)", "📏 XBI (Biotech ETF)",
        "📏 GSG (Commodity ETF)", "📏 SMH (Chip ETF)", "📏 GLD (Gold ETF)",
    ],
}

# ╔═══════════════════════════════════════════════════════════╗
# ║  💾 PERSISTENCE: Save/Load custom basket additions        ║
# ╚═══════════════════════════════════════════════════════════╝
def load_custom_baskets():
    """Load previously AI-added assets from disk."""
    if os.path.exists(SAVINGS_FILE):
        try:
            with open(SAVINGS_FILE, "r") as f:
                custom = json.load(f)
            for basket_name, holdings in custom.items():
                if basket_name not in BASKETS:
                    BASKETS[basket_name] = holdings
                else:
                    BASKETS[basket_name].update(holdings)
        except Exception:
            pass

def save_custom_basket(basket_name, holdings):
    """Persist a new/updated basket to disk."""
    custom = {}
    if os.path.exists(SAVINGS_FILE):
        try:
            with open(SAVINGS_FILE, "r") as f:
                custom = json.load(f)
        except Exception:
            custom = {}
    if basket_name not in custom:
        custom[basket_name] = {}
    custom[basket_name].update(holdings)
    with open(SAVINGS_FILE, "w") as f:
        json.dump(custom, f, indent=2)

def add_to_saf(ticker, weight, basket_name):
    """Add a ticker to a SAF basket and persist."""
    if basket_name not in BASKETS:
        BASKETS[basket_name] = {}
    BASKETS[basket_name][ticker] = weight
    save_custom_basket(basket_name, {ticker: weight})
    console.print(f"[bold green]✅ {ticker} (weight {weight}) added to '{basket_name}'[/bold green]")

# ╔═══════════════════════════════════════════════════════════╗
# ║  🩹 CACHE & CONNECTIVITY HEALING LAYER                     ║
# ╚═══════════════════════════════════════════════════════════╝
def clear_yf_cache():
    """Delete corrupted yfinance SQLite cache on every OS."""
    candidates = [
        os.path.expanduser("~/.cache/py-yfinance"),                 # Linux
        os.path.expanduser("~/Library/Caches/py-yfinance"),         # macOS
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "py-yfinance"),  # Windows
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
    """Single yf.download attempt for a chunk. Returns dict of Series."""
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
def fetch_all_prices():
    """Self-healing fetch: clear cache -> retry chunks -> single-ticker fallback."""
    all_tickers = list(set(t for h in BASKETS.values() for t in h.keys()))
    if BENCHMARK not in all_tickers:
        all_tickers.append(BENCHMARK)

    clear_yf_cache()

    if not internet_ok():
        console.print("[bold red]❌ No internet connection detected.[/bold red]")
        return pd.DataFrame()

    end   = datetime.now()
    start = (end - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    console.print(f"[dim]📡 Fetching {len(all_tickers)} tickers (self-healing mode)...[/dim]")
    all_data = {}
    chunk_size = 25

    for i in range(0, len(all_tickers), chunk_size):
        chunk = all_tickers[i:i + chunk_size]
        got = {}
        for attempt in range(1, 4):
            got = _download_batch(chunk, start, end_s)
            if got:
                break
            console.print(f"[yellow]⚠️ Batch {i//chunk_size+1} failed (attempt {attempt}/3) — clearing cache...[/yellow]")
            clear_yf_cache()
            time.sleep(1.5 * attempt)
        all_data.update(got)

        missing = [t for t in chunk if t not in all_data]
        for t in missing:
            single = _download_batch([t], start, end_s)
            if single:
                all_data.update(single)

    if not all_data:
        console.print("[bold red]❌ No data fetched after retries.[/bold red]")
        return pd.DataFrame()

    console.print(f"[green]✅ Fetched {len(all_data)}/{len(all_tickers)} tickers.[/green]")
    if len(all_data) < len(all_tickers):
        miss = sorted(set(all_tickers) - set(all_data.keys()))
        console.print(f"[yellow]⚠️ {len(miss)} unavailable: {', '.join(miss[:15])}{'...' if len(miss)>15 else ''}[/yellow]")
    return pd.DataFrame(all_data)

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
        except Exception:
            clear_yf_cache()
            time.sleep(1.0 * attempt)
    return pd.DataFrame()

def compute_indicators(df):
    if df.empty or len(df) < 50:
        return {"error": "Insufficient data"}
    c = df["Close"]
    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / loss))
    macd  = ema12 - ema26
    macd_sig = macd.ewm(span=9).mean()
    bb_mid, bb_std = c.rolling(20).mean(), c.rolling(20).std()
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - c.shift()).abs(),
                    (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    def safe(v):
        try:
            return round(float(v), 4) if pd.notna(v) else None
        except Exception:
            return None
    L = df.iloc[-1]
    return {
        "date": str(df.index[-1].date()), "close": safe(L["Close"]),
        "SMA20": safe(sma20.iloc[-1]), "SMA50": safe(sma50.iloc[-1]),
        "SMA200": safe(sma200.iloc[-1]), "RSI14": safe(rsi.iloc[-1]),
        "MACD": safe(macd.iloc[-1]), "MACD_signal": safe(macd_sig.iloc[-1]),
        "BB_upper": safe(bb_mid.iloc[-1] + 2*bb_std.iloc[-1]),
        "BB_lower": safe(bb_mid.iloc[-1] - 2*bb_std.iloc[-1]),
        "ATR14": safe(atr14.iloc[-1]),
        "return_5d_%": safe(c.pct_change(5).iloc[-1] * 100),
        "return_20d_%": safe(c.pct_change(20).iloc[-1] * 100),
    }

def get_news(ticker, limit=8):
    try:
        raw = yf.Ticker(ticker).news or []
        items = []
        for n in raw[:limit]:
            content = n.get("content", n) if isinstance(n, dict) else n
            if isinstance(content, dict):
                title = content.get("title", "")
                if title:
                    items.append(title)
        return items
    except Exception:
        return []

def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {}
    keep = ["marketCap", "trailingPE", "forwardPE", "profitMargins", "grossMargins",
            "operatingMargins", "returnOnEquity", "revenueGrowth", "debtToEquity",
            "currentRatio", "dividendYield", "targetMeanPrice", "sector", "industry"]
    return {k: info[k] for k in keep if info.get(k) is not None}

# ╔═══════════════════════════════════════════════════════════╗
# ║  📊 CALCULATIONS                                          ║
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

def safe_return(pdf, ticker, days):
    if ticker not in pdf.columns:
        return None
    return period_return(pdf[ticker], days)

def basket_return(pdf, holdings, days):
    total_w = sum(holdings.values())
    w_ret, count = 0.0, 0
    for ticker, weight in holdings.items():
        r = safe_return(pdf, ticker, days)
        if r is not None:
            w_ret += r * (weight / total_w)
            count += 1
    return w_ret if count > 0 else None

def fmt(val):
    if val is None:
        return "[dim]—[/dim]"
    c = "green" if val >= 0 else "red"
    a = "▲" if val >= 0 else "▼"
    return f"[{c}]{a} {val:+.2f}%[/{c}]"

def get_unique_tickers():
    s = set()
    for h in BASKETS.values():
        s.update(h.keys())
    return sorted(s)

# ╔═══════════════════════════════════════════════════════════╗
# ║  🤖 AGENT PROMPTS (5-analyst team, Shadow Alpha tuned)     ║
# ╚═══════════════════════════════════════════════════════════╝
TECH_SYS = """You are the TECHNICAL ANALYST. Analyze momentum, trend, volatility, support/resistance.
End with 3-5 bullet 'Key Points Summary'. Cite actual numbers."""

NEWS_SYS = """You are the NEWS ANALYST. 
When analyzing headlines, cross-reference against these supply-chain signals:
- Congressional trading disclosures (Pelosi, bipartisan stock ban votes)
- 13F filing season (Burry, Ackman, Druckenmiller position changes)
- Semiconductor bottleneck keywords: CoWoS, ABF substrate, HBM3e
- Geopolitical disruption: sanctions designations, arms trafficking route shifts
- Physical constraints: transformer lead times, CDMO capacity, helium-3 shortage
Score news impact from -1.0 to +1.0. State score first.
End with 3-5 bullet 'Key Points Summary'."""

SENT_SYS = """You are the SOCIAL MEDIA / SENTIMENT ANALYST.
Goal: Gauge crowd and GEOPOLITICAL sentiment.
Look for 'Shadow Supply Chain' signals: 'port seizure', 'dark fleet', 'sanctions evasion', 'dual-use smuggling', 'precursor shortage', 'cartel disruption'.
Score sentiment from -1.0 (Supply Shock/Bearish) to +1.0 (Clear/Bullish).
State score first. Under 100 words."""

FUND_SYS = """You are the FUNDAMENTALS ANALYST. Assess profitability, growth, valuation, leverage.
End with under/overvalued assessment."""

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

BULL_SYS = """You are the BULLISH RESEARCHER. Build the strongest case FOR investing. Counter the bear. Cite numbers. Max 150 words."""
BEAR_SYS = """You are the BEARISH RESEARCHER. Build the strongest case AGAINST investing. Counter the bull. Cite numbers. Max 150 words."""
JUDGE_SYS = """You are the DEBATE FACILITATOR. Return ONLY JSON: {"winner": "BULL" or "BEAR", "confidence": 0.0-1.0, "rationale": "..."}"""

TRADER_SYS = """You are the TRADER. Decide action, timing, sizing. Return ONLY JSON:
{"action": "BUY" or "SELL" or "HOLD", "position_pct": 0-100, "confidence": 0.0-1.0, "rationale": "..."}

NOTE: The GEOPOLITICAL analyst's score is a RISK PREMIUM score, not a sentiment score.
A high geopolitical score (+) favors Shadow Alpha bottleneck assets (BUY signal for them)
even when general sentiment is bearish. Weigh this accordingly."""

RISK_AGG_SYS = """You are the RISKY ANALYST. Advocate high-reward strategies. Max 80 words."""
RISK_NEU_SYS = """You are the NEUTRAL ANALYST. Suggest hedges, scaling. Max 80 words."""
RISK_SAFE_SYS = """You are the SAFE ANALYST. Flag drawdown risk, suggest cuts or veto. Max 80 words."""
RISK_JUDGE_SYS = """You are the RISK MANAGER. Return ONLY JSON:
{"adjusted_action": "BUY" or "SELL" or "HOLD", "adjusted_position_pct": 0-100, "risk_score": 1-10, "risk_notes": "..."}"""
MANAGER_SYS = """You are the FUND MANAGER. Final authority. Return ONLY JSON:
{"approved": true or false, "final_action": "BUY" or "SELL" or "HOLD", "final_position_pct": 0-100, "notes": "..."}"""
CORRELATE_SYS = """You are the SKIA ALPHA FUND PORTFOLIO STRATEGIST.
You specialize in Shadow Alpha: physical bottlenecks, oligopolies, supply chain choke points.
Given a ticker and its analysis, determine:
1. Which existing SAF basket it belongs to (or suggest a new basket name)
2. The appropriate weight (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
3. How it correlates with existing holdings
4. Whether it strengthens or dilutes the thesis
Return ONLY JSON:
{
  "recommended_basket": "exact basket name from the list",
  "new_basket_needed": true or false,
  "new_basket_name": "name if new",
  "suggested_weight": 0.5-3.0,
  "correlation_notes": "...",
  "thesis_fit": "STRONG" or "MODERATE" or "WEAK",
  "rationale": "..."
}"""

def run_tradingagents_pipeline(ticker):
    """Run the full 5-stage TradingAgents analysis on a ticker (5-analyst team)."""
    if not AI_ENABLED:
        console.print("[red]❌ AI not available. Set GROQ_API_KEY.[/red]")
        return None

    console.print(Panel.fit(
        f"[bold magenta]⚛️ TRADINGAGENTS PIPELINE — {ticker}[/bold magenta]\n"
        f"[dim]5-Stage Firm Simulation · 5-Analyst Team[/dim]",
        box=box.DOUBLE, border_style="magenta",
    ))

    # Gather data
    df = get_price_df(ticker)
    tech_data = compute_indicators(df) if not df.empty else {"error": "No data"}
    news_data = get_news(ticker)
    fund_data = get_fundamentals(ticker)
    news_txt = "\n".join("- " + n for n in news_data) or "No recent news."

    # STAGE I: Analyst Team (5 agents)
    console.print("\n[bold cyan]━━━ STAGE I: ANALYST TEAM ━━━[/bold cyan]")
    analysts = {}
    with console.status("Running 5 analysts..."):
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {
                "technical": ex.submit(llm, TECH_SYS,
                    f"Ticker: {ticker}\nData:\n{json.dumps(tech_data, indent=2)}"),
                "news": ex.submit(llm, NEWS_SYS,
                    f"Ticker: {ticker}\nHeadlines:\n{news_txt}"),
                "sentiment": ex.submit(llm, SENT_SYS,
                    f"Ticker: {ticker}\nHeadlines: {json.dumps(news_data)}", MODEL_FAST, 0.3),
                "fundamentals": ex.submit(llm, FUND_SYS,
                    f"Ticker: {ticker}\nFundamentals:\n{json.dumps(fund_data, indent=2)}"),
                "geopolitical": ex.submit(llm, GEOPOL_SYS,
                    f"Ticker: {ticker}\nHeadlines:\n{news_txt}\n"
                    f"Sector context: {fund_data.get('sector', 'N/A')} / {fund_data.get('industry', 'N/A')}"),
            }
            for name, fut in futures.items():
                try:
                    analysts[name] = fut.result() or "(no output)"
                except Exception as e:
                    analysts[name] = f"Error: {e}"
    for k in analysts:
        console.print(f"  [green]✓[/green] {k} analyst complete")

    # STAGE II: Bull vs Bear Debate
    console.print("\n[bold cyan]━━━ STAGE II: BULL vs BEAR DEBATE ━━━[/bold cyan]")
    reports_txt = "\n".join(f"### {k.upper()} ###\n{v[:600]}" for k, v in analysts.items())
    bull_arg = llm(BULL_SYS, f"Ticker: {ticker}\nREPORTS:\n{reports_txt}\nMake the bullish case.")
    bear_arg = llm(BEAR_SYS, f"Ticker: {ticker}\nREPORTS:\n{reports_txt}\nBULL says: {bull_arg}\nRebut.")
    transcript = f"[BULL]\n{bull_arg}\n[BEAR]\n{bear_arg}"
    verdict = extract_json(llm(JUDGE_SYS, f"DEBATE:\n{transcript}", temperature=0.2, force_json=True))
    if not verdict:
        verdict = {"winner": "BEAR", "confidence": 0.5, "rationale": "Judge unavailable."}
    console.print(f"  [green]✓[/green] Debate winner: [bold]{verdict.get('winner')}[/bold] "
                  f"(confidence {verdict.get('confidence', '?')})")

    # STAGE III: Trader
    console.print("\n[bold cyan]━━━ STAGE III: TRADER ━━━[/bold cyan]")
    trader_prompt = (f"Ticker: {ticker}\nREPORTS:\n{reports_txt}\n"
                     f"DEBATE WINNER: {verdict.get('winner')}\nMake today's decision.")
    trader = extract_json(llm(TRADER_SYS, trader_prompt, temperature=0.3, force_json=True))
    if not trader:
        trader = {"action": "HOLD", "position_pct": 0, "confidence": 0, "rationale": "Trader unavailable."}
    console.print(f"  [green]✓[/green] Trader: [bold]{trader.get('action')}[/bold] "
                  f"@ {trader.get('position_pct')}%")

    # STAGE IV: Risk Team
    console.print("\n[bold cyan]━━━ STAGE IV: RISK MANAGEMENT ━━━[/bold cyan]")
    plan_txt = json.dumps(trader, indent=2)
    risk_opinions = {}
    for label, sysp in [("Aggressive", RISK_AGG_SYS), ("Neutral", RISK_NEU_SYS), ("Safe", RISK_SAFE_SYS)]:
        risk_opinions[label] = llm(sysp, f"Ticker: {ticker}\nTRADER PLAN:\n{plan_txt}")
    all_risk = "\n".join(f"[{k}]\n{v}" for k, v in risk_opinions.items())
    risk_adj = extract_json(llm(RISK_JUDGE_SYS,
                     f"TRADER PLAN:\n{plan_txt}\nRISK OPINIONS:\n{all_risk}", temperature=0.2, force_json=True))
    if not risk_adj:
        risk_adj = {"adjusted_action": trader.get("action", "HOLD"),
                    "adjusted_position_pct": trader.get("position_pct", 0),
                    "risk_score": 5, "risk_notes": "Risk judge unavailable."}
    console.print(f"  [green]✓[/green] Risk-adjusted: [bold]{risk_adj.get('adjusted_action')}[/bold] "
                  f"@ {risk_adj.get('adjusted_position_pct')}% "
                  f"(risk {risk_adj.get('risk_score')}/10)")

    # STAGE V: Fund Manager
    console.print("\n[bold cyan]━━━ STAGE V: FUND MANAGER ━━━[/bold cyan]")
    mgr_prompt = (f"TRADER: {json.dumps(trader)}\n"
                  f"RISK-ADJUSTED: {json.dumps(risk_adj)}\n"
                  f"DEBATE WINNER: {verdict.get('winner')}\nFinal call.")
    final = extract_json(llm(MANAGER_SYS, mgr_prompt, temperature=0.2, force_json=True))
    if not final:
        final = {"approved": True, "final_action": risk_adj.get("adjusted_action", "HOLD"),
                 "final_position_pct": risk_adj.get("adjusted_position_pct", 0),
                 "notes": "Fund manager unavailable."}

    # Display verdict
    action = final.get("final_action", "HOLD")
    color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(action, "white")
    console.print(Panel.fit(
        f"[bold {color}]FUND MANAGER VERDICT: {action}[/bold {color}]\n"
        f"Position: {final.get('final_position_pct', 0)}% | "
        f"Approved: {final.get('approved')}\n"
        f"[italic]{str(final.get('notes', ''))[:250]}[/italic]",
        title=f"⚛️ {ticker} — Final Signal", box=box.DOUBLE, border_style=color,
    ))

    return {
        "ticker": ticker,
        "analysts": analysts,
        "verdict": verdict,
        "trader": trader,
        "risk_adjusted": risk_adj,
        "final": final,
        "tech_data": tech_data,
        "fund_data": fund_data,
    }

# ╔═══════════════════════════════════════════════════════════╗
# ║  🤖 AI CORRELATION & BASKET ASSIGNMENT                    ║
# ╚═══════════════════════════════════════════════════════════╝
def ai_correlate_and_assign(ticker, pipeline_result):
    """AI determines which SAF basket the ticker belongs to and suggests weight."""
    if not AI_ENABLED:
        return None
    console.print(f"\n[bold cyan]━━━ AI CORRELATION & BASKET ASSIGNMENT ━━━[/bold cyan]")

    basket_summary = "\n".join(f"- {name}: {', '.join(list(h.keys())[:6])}..."
                               for name, h in list(BASKETS.items())[:20])
    analysis_summary = (
        f"Fund Manager Action: {pipeline_result['final'].get('final_action')}\n"
        f"Risk Score: {pipeline_result['risk_adjusted'].get('risk_score')}/10\n"
        f"Debate Winner: {pipeline_result['verdict'].get('winner')}\n"
        f"Trader Rationale: {pipeline_result['trader'].get('rationale', '')[:200]}\n"
        f"Fundamentals: {json.dumps(pipeline_result.get('fund_data', {}), default=str)[:300]}"
    )

    with console.status("[magenta]AI correlating with SAF portfolio...[/magenta]"):
        out = llm(CORRELATE_SYS,
                  f"TICKER: {ticker}\n"
                  f"ANALYSIS RESULTS:\n{analysis_summary}\n"
                  f"EXISTING SAF BASKETS:\n{basket_summary}\n"
                  f"Determine the best basket and weight for this asset.",
                  temperature=0.3, force_json=True)
    result = extract_json(out)
    if not result:
        console.print("[yellow]⚠️ AI correlation failed. Manual assignment needed.[/yellow]")
        return None

    corr_tbl = Table(title=f"🤖 AI Assignment — {ticker}", box=box.ROUNDED)
    corr_tbl.add_column("Attribute", style="bold")
    corr_tbl.add_column("Value", overflow="fold")
    corr_tbl.add_row("Recommended Basket", result.get("recommended_basket", "?"))
    corr_tbl.add_row("New Basket Needed?", str(result.get("new_basket_needed", False)))
    if result.get("new_basket_needed"):
        corr_tbl.add_row("New Basket Name", result.get("new_basket_name", "?"))
    corr_tbl.add_row("Suggested Weight", str(result.get("suggested_weight", 1.0)))
    corr_tbl.add_row("Thesis Fit", result.get("thesis_fit", "?"))
    corr_tbl.add_row("Correlation Notes", result.get("correlation_notes", ""))
    corr_tbl.add_row("Rationale", result.get("rationale", ""))
    console.print(corr_tbl)
    return result

# ╔═══════════════════════════════════════════════════════════╗
# ║  🔄 THE FULL PIPELINE: Screen → Analyze → Assign → Add    ║
# ╚═══════════════════════════════════════════════════════════╝
def run_full_pipeline():
    """The complete pipeline: TradingAgents → Correlate → Add to SAF."""
    if not AI_ENABLED:
        console.print("[red]❌ AI not available. Set GROQ_API_KEY.[/red]")
        return
    ticker = Prompt.ask("\n[bold]Enter ticker to run through SAF pipeline[/bold]",
                        default="MKSI").upper().strip()

    existing = [b for b, h in BASKETS.items() if ticker in h]
    if existing:
        console.print(f"[yellow]⚠️ {ticker} is already in: {', '.join(existing)}[/yellow]")
        if not Confirm.ask("[bold]Run pipeline anyway?[/bold]", default=True):
            return

    pipeline_result = run_tradingagents_pipeline(ticker)
    if not pipeline_result:
        return

    assignment = ai_correlate_and_assign(ticker, pipeline_result)

    if not assignment:
        console.print("\n[yellow]Manual basket selection:[/yellow]")
        basket_names = list(BASKETS.keys())
        for i, b in enumerate(basket_names[:15], 1):
            console.print(f"  [{i:2}] {b}")
        choice = Prompt.ask("[bold]Enter basket number or name[/bold]", default="1")
        try:
            idx = int(choice) - 1
            target_basket = basket_names[idx]
        except (ValueError, IndexError):
            target_basket = choice
        weight = float(Prompt.ask("[bold]Weight (0.5/1.0/1.5/2.0/2.5/3.0)[/bold]", default="1.0"))
    else:
        if assignment.get("new_basket_needed"):
            target_basket = assignment.get("new_basket_name", f"🆕 {ticker} THESIS")
        else:
            target_basket = assignment.get("recommended_basket", "")
        if target_basket not in BASKETS:
            matches = [b for b in BASKETS if target_basket.lower()[:10] in b.lower()]
            target_basket = matches[0] if matches else f"🆕 {ticker} THESIS"
        weight = float(assignment.get("suggested_weight", 1.0))

    console.print(f"\n[bold]Ready to add:[/bold]")
    console.print(f"  Ticker: [cyan]{ticker}[/cyan]")
    console.print(f"  Basket: [cyan]{target_basket}[/cyan]")
    console.print(f"  Weight: [cyan]{weight}[/cyan]")
    console.print(f"  Fund Manager: [cyan]{pipeline_result['final'].get('final_action')}[/cyan]")

    if Confirm.ask("\n[bold green]✅ Add to SAF?[/bold green]", default=True):
        add_to_saf(ticker, weight, target_basket)
        console.print(f"\n[bold green]🎉 {ticker} is now part of the Skia Alpha Fund![/bold green]")
        console.print("[dim]Run Option 1 (Dashboard) to see it tracked alongside all baskets.[/dim]")
    else:
        console.print("[yellow]Skipped. Asset not added.[/yellow]")

# ╔═══════════════════════════════════════════════════════════╗
# ║  🖥️  DISPLAY FUNCTIONS (from sfa2.py)                      ║
# ╚═══════════════════════════════════════════════════════════╝
def show_dashboard(pdf):
    console.clear()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(Panel.fit(
        f"[bold magenta]🦅 SKIA ALPHA FUND v3 — DASHBOARD[/bold magenta]\n"
        f"[dim]{now} · {len(BASKETS)} baskets · {len(get_unique_tickers())} tickers[/dim]",
        box=box.DOUBLE, border_style="magenta",
    ))
    tbl = Table(title="📈 All Baskets Performance", box=box.HEAVY_HEAD, show_lines=True)
    tbl.add_column("Basket", style="bold", min_width=38)
    tbl.add_column("1D", justify="right", width=10)
    tbl.add_column("1W", justify="right", width=10)
    tbl.add_column("1M", justify="right", width=10)
    tbl.add_column("YTD", justify="right", width=10)
    for sec, names in SECTIONS.items():
        tbl.add_row(f"[bold yellow]── {sec} ──[/bold yellow]", "", "", "", "")
        for n in names:
            if n not in BASKETS:
                continue
            h = BASKETS[n]
            tbl.add_row(n,
                        fmt(basket_return(pdf, h, 1)), fmt(basket_return(pdf, h, 7)),
                        fmt(basket_return(pdf, h, 30)), fmt(basket_return(pdf, h, "YTD")))
    custom = [b for b in BASKETS if not any(b in names for names in SECTIONS.values())]
    if custom:
        tbl.add_row(f"[bold yellow]── 🆕 AI-ADDED BASKETS ──[/bold yellow]", "", "", "", "")
        for n in custom:
            h = BASKETS[n]
            tbl.add_row(n,
                        fmt(basket_return(pdf, h, 1)), fmt(basket_return(pdf, h, 7)),
                        fmt(basket_return(pdf, h, 30)), fmt(basket_return(pdf, h, "YTD")))
    console.print(tbl)

def show_all_tickers(pdf):
    console.clear()
    console.print(Panel.fit("[bold cyan]🏷️ ALL TICKERS[/bold cyan]", box=box.DOUBLE, border_style="cyan"))
    for sec, names in SECTIONS.items():
        console.print(f"\n[bold yellow]══ {sec} ══[/bold yellow]")
        for basket in names:
            if basket not in BASKETS:
                continue
            holdings = BASKETS[basket]
            tbl = Table(title=basket, box=box.SIMPLE, show_edge=False)
            tbl.add_column("Ticker", style="bold cyan")
            tbl.add_column("Weight", justify="right", width=7)
            tbl.add_column("Price", justify="right", width=10)
            tbl.add_column("YTD", justify="right", width=10)
            for ticker, weight in holdings.items():
                price = "—"
                if ticker in pdf.columns:
                    p = pdf[ticker].dropna()
                    if len(p):
                        price = f"${p.iloc[-1]:.2f}"
                tbl.add_row(ticker, f"{weight:.1f}", price, fmt(safe_return(pdf, ticker, "YTD")))
            console.print(tbl)

def show_ranking(pdf):
    perf = []
    for n, h in BASKETS.items():
        y = basket_return(pdf, h, "YTD")
        if y is not None:
            perf.append((n, y))
    perf.sort(key=lambda x: x[1], reverse=True)
    rk = Table(title="🏆 YTD Ranking", box=box.SIMPLE_HEAVY)
    rk.add_column("#", justify="center", width=4)
    rk.add_column("Basket", style="bold")
    rk.add_column("YTD", justify="right", width=12)
    for i, (n, y) in enumerate(perf, 1):
        tag = f"[green]{i}[/green]" if i <= 5 else (f"[red]{i}[/red]" if i > len(perf)-3 else str(i))
        rk.add_row(tag, n, fmt(y))
    console.print(rk)

def search_ticker(pdf):
    query = Prompt.ask("\n[bold]Enter ticker[/bold]", default="NVDA").upper().strip()
    found = [(b, w) for b, h in BASKETS.items() if query in h for w in [h[query]]]
    if not found:
        console.print(f"[red]❌ '{query}' not found.[/red]")
        return
    console.print(f"\n[bold green]✅ '{query}' in {len(found)} basket(s):[/bold green]")
    tbl = Table(box=box.SIMPLE_HEAVY)
    tbl.add_column("Basket", style="bold")
    tbl.add_column("Weight", justify="right")
    for b, w in found:
        tbl.add_row(b, f"{w:.1f}")
    console.print(tbl)

def export_csv(pdf):
    rows = []
    for basket, holdings in BASKETS.items():
        for ticker, weight in holdings.items():
            p = pdf[ticker].dropna() if ticker in pdf.columns else pd.Series(dtype=float)
            rows.append({
                "Basket": basket, "Ticker": ticker, "Weight": weight,
                "Price": round(p.iloc[-1], 2) if len(p) else None,
                "YTD_%": round(safe_return(pdf, ticker, "YTD") or 0, 2),
            })
    pd.DataFrame(rows).to_csv(EXPORT_FILE, index=False)
    console.print(f"[green]✅ Exported to {EXPORT_FILE}[/green]")

# ╔═══════════════════════════════════════════════════════════╗
# ║  🧪 SYSTEM DIAGNOSTICS — full pipeline health check        ║
# ╚═══════════════════════════════════════════════════════════╝
def run_diagnostics():
    console.print(Panel.fit(
        "[bold cyan]🧪 SAF SYSTEM DIAGNOSTICS[/bold cyan]\n"
        "[dim]Testing data pipeline, AI connection, and file I/O[/dim]",
        box=box.DOUBLE, border_style="cyan"))

    results = []

    # 1. Dependencies
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

    # 2. Internet
    net = internet_ok()
    results.append(("Internet connectivity", net,
                    "Yahoo/Groq reachable" if net else "no connection"))

    # 3. Yahoo Finance data (live pull)
    yf_ok, yf_detail = False, "no data"
    if net:
        try:
            clear_yf_cache()
            t = yf.Ticker(BENCHMARK)
            h = t.history(period="5d")
            if h is not None and not h.empty:
                yf_ok = True
                last = h["Close"].iloc[-1]
                yf_detail = f"{BENCHMARK} close ${last:.2f} ({len(h)} bars)"
            else:
                yf_detail = "empty history"
        except Exception as e:
            yf_detail = f"error: {str(e)[:60]}"
    results.append(("Yahoo Finance data feed", yf_ok, yf_detail))

    # 4. yfinance cache health
    cache_ok = True
    try:
        clear_yf_cache()
        _ = yf.Ticker("AAPL").history(period="1d")
    except Exception:
        cache_ok = False
    results.append(("yfinance cache health", cache_ok,
                    "cleared & re-writable" if cache_ok else "still failing — check disk permissions"))

    # 5. Groq API key
    key_ok = bool(GROQ_API_KEY)
    results.append(("GROQ_API_KEY present", key_ok,
                    "set" if key_ok else "NOT set — AI features disabled"))

    # 6-8. AI model tests + JSON mode
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

    # 9. File I/O
    def writable(path):
        try:
            d = os.path.dirname(os.path.abspath(path)) or "."
            return os.access(d, os.W_OK)
        except Exception:
            return False
    fio_ok = writable(SAVINGS_FILE) and writable(EXPORT_FILE) and writable(MEMORY_FILE)
    results.append(("File write permissions", fio_ok,
                    "savings/export/memory writable" if fio_ok else "check directory permissions"))

    # 10. Baskets integrity
    btickers = get_unique_tickers()
    b_ok = len(BASKETS) > 0 and len(btickers) > 0
    results.append(("Basket config", b_ok,
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
    load_custom_baskets()
    ai_status = "[green]ENABLED[/green]" if AI_ENABLED else "[red]DISABLED[/red]"
    console.print(Panel.fit(
        "[bold magenta]🦅 SKIA ALPHA FUND v3[/bold magenta]\n"
        f"[dim]Unified AI Pipeline · AI: {ai_status}[/dim]",
        box=box.DOUBLE, border_style="magenta",
    ))

    if not internet_ok():
        console.print("[bold red]⚠️ No internet detected — data features will fail. Run [9] Diagnostics.[/bold red]")

    pdf = fetch_all_prices()
    if pdf.empty:
        console.print("[red]❌ No data. Run [9] Diagnostics to troubleshoot, or retry.[/red]")

    while True:
        console.print("\n[bold cyan]╔════════════ SAF v3 MAIN MENU ═════════════╗[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [bold]PORTFOLIO TRACKING[/bold]                      [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [1] 📈 Full Dashboard                     [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [2] 🏷️  All Tickers                        [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [3] 🏆 YTD Ranking                        [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [4] 🔍 Search Ticker                      [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [5] 💾 Export to CSV                      [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [6] 🔄 Refresh Data                       [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [bold]AI PIPELINE (Requires GROQ_API_KEY)[/bold]     [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [7] 🤖 FULL PIPELINE (Analyze → Add)     [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [8] 🤖 Quick AI Correlation Only          [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [bold]SYSTEM[/bold]                                  [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [9] 🧪 Run Diagnostics (Health Check)     [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]║[/bold cyan]  [0] 🚪 Exit                               [bold cyan]║[/bold cyan]")
        console.print("[bold cyan]╚════════════════════════════════════════════╝[/bold cyan]")
        choice = Prompt.ask("\n[bold]Select[/bold]",
                            choices=["0","1","2","3","4","5","6","7","8","9"], default="1")
        if choice == "1":
            show_dashboard(pdf)
        elif choice == "2":
            show_all_tickers(pdf)
        elif choice == "3":
            show_ranking(pdf)
        elif choice == "4":
            search_ticker(pdf)
        elif choice == "5":
            export_csv(pdf)
        elif choice == "6":
            pdf = fetch_all_prices()
        elif choice == "7":
            run_full_pipeline()
        elif choice == "8":
            if not AI_ENABLED:
                console.print("[red]❌ AI not available.[/red]")
            else:
                ticker = Prompt.ask("[bold]Ticker to correlate[/bold]", default="MKSI").upper().strip()
                fund_data = get_fundamentals(ticker)
                fake_result = {
                    "final": {"final_action": "BUY", "final_position_pct": 50},
                    "risk_adjusted": {"risk_score": 5},
                    "verdict": {"winner": "BULL"},
                    "trader": {"rationale": "Quick correlation mode"},
                    "fund_data": fund_data,
                }
                ai_correlate_and_assign(ticker, fake_result)
        elif choice == "9":
            run_diagnostics()
        elif choice == "0":
            console.print("\n[bold red]👋 Goodbye, Skia Alpha Fund Manager.[/bold red]")
            break
        if choice != "0":
            console.input("\n[dim]Press Enter to return...[/dim]")

if __name__ == "__main__":
    main_menu()