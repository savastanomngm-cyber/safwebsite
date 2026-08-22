"""Evidence Pack builder v2 — SEC EDGAR enrichment (improvements.txt PART 5 Step 1).
PATCHED: Broadened concentration regex to catch "largest [industry] company worldwide"
(which previously missed LIN and similar oligopolies)."""
import html
import re
import time
import requests
import yfinance as yf
from ..security import clean_text

SEC_HEADERS = {"User-Agent": "SkiaAlphaFund research contact@example.com"}
_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

_CIK_MAP = None

def _cik_map():
    global _CIK_MAP
    if _CIK_MAP is None:
        try:
            r = requests.get(_TICKER_MAP_URL, headers=SEC_HEADERS, timeout=15)
            r.raise_for_status()
            _CIK_MAP = {v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                        for v in r.json().values()}
        except Exception:
            _CIK_MAP = {}
    return _CIK_MAP

def sec_cik_lookup(ticker):
    return _cik_map().get(ticker.upper())

def latest_annual_filing(cik, forms=("10-K", "10-K/A", "20-F")):
    try:
        r = requests.get(_SUBMISSIONS.format(cik=cik), headers=SEC_HEADERS, timeout=15)
        r.raise_for_status()
        recent = r.json().get("filings", {}).get("recent", {})
        for i, f in enumerate(recent.get("form", [])):
            if f in forms:
                return recent["accessionNumber"][i], recent["primaryDocument"][i]
    except Exception:
        pass
    return None

def fetch_filing_text(cik, accession, doc):
    try:
        acc = accession.replace("-", "")
        cik_raw = cik.lstrip("0") or "0"
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{acc}/{doc}"
        r = requests.get(url, headers=SEC_HEADERS, timeout=30)
        r.raise_for_status()
        txt = re.sub(r"<[^>]+>", " ", r.text)
        txt = html.unescape(txt)
        txt = re.sub(r"\s+", " ", txt)
        return txt
    except Exception:
        return ""

_SECTION_STARTS = [
    r"Item\s+1[.\s:]+Business",
    r"Item\s+1\.?\s+Business",
    r"Item\s+4[.\s:]+Information\s+on\s+the\s+Company",
    r"Item\s+4\.?\s+Information\s+on\s+the\s+Company",
]
_SECTION_BOUNDARY = r"Item\s*(?:1[A-C]|2|3|4[AB]|5)[.\s:]"
_SECTION_MAX_CHARS = 15000

def extract_business_section(filing_text):
    if not filing_text: return ""
    best = ""
    for pat in _SECTION_STARTS:
        for m in re.finditer(pat, filing_text, flags=re.I):
            start = m.end()
            window = filing_text[start:start + _SECTION_MAX_CHARS + 5000]
            end_m = re.search(_SECTION_BOUNDARY, window, flags=re.I)
            chunk = window[:end_m.start()] if end_m else window[:_SECTION_MAX_CHARS]
            if len(chunk) > len(best): best = chunk
    return best[:_SECTION_MAX_CHARS]

# FIX: Broadened patterns to catch "largest [industry] company worldwide"
CONCENTRATION_PATTERNS = [
    r"(?:largest|leading|top|world'?s\s+largest)\s+(?:two|three|four|five|few|provider|supplier|producer|manufacturer|company|companies|player|players)",
    r"(?:largest|leading)\s+[\w\s]{2,40}?\s+(?:company|provider|supplier|manufacturer|producer)\s+(?:worldwide|globally|in\s+the\s+world)",
    r"control[s]?\s+(?:approximately\s+)?\d{2,3}\s*(?:%|percent)",
    r"oligopoly", r"duopoly", r"near[- ]monopoly", r"monopoly",
    r"no\s+(?:viable|known|commercial)\s+substitute",
    r"sole\s+(?:approved\s+)?(?:supplier|source|manufacturer)",
    r"limited\s+number\s+of\s+(?:qualified\s+)?(?:suppliers|competitors|vendors)",
    r"high\s+switching\s+costs", r"barriers\s+to\s+entry",
    r"(?:significant|substantial)\s+market\s+share",
]

def grep_concentration(text, max_hits=8):
    hits, seen = [], set()
    for pat in CONCENTRATION_PATTERNS:
        for m in re.finditer(pat, text, flags=re.I):
            s = max(0, m.start() - 80)
            e = min(len(text), m.end() + 80)
            hit = clean_text(text[s:e], 250)
            if hit and hit not in seen:
                seen.add(hit)
                hits.append(hit)
            if len(hits) >= max_hits: return hits
    return hits

def build_evidence_pack(ticker: str) -> dict:
    pack = {"ticker": ticker, "business_desc": "", "concentration_hits": [],
            "fundamentals": {}, "recent_headlines": [], "sec_source": None}

    cik = sec_cik_lookup(ticker)
    if cik:
        filing = latest_annual_filing(cik)
        if filing:
            acc, doc = filing
            text = fetch_filing_text(cik, acc, doc)
            section = extract_business_section(text)
            if len(section) > 300:
                pack["business_desc"] = section
                pack["sec_source"] = f"SEC EDGAR annual report (CIK {cik})"
                pack["concentration_hits"] = grep_concentration(text)
            time.sleep(0.15)

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        if not pack["business_desc"]:
            pack["business_desc"] = clean_text(info.get("longBusinessSummary", ""), 2500)
            pack["sec_source"] = "yfinance summary (no SEC filing found)"
        pack["fundamentals"] = {
            "gross_margin": info.get("grossMargins"),
            "oper_margin": info.get("operatingMargins"),
            "market_cap": info.get("marketCap"),
            "industry": info.get("industry"),
            "sector": info.get("sector"),
        }
        for n in (t.news or [])[:5]:
            title = n.get("title") or n.get("content", {}).get("title", "")
            if title: pack["recent_headlines"].append(clean_text(title, 150))
    except Exception as e:
        pack["error"] = str(e)[:100]

    if not pack["concentration_hits"] and pack["business_desc"]:
        pack["concentration_hits"] = grep_concentration(pack["business_desc"], max_hits=4)

    return pack