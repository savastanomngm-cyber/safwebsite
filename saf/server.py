"""SAF v4 — FastAPI research server."""
import json, time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from . import config, store, data
from .security import load_env, clean_text
from .quant import score as S

GROQ_API_KEY = load_env()
store.init()
app = FastAPI(title="Skia Alpha Fund v4", version="4.0.6")

@asynccontextmanager
async def lifespan(_app):
    store.audit_log("server_start", {"ai_key_present": bool(GROQ_API_KEY)})
    yield

app.router.lifespan_context = lifespan
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: JSONResponse(429, {"detail": f"Rate limit: {e.detail}"}))
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

def provenance(source: str, asof: str = None, note: str = None) -> dict:
    return {"source": source, "asof": asof, "note": note, "served_at": datetime.now().isoformat(timespec="seconds")}

_PX = {"cache": {}, "built": 0.0}
def _px(ticker: str) -> pd.DataFrame:
    if time.time() - _PX["built"] > 900: _PX["cache"], _PX["built"] = {}, time.time()
    if ticker not in _PX["cache"]: _PX["cache"][ticker] = store.load_prices(ticker)
    return _PX["cache"][ticker]

def _period_return(series: pd.Series, days):
    series = series.dropna()
    if len(series) < 2: return None
    if days == "YTD":
        yp = series[series.index.year == series.index[-1].year]
        if len(yp) < 2: return None
        base = yp.iloc[0]
    else:
        valid = series[series.index >= series.index[-1] - pd.Timedelta(days=days)]
        if len(valid) < 2: return None
        base = valid.iloc[0]
    return round((series.iloc[-1] / base - 1) * 100, 2)

def _flags(raw):
    try:
        parsed = json.loads(raw or "[]")
        return parsed if isinstance(parsed, list) else []
    except Exception: return []

def _f(v):
    try: return None if pd.isna(float(v)) else round(float(v), 4)
    except Exception: return None

@app.get("/api/system/health")
@limiter.limit("30/minute")
def system_health(request: Request):
    cfg = config.load(); bench = cfg["settings"]["benchmark"]; spy = _px(bench)
    return {"status": "ok", "ai_key_present": bool(GROQ_API_KEY), "benchmark": bench,
            "benchmark_bars": int(len(spy)), "baskets": len(cfg["baskets"]),
            "universe_tickers": len(config.all_tickers(cfg)), "audit_chain_ok": store.verify_audit_chain(),
            "provenance": provenance("live")}

@app.get("/api/baskets")
@limiter.limit("30/minute")
def api_baskets(request: Request):
    cfg = config.load(); out = []
    for b in cfg["baskets"]:
        total_w = sum(b["holdings"].values()) or 1; rets = {}
        for label, days in (("1d", 1), ("1w", 7), ("1m", 30), ("ytd", "YTD")):
            w_ret, count = 0.0, 0
            for t, w in b["holdings"].items():
                px = _px(t)
                if px.empty: continue
                r = _period_return(px["px"], days)
                if r is not None: w_ret += r * (w / total_w); count += 1
            if count: rets[label] = round(w_ret, 2)
        out.append({"name": clean_text(b["name"], 80), "section": b.get("section", ""),
                    "timing_class": b.get("timing_class", "hold_only"),
                    "n_holdings": len(b["holdings"]), "returns_pct": rets})
    return {"baskets": out, "provenance": provenance("cached")}

@app.get("/api/ticker/{t}")
@limiter.limit("30/minute")
def api_ticker(request: Request, t: str):
    t = t.upper().strip(); px = _px(t)
    if px.empty: raise HTTPException(404, f"No data for {t}")
    cfg = config.load(); spy = _px(cfg["settings"]["benchmark"])
    fund = store.get_fundamentals(t); q = data.quality_report(t)
    s = S.score_v2(t, px.index[-1], {t: px}, spy, fund=fund) # B-score automatically pulls from DB!
    return {"ticker": t, "price": _f(px["px"].iloc[-1]), "asof": str(px.index[-1].date()),
            "quality": {"usable": q["usable"], "bars": q["bars"], "flags": _flags(q["flags"])},
            "fundamentals": {k: v for k, v in (fund or {}).items() if not k.startswith("_")},
            "score_v2_core": s, "provenance": provenance("cached")}

@app.get("/api/ticker/{t}/rubric")
@limiter.limit("10/minute")
def api_rubric(request: Request, t: str):
    from .ai import evidence, rubric
    t = t.upper().strip()
    
    # 1. Check cache first (saves API tokens!)
    cached = store.get_cached_rubric(t)
    if cached and not request.query_params.get("force"):
        return {"ticker": t, "ok": True, "rubric": cached["raw"], "cached": True, 
                "age_days": cached["age_days"], "provenance": provenance("cached", note="Rubric cache hit")}

    # 2. Cache miss -> build evidence and call LLM
    pack = evidence.build_evidence_pack(t)
    if not pack.get("business_desc"): raise HTTPException(404, f"No evidence found for {t}")
    result = rubric.score_bottleneck(t, pack)
    ok = "error" not in result

    # 3. Save to cache if successful
    if ok:
        store.save_rubric(t, result["total"], result)
        store.audit_log("ai_rubric", {"ticker": t, "ok": True, "cached": False})
        return {"ticker": t, "ok": True, "rubric": result, "cached": False,
                "evidence_hits": pack.get("concentration_hits", []),
                "fundamentals": pack.get("fundamentals", {}),
                "sec_source": pack.get("sec_source"),
                "provenance": provenance("live", note="Grounded AI w/ citation check")}
    else:
        return {"ticker": t, "ok": False, "rubric": result, "cached": False,
                "provenance": provenance("live", note="LLM failed")}

@app.get("/api/screen")
@limiter.limit("5/minute")
def api_screen(request: Request, top: int = 15):
    cfg = config.load(); bench = cfg["settings"]["benchmark"]; spy = _px(bench)
    if spy.empty: raise HTTPException(503, "Benchmark not fetched")
    upto = spy.index[-1]; rows = []
    for t in config.all_tickers(cfg):
        if t == bench: continue
        px = _px(t)
        if px.empty or len(px) < 250: continue
        # score_v2 automatically pulls the B-score from the SQLite cache!
        s = S.score_v2(t, upto, {t: px}, spy, fund=store.get_fundamentals(t))
        if s: rows.append(s)
    rows.sort(key=lambda r: r["total"], reverse=True)
    store.audit_log("api_screen", {"asof": str(upto.date()), "n": len(rows)})
    return {"asof": str(upto.date()), "n_scored": len(rows), "top": rows[:top],
            "provenance": provenance("cached", note="Score v2 core w/ Bottleneck Prior")}

_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("SAF v4 server  → http://127.0.0.1:8000/static/")
    print("AI key:", "present (server-side)" if GROQ_API_KEY else "MISSING")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")