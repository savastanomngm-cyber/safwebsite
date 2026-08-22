"""Incremental fetcher with quality gates. Never re-downloads what it has."""
import yfinance as yf
import pandas as pd

from . import config, store


def refresh_ticker(ticker: str, lookback_days=None) -> bool:
    """Fetch only missing dates. Returns True if data was stored/already current."""
    lookback = lookback_days or config.load()["settings"]["lookback_days"]
    last = store.last_price_date(ticker)
    start = (pd.Timestamp(last) + pd.Timedelta(days=1)) if last \
        else pd.Timestamp.now() - pd.Timedelta(days=lookback)

    if pd.Timestamp(start).date() >= pd.Timestamp.now().date():
        return True  # already current

    try:
        raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False, threads=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw is None or raw.empty:
            _record_failure(ticker)
            return False

        raw.columns = [str(c).lower() for c in raw.columns]
        if "adj_close" not in raw.columns:       # newer yfinance: auto_adjust folds it
            raw["adj_close"] = raw["close"]
        if "volume" not in raw.columns:
            raw["volume"] = 0

        store.upsert_prices(ticker, raw.dropna(subset=["close"]))
        store.set_flags(ticker, compute_quality_flags(ticker))
        return True
    except Exception as e:
        _record_failure(ticker)
        store.audit_log("data_error", {"ticker": ticker, "err": str(e)[:120]})
        return False


def _record_failure(ticker):
    with store.con() as c:
        c.execute("""INSERT INTO meta (ticker,fetch_failures) VALUES (?,1)
                     ON CONFLICT(ticker) DO UPDATE SET
                       fetch_failures=fetch_failures+1""", (ticker,))


def refresh_fundamentals(ticker: str) -> dict | None:
    cached = store.get_fundamentals(ticker, max_age_days=7)
    if cached and not cached.get("_stale"):
        return cached
    try:
        info = yf.Ticker(ticker).info or {}
        keep = ["longName", "marketCap", "trailingPE", "forwardPE",
                "grossMargins", "operatingMargins", "returnOnEquity",
                "revenueGrowth", "debtToEquity", "currentRatio",
                "dividendYield", "targetMeanPrice", "sector", "industry",
                "averageVolume", "shortPercentOfFloat"]
        slim = {k: info[k] for k in keep if info.get(k) is not None}
        if slim:
            store.save_fundamentals(ticker, slim)
            return slim
    except Exception:
        pass
    return cached  # stale cache beats nothing


def compute_quality_flags(ticker: str) -> list:
    """Data-honesty layer: every anomaly that should disqualify or warn."""
    df = store.load_prices(ticker)
    if df.empty:
        return ["NO_DATA"]

    flags = []
    now = pd.Timestamp.now(tz=df.index.tz) if df.index.tz else pd.Timestamp.now()
    stale_days = (now - df.index[-1]).days

    if stale_days > 7:
        flags.append("STALE>7d")
    if len(df) < 200:
        flags.append("SHORT_HISTORY")
    if (df["volume"].fillna(0) == 0).mean() > 0.10:
        flags.append("LOW_VOLUME")

    gap_days = int((df["px"].pct_change().abs() > 0.25).sum())
    if gap_days > 3:
        flags.append(f"GAP_ANOMALY:{gap_days}")

    # Crude corporate-action suspicion: >25% one-day move with volume >= 1.8x
    # 20d average that does NOT mean-revert within 20 bars is likely news, not
    # a bad split adjustment — flag either way for human review.
    vol_ratio = df["volume"] / df["volume"].rolling(20).mean()
    big_moves = df["px"].pct_change().abs() > 0.25
    if int((big_moves & (vol_ratio > 1.8)).sum()) > 0:
        flags.append("ACTION_REVIEW")

    meta = store.get_meta(ticker)
    if meta.get("fetch_failures", 0) >= 3:
        flags.append("FETCH_UNRELIABLE")
    return flags


def quality_report(ticker: str) -> dict:
    """Rendered by CLI/UI; `usable=False` bars the ticker from screening."""
    df = store.load_prices(ticker)
    if df.empty:
        return {"ticker": ticker, "usable": False, "bars": 0, "last_date": None,
                "stale_days": None, "zero_vol_days": None, "gap_anomalies": None,
                "flags": store.get_meta(ticker).get("quality_flags", [])}
    now = pd.Timestamp.now(tz=df.index.tz) if df.index.tz else pd.Timestamp.now()
    stale = (now - df.index[-1]).days
    return {
        "ticker": ticker,
        "usable": len(df) >= 200 and stale < 10,
        "bars": len(df),
        "last_date": str(df.index[-1].date()),
        "stale_days": stale,
        "zero_vol_days": int((df["volume"].fillna(0) == 0).sum()),
        "gap_anomalies": int((df["px"].pct_change().abs() > 0.25).sum()),
        "flags": store.get_meta(ticker).get("quality_flags", []),
    }


def refresh_universe(with_fundamentals=False) -> dict:
    """The Phase-1 pipeline: incremental refresh of every configured ticker."""
    tickers = config.all_tickers()
    ok = fail = 0
    failures = []
    for i, t in enumerate(tickers, 1):
        if refresh_ticker(t):
            ok += 1
        else:
            fail += 1
            failures.append(t)
        if i % 25 == 0:
            print(f"  {i}/{len(tickers)} refreshed...")

    if with_fundamentals:
        for t in config.basket_tickers():
            refresh_fundamentals(t)

    summary = {"tickers": len(tickers), "ok": ok, "fail": fail,
               "failures": failures[:20]}
    store.audit_log("data_refresh", summary)
    return summary