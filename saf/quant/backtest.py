"""Walk-forward validation of Score v2 core (Option A: T+A+R).
Q=0 and B=5 are constants here -> ranking is purely the price-based core.

Outputs match improvements.txt PART 4 exactly:
  q5_minus_q1_annual, ic_mean, ic_ir, monotonic, turnover_est, decay curve

PATCHED: annualization now uses 252/eval_horizon (was hardcoded *12,
which was only correct for the 21-day horizon). The Shadow Alpha thesis
is medium-term — grade it on the horizon where the thesis lives.

Demotion rule: spread < 3%/yr OR ic_ir < 0.3 (after costs) -> RESEARCH INDICATOR."""
import numpy as np
import pandas as pd
from .. import config, store
from . import score as S

COST_BPS_DEFAULT = 10.0                 # per side; round trip = 2x
HORIZONS_DECAY = (5, 21, 63, 126)
DEFAULT_EVAL_HORIZON = 63               # thesis horizon (quarterly)
MIN_BARS = 250                          # spec: skip names with <250 bars


def _fwd_return(series, date, horizon, cost_bps):
    """Forward return over `horizon` trading days, net of round-trip cost."""
    past = series[series.index <= date]
    if past.empty:
        return None
    entry = past.iloc[-1]
    future = series[series.index > date]
    if len(future) < horizon:
        return None
    return future.iloc[horizon - 1] / entry - 1 - 2 * cost_bps / 10_000


def _regime(spy_px, date):
    """STRESS vs CALM — Shadow Alpha thesis is regime-dependent."""
    past = spy_px[spy_px.index <= date]
    if len(past) < 60:
        return "CALM"
    dd = past.iloc[-1] / past.rolling(60).max().iloc[-1] - 1
    vol = past.pct_change().tail(20).std() * np.sqrt(252)
    vol_hist = (past.pct_change().rolling(20).std().dropna() * np.sqrt(252)).tail(252)
    vol_q75 = vol_hist.quantile(0.75) if len(vol_hist) else 0.20
    return "STRESS" if (dd < -0.05 or vol > vol_q75) else "CALM"


def _assign_quintiles(sub):
    """Per-date cross-sectional quintiles."""
    sub = sub.copy()
    sub["q"] = sub.groupby("date")["score"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
    return sub


def _top_turnover(sub):
    """Fraction of top-quintile membership that changes between rebalances."""
    prev, tots = None, []
    for _, g in sub.groupby("date"):
        top = set(g[g["q"] == g["q"].max()]["ticker"])
        if prev is not None and top:
            tots.append(1 - len(top & prev) / len(top))
        prev = top
    return f"~{int(np.mean(tots) * 100)}%/rebalance" if tots else "n/a"


def evaluate(df, eval_horizon=DEFAULT_EVAL_HORIZON):
    """Spec-exact output dict. eval_horizon selects the forward-return column
    AND the correct annualization factor (252 / horizon)."""
    col = f"fwd_{eval_horizon}"
    sub = df.dropna(subset=[col])
    empty = {"q5_minus_q1_annual": None, "quintiles_annual_pct": None,
             "ic_mean": None, "ic_ir": None, "monotonic": None,
             "turnover_est": "n/a", "n_dates": 0, "n_obs": len(sub),
             "eval_horizon": eval_horizon}
    if len(sub) < 50:
        return empty

    sub = _assign_quintiles(sub)
    per_date = sub.groupby(["date", "q"])[col].mean().unstack("q")

    # ── THE FIX: correct annualization for any horizon ──
    ann_factor = 252 / eval_horizon
    q_means = per_date.mean() * ann_factor * 100

    # Per-date cross-sectional Spearman IC
    ics = []
    for _, g in sub.groupby("date"):
        if len(g) >= 10:
            c = g["score"].corr(g[col], method="spearman")
            if not np.isnan(c):
                ics.append(c)
    ic = pd.Series(ics)

    return {
        "q5_minus_q1_annual": round(q_means.iloc[-1] - q_means.iloc[0], 2),
        "quintiles_annual_pct": q_means.round(2),
        "ic_mean": round(ic.mean(), 4) if len(ic) else None,
        "ic_ir": round(ic.mean() / ic.std(), 3) if len(ic) > 2 and ic.std() > 0 else None,
        "monotonic": bool(q_means.is_monotonic_increasing),
        "turnover_est": _top_turnover(sub),
        "n_dates": per_date.shape[0], "n_obs": len(sub),
        "eval_horizon": eval_horizon,
    }


def decay_curve(df):
    """Raw (non-annualized) Q5-Q1 spread at each horizon.
    Shows where the edge lives: 1 week? 1 month? 1 quarter? 2 quarters?"""
    out = {}
    for h in HORIZONS_DECAY:
        col = f"fwd_{h}"
        sub = df.dropna(subset=[col])
        if len(sub) < 50:
            out[h] = None
            continue
        sub = _assign_quintiles(sub)
        per_date = sub.groupby(["date", "q"])[col].mean().unstack("q")
        qm = per_date.mean() * 100
        out[h] = round(qm.iloc[-1] - qm.iloc[0], 2)
    return out


def verdict(res):
    """The demotion rule from improvements.txt PART 16 — applied mechanically."""
    spread = res.get("q5_minus_q1_annual")
    ic_ir = res.get("ic_ir")
    if spread is None:
        return "INSUFFICIENT_DATA", ("Not enough history/cross-section to judge. "
                                     "Increase lookback_days.")
    spread_ok = spread >= 3.0
    ic_ok = (ic_ir is not None) and ic_ir >= 0.3
    if spread_ok and ic_ok:
        return "VALIDATED", (f"Q5-Q1 {spread:+.2f}%/yr (>=3) and IC_IR {ic_ir} (>=0.3) "
                             f"over the {res.get('eval_horizon')}d horizon, after costs. "
                             "The score's thresholds earn their place.")
    return "DEMOTED", (f"Q5-Q1 {spread:+.2f}%/yr, IC_IR {ic_ir} over the "
                       f"{res.get('eval_horizon')}d horizon, after costs — below the 3% / 0.3 bar. "
                       "Score demoted to RESEARCH INDICATOR: display it, do not screen or size on it.")


def run_backtest(step_days=21, cost_bps=COST_BPS_DEFAULT,
                 eval_horizon=DEFAULT_EVAL_HORIZON, universe=None):
    cfg = config.load()
    bench = cfg["settings"]["benchmark"]
    tickers = [t for t in (universe or config.all_tickers(cfg)) if t != bench]

    prices = {t: store.load_prices(t) for t in tickers + [bench]}
    spy = prices[bench]
    if spy.empty:
        raise RuntimeError(f"No data for {bench}. Run: python -m saf.cli fetch")

    cal = spy.index
    max_h = max(HORIZONS_DECAY)
    if len(cal) < MIN_BARS + max_h + step_days * 3:
        raise RuntimeError("Insufficient history. Raise settings.lookback_days in "
                           "universe.yaml (recommend 1500) and re-run fetch.")

    dates = cal[MIN_BARS:-max_h][::step_days]
    rows = []
    for d in dates:
        regime = _regime(spy["px"], d)
        cross = {}
        for t in tickers:
            px = prices[t]
            if px.empty or len(px[px.index <= d]) < MIN_BARS:
                continue
            s = S.score_v2(t, d, prices, spy, fund=None, rubric=None)  # core mode
            if s:
                cross[t] = s["total"]
        if len(cross) < 25:                     # need a real cross-section
            continue
        for t, total in cross.items():
            row = {"date": d, "ticker": t, "score": total, "regime": regime}
            for h in HORIZONS_DECAY:
                row[f"fwd_{h}"] = _fwd_return(prices[t]["px"], d, h, cost_bps)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No scoreable cross-sections produced.")

    res = evaluate(df, eval_horizon=eval_horizon)
    res["by_regime"] = {r: evaluate(df[df["regime"] == r], eval_horizon=eval_horizon)
                        for r in sorted(df["regime"].unique())}
    res["decay"] = decay_curve(df)
    res["status"], res["verdict_text"] = verdict(res)
    return {"df": df, "results": res}