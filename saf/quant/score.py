"""Shadow Alpha Score v2 — spec-compliant (improvements.txt PART 3).
S2 = T(25) + A(30) + R(20) + Q(15) + B(10) = 100
A and R can go NEGATIVE to punish uncorrelated disasters (by design)."""
import numpy as np
import pandas as pd

WEIGHTS = {"T": 25, "A": 30, "R": 20, "Q": 15, "B": 10}
MOM6, MOM3 = 126, 63

def _asof(df, upto):
    return df[df.index <= pd.Timestamp(upto)]

def t_score(px, upto):
    s = _asof(px, upto)["px"].dropna()
    if len(s) < MOM6: return None
    r6 = s.pct_change(MOM6).iloc[-1]
    vol = s.pct_change().std() * np.sqrt(252)
    sharpe6 = r6 / (vol * np.sqrt(0.5)) if vol and vol > 0 else 0.0
    above50 = int(s.iloc[-1] > s.rolling(50).mean().iloc[-1])
    return float(np.clip(12.5 * sharpe6 + 6 * above50, 0, 25))

def a_score(px, spy, upto):
    p, b = _asof(px, upto)["px"].dropna(), _asof(spy, upto)["px"].dropna()
    common = p.index.intersection(b.index)
    if len(common) < MOM6: return None
    rt, rs = p[common].pct_change().dropna(), b[common].pct_change().dropna()
    c2 = rt.index.intersection(rs.index); rt, rs = rt[c2], rs[c2]
    if len(rt) < 120 or rs.var() == 0 or rt.std() == 0: return 0.0
    beta = rt.cov(rs) / rs.var()
    resid = rt - beta * rs
    if resid.std() == 0: return 0.0
    ir = (resid.mean() / resid.std()) * np.sqrt(252)
    corr = rt.corr(rs)
    gate = np.tanh(resid.sum() * 10)
    raw = (1 - abs(corr)) * 15 + np.clip(ir, -2, 2) * 7.5
    return float(np.clip(gate * raw, -15, 30))

def r_score(px, spy, upto):
    p, b = _asof(px, upto)["px"].dropna(), _asof(spy, upto)["px"].dropna()
    common = p.index.intersection(b.index)
    if len(common) < MOM3 + 10: return None
    rt = p[common].pct_change(MOM3).iloc[-1]
    rs = b[common].pct_change(MOM3).iloc[-1]
    vol = p[common].pct_change().std() * np.sqrt(252)
    return float(np.clip((rt - rs) / max(vol * np.sqrt(0.25), 0.05) * 5, -10, 20))

def q_score(fund):
    if not fund: return 0.0
    gm = fund.get("gross_margin") or 0
    om = fund.get("oper_margin") or 0
    roe = fund.get("returnOnEquity") or 0
    s = np.clip((gm - 0.25) / 0.45, 0, 1) * 8
    s += np.clip(om / max(gm, 0.01), 0, 1) * 4
    s += np.clip(roe / 0.30, 0, 1) * 3
    return float(np.clip(s, 0, 15))

def b_score(ticker, rubric=None):
    """Bottleneck prior (10). Reads from SQLite cache if rubric not passed directly.
    Neutral prior (5.0) if no cache exists, so unscored tickers aren't penalized."""
    if rubric and isinstance(rubric, dict) and "total" in rubric:
        total = rubric["total"]
    else:
        from .. import store
        cached = store.get_cached_rubric(ticker)
        if cached:
            total = cached.get("total", 15)
        else:
            total = 15  # 15/30 maps to 5.0/10 (neutral prior)
    return float(np.clip(total / 30 * 10, 0, 10))

def confidence_flag(px, fund):
    s = px["px"].dropna()
    now = pd.Timestamp.now(tz=s.index.tz) if s.index.tz else pd.Timestamp.now()
    fresh = (now - s.index[-1]).days < 7
    has_fund = bool(fund and fund.get("gross_margin") is not None)
    if len(s) >= 250 and fresh and has_fund: return "HIGH"
    if len(s) >= 150 and fresh:              return "MEDIUM"
    return "LOW"

def score_v2(ticker, upto, prices, spy, fund=None, rubric=None):
    px = prices.get(ticker)
    if px is None or px.empty: return None
    T, A, R = t_score(px, upto), a_score(px, spy, upto), r_score(px, spy, upto)
    if T is None or A is None or R is None: return None
    
    # Pass the ticker to b_score so it can lookup the DB cache!
    comps = {"trend": round(T, 2), "alpha_indep": round(A, 2),
             "rel_strength": round(R, 2), "quality": round(q_score(fund), 2),
             "bottleneck_prior": round(b_score(ticker, rubric), 2)}
    total = round(sum(comps.values()), 2)
    return {"ticker": ticker, "components": comps, "total": total,
            "verdict": ("CANDIDATE" if total >= 60 else "WATCH" if total >= 45 else "PASS"),
            "confidence": confidence_flag(px, fund)}