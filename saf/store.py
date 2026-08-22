"""SQLite persistence. Single source of truth for all state."""
import sqlite3, json, hashlib, time
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "saf.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL, date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume INTEGER,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    ticker TEXT PRIMARY KEY, last_fetch TEXT, last_success TEXT,
    fetch_failures INTEGER DEFAULT 0, quality_flags TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
    kind TEXT NOT NULL, payload TEXT NOT NULL, prev_hash TEXT, hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT NOT NULL, ticker TEXT NOT NULL,
    action TEXT, position_pct REAL, notes TEXT, outcome TEXT, realized_ret REAL, signal_json TEXT
);

CREATE TABLE IF NOT EXISTS rubric_cache (
    ticker TEXT PRIMARY KEY, scored_at TEXT NOT NULL,
    total_score REAL, raw_json TEXT
);
"""

def con() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init():
    with con() as c:
        c.executescript(SCHEMA)

# ── prices ──────────────────────────────────────────────────
def upsert_prices(ticker: str, df: pd.DataFrame):
    rows = [(ticker, str(idx.date()), r.open, r.high, r.low,
             r.close, r.adj_close, int(r.volume)) for idx, r in df.iterrows()]
    with con() as c:
        c.executemany("""INSERT OR REPLACE INTO prices
                         (ticker,date,open,high,low,close,adj_close,volume)
                         VALUES (?,?,?,?,?,?,?,?)""", rows)
        c.execute("""INSERT INTO meta (ticker,last_fetch,last_success)
                     VALUES (?,datetime('now'),datetime('now'))
                     ON CONFLICT(ticker) DO UPDATE SET
                     last_fetch=datetime('now'), last_success=datetime('now')""", (ticker,))

def load_prices(ticker: str) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM prices WHERE ticker=? ORDER BY date", con(), params=(ticker,))
    if df.empty: return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df["px"] = df["adj_close"].fillna(df["close"])
    return df

def last_price_date(ticker: str):
    row = con().execute("SELECT MAX(date) AS d FROM prices WHERE ticker=?", (ticker,)).fetchone()
    return row["d"] if row else None

# ── fundamentals ────────────────────────────────────────────
def save_fundamentals(ticker: str, info: dict):
    with con() as c:
        c.execute("""INSERT OR REPLACE INTO fundamentals (ticker,fetched_at,json)
                     VALUES (?,datetime('now'),?)""", (ticker, json.dumps(info, default=str)))

def get_fundamentals(ticker: str, max_age_days=7):
    row = con().execute("SELECT fetched_at,json FROM fundamentals WHERE ticker=?", (ticker,)).fetchone()
    if not row: return None
    age = (pd.Timestamp.now() - pd.Timestamp(row["fetched_at"])).days
    info = json.loads(row["json"])
    info["_stale_days"] = age
    return info if age <= max_age_days else {**info, "_stale": True}

# ── quality flags ───────────────────────────────────────────
def set_flags(ticker: str, flags: list):
    with con() as c:
        c.execute("""INSERT INTO meta (ticker,quality_flags) VALUES (?,?)
                     ON CONFLICT(ticker) DO UPDATE SET quality_flags=excluded.quality_flags""",
                  (ticker, json.dumps(flags)))

def get_meta(ticker: str) -> dict:
    row = con().execute("SELECT * FROM meta WHERE ticker=?", (ticker,)).fetchone()
    return dict(row) if row else {}

# ── audit log ───────────────────────────────────────────────
def audit_log(kind: str, payload: dict):
    prev = con().execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
    prev_hash = prev["hash"] if prev else "GENESIS"
    entry = {"ts": time.time(), "kind": kind, "payload": payload, "prev": prev_hash}
    h = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
    with con() as c:
        c.execute("""INSERT INTO audit (ts,kind,payload,prev_hash,hash) VALUES (?,?,?,?,?)""",
                  (entry["ts"], kind, json.dumps(payload, default=str), prev_hash, h))

def verify_audit_chain() -> bool:
    rows = con().execute("SELECT * FROM audit ORDER BY id").fetchall()
    prev = "GENESIS"
    for r in rows:
        entry = {"ts": r["ts"], "kind": r["kind"], "payload": json.loads(r["payload"]), "prev": prev}
        expect = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
        if expect != r["hash"]: return False
        prev = r["hash"]
    return True

# ── rubric cache (Phase 4.5) ────────────────────────────────
def save_rubric(ticker: str, total: float, raw_dict: dict):
    with con() as c:
        c.execute("""INSERT OR REPLACE INTO rubric_cache
                     (ticker, scored_at, total_score, raw_json)
                     VALUES (?, datetime('now'), ?, ?)""",
                  (ticker, total, json.dumps(raw_dict, default=str)))

def get_cached_rubric(ticker: str, max_age_days=30) -> dict | None:
    row = con().execute("SELECT scored_at, total_score, raw_json FROM rubric_cache WHERE ticker=?",
                        (ticker,)).fetchone()
    if not row: return None
    age = (pd.Timestamp.now() - pd.Timestamp(row["scored_at"])).days
    if age > max_age_days: return None
    return {"total": row["total_score"], "raw": json.loads(row["raw_json"]), "age_days": age}