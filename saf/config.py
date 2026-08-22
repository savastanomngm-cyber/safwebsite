"""Config loading with validation. Fails loudly on bad config.
Duplicate tickers across baskets are allowed (by design) and only
reported when SAF_VERBOSE=1."""
import os
import yaml
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).resolve().parent.parent
VERBOSE = bool(os.getenv("SAF_VERBOSE"))


class ConfigError(Exception):
    pass


@lru_cache(maxsize=1)
def load() -> dict:
    path = ROOT / "saf" / "universe.yaml"
    if not path.exists():
        raise ConfigError(f"Missing {path} — run scripts/migrate_baskets.py first")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    _validate(cfg)
    return cfg


def _validate(cfg):
    s = cfg.get("settings", {})
    if "benchmark" not in s:
        raise ConfigError("settings.benchmark required")
    th = s.get("score_thresholds", {})
    if not (0 < th.get("watch", 45) < th.get("candidate", 60) <= 100):
        raise ConfigError("thresholds must satisfy 0 < watch < candidate <= 100")

    seen = set()
    for b in cfg.get("baskets", []):
        name = b.get("name")
        if not name or not b.get("holdings"):
            raise ConfigError(f"basket needs name+holdings: {b}")
        for t, w in b["holdings"].items():
            if not isinstance(w, (int, float)) or w <= 0:
                raise ConfigError(f"basket '{name}': invalid weight for {t}: {w}")
        if VERBOSE:
            dupes = [t for t in b["holdings"] if t in seen]
            if dupes:
                print(f"⚠️  {dupes} appear in multiple baskets (allowed — check intent)")
        wsum = sum(b["holdings"].values())
        if not 0 < wsum < 100:
            raise ConfigError(f"basket '{name}' weight sum suspicious: {wsum}")
        seen.update(b["holdings"])


def baskets(cfg=None) -> dict:
    """name -> holdings dict"""
    cfg = cfg or load()
    return {b["name"]: b["holdings"] for b in cfg["baskets"]}


def basket_tickers(cfg=None) -> list:
    cfg = cfg or load()
    ts = set()
    for b in cfg["baskets"]:
        ts.update(b["holdings"])
    return sorted(ts)


def all_tickers(cfg=None) -> list:
    cfg = cfg or load()
    ts = set(basket_tickers(cfg))
    for group in cfg.get("screening_universe", {}).values():
        ts.update(group)
    ts.add(cfg["settings"]["benchmark"])
    return sorted(ts)


def sector_of(ticker, cfg=None):
    cfg = cfg or load()
    for sector, tickers in cfg.get("screening_universe", {}).items():
        if ticker in tickers:
            return sector
    for b in cfg["baskets"]:
        if ticker in b["holdings"]:
            return b.get("section", "BASKET")
    return None