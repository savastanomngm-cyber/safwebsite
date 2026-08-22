"""Grounded rubric scoring with post-hoc citation verification.
FULL PATCHED FILE — model list rebuilt from the ACTUAL /v1/models output:
  - qwen/qwen3.6-27b first (only model with fresh TPD budget)
  - gpt-oss models as fallbacks (TPD resets daily)
  - NO Llama models (not provisioned on this account)
Also includes: per-model rate-limit fallback, tolerant citation matching,
max_tokens=2048, regex JSON extraction, JSON-mode guard.
"""
import json
import re
import time
from openai import OpenAI
from ..security import load_env

# Rebuilt from `GET /v1/models` on this account — only real chat models.
MODEL_CANDIDATES = [
    "qwen/qwen3.6-27b",        # fresh budget right now (same as legacy MODEL_BACKUP)
    "openai/gpt-oss-20b",      # fast; TPD resets daily
    "openai/gpt-oss-120b",     # best quality; TPD resets daily
]
MAX_RETRIES = 2
RETRY_DELAY = 2

RUBRIC_GROUNDED_SYS = """You are a bottleneck analyst. You will receive an
EVIDENCE PACK containing the company's own business description and fundamentals.
Rules:
1. Score each criterion 1-5 ONLY if the evidence pack contains direct support.
2. For each score, quote the SHORT exact supporting sentence from the pack.
3. If no support exists, score the criterion 2 (neutral) and write "INSUFFICIENT EVIDENCE".
4. Never use your own knowledge to fill gaps.
5. Return ONLY valid JSON — no markdown fences, no commentary:
{"scores": {"market_concentration": N, "substitutability": N, "capital_intensity": N, "regulatory_moat": N, "demand_inelasticity": N, "cross_sector_demand": N},
 "citations": {"market_concentration": "...", "substitutability": "...", "capital_intensity": "...", "regulatory_moat": "...", "demand_inelasticity": "...", "cross_sector_demand": "..."},
 "total": N}"""


def extract_json(text):
    if not text: return None
    try: return json.loads(text)
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    return None


def call_llm(system, user):
    """LLM call with per-model rate-limit fallback.
    Returns (parsed_json_or_None, debug_dict)."""
    key = load_env()
    if not key:
        return None, {"error": "no GROQ_API_KEY in .env"}
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    model_idx, use_json = 0, True
    last_err = ""
    
    for attempt in range(1, MAX_RETRIES + len(MODEL_CANDIDATES) + 1):
        model = MODEL_CANDIDATES[min(model_idx, len(MODEL_CANDIDATES) - 1)]
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.2,
                max_tokens=2048,
            )
            if use_json:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            parsed = extract_json(raw)
            if parsed is not None:
                return parsed, {"model": model, "attempts": attempt}
            last_err = f"unparseable output: {raw[:120]}"
            if use_json:
                use_json = False
                continue
            return None, {"model": model, "error": last_err, "raw": raw[:300]}
        except Exception as e:
            err = str(e)
            last_err = err[:200]
            if "response_format" in err or ("json" in err.lower() and "400" in err):
                use_json = False              # retry without strict JSON mode
                continue
            if "429" in err or "rate" in err.lower():
                # Rate limit -> advance to the next model's budget bucket
                if model_idx < len(MODEL_CANDIDATES) - 1:
                    model_idx += 1
                    continue
                time.sleep(RETRY_DELAY * attempt)   # last model: back off
                continue
            if ("400" in err or "404" in err or "not_found" in err or "decommissioned" in err) \
                    and model_idx < len(MODEL_CANDIDATES) - 1:
                model_idx += 1                # model unavailable -> next
                continue
            return None, {"model": model, "error": last_err}
            
    return None, {"model": MODEL_CANDIDATES[min(model_idx, len(MODEL_CANDIDATES) - 1)],
                  "error": last_err or "exhausted retries"}


# ── citation matching (tolerant — the false-positive fix) ─────────
def _norm(s):
    return re.sub(r"\s+", " ", str(s)).lower().strip()

def _clean_quote(q):
    """Strip ellipsis / surrounding quotes / trailing punctuation so a genuinely
    truncated quote still matches the evidence."""
    q = str(q).strip()
    q = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", q)
    q = re.sub(r"[.…\s]+$", "", q)
    return _norm(q)

def _cite_present(quote, pack_text_norm):
    """True if the quote (or a substantial prefix of it) appears in the pack."""
    c = _clean_quote(quote)
    if not c: return False
    if c in pack_text_norm: return True
    prefix = c[:40]
    return len(prefix) >= 20 and prefix in pack_text_norm


def score_bottleneck(ticker: str, pack: dict) -> dict:
    if not pack.get("business_desc"):
        return {"error": "No business description available"}

    prompt = f"EVIDENCE PACK FOR {ticker}:\n{json.dumps(pack, indent=1)}"
    out, debug = call_llm(RUBRIC_GROUNDED_SYS, prompt)

    if not out or "scores" not in out:
        return {"error": "LLM failed to return valid JSON", "debug": debug}

    # ── POST-HOC VERIFICATION (the hallucination penalty) ──
    pack_text = _norm(pack.get("business_desc", "") + " " +
                      " ".join(pack.get("concentration_hits", [])) + " " +
                      " ".join(pack.get("recent_headlines", [])))

    flagged = []
    citations = out.get("citations", {})
    scores = out.get("scores", {})

    for crit, quote in citations.items():
        if quote and quote != "INSUFFICIENT EVIDENCE":
            if not _cite_present(quote, pack_text):
                if crit in scores:
                    scores[crit] = 2          # hallucinated citation -> neutral
                flagged.append(crit)

    out["scores"] = scores
    out["total"] = sum(v for v in scores.values() if isinstance(v, (int, float)))
    out["flagged_hallucinations"] = flagged
    out["llm_meta"] = debug
    return out