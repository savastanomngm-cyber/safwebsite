"""Security hardening (improvements.txt PART 10)."""
import os
import re
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env() -> str:
    """Load .env from project root. Keys live server-side ONLY."""
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return os.getenv("GROQ_API_KEY", "")


def clean_text(s, maxlen: int = 300) -> str:
    """Sanitize ANY feed content before it touches storage or the browser."""
    s = html.unescape(str(s))
    s = re.sub(r"<[^>]*>", "", s)                          # strip tags entirely
    s = re.sub(r"(javascript|data|vbscript)\s*:", "", s, flags=re.I)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)     # control chars
    return s[:maxlen].strip()