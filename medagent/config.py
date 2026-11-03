"""Central configuration, loaded from environment variables / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "data" / "raw"
DB_DIR = Path(os.getenv("DB_DIR", ROOT / "databases")).resolve()

SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "tavily").lower()
VERBOSE = os.getenv("AGENT_VERBOSE", "false").lower() in {"1", "true", "yes"}

# --- LLM provider (openai | groq | gemini) -------------------------------
# provider: (api-key env var, base_url, default model)
PRESETS = {
    "openai": ("OPENAI_API_KEY", None, "gpt-4o-mini"),
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "gemini": (
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.5-flash",
    ),
}
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
if LLM_PROVIDER not in PRESETS:
    raise RuntimeError(f"LLM_PROVIDER must be one of {list(PRESETS)}, got '{LLM_PROVIDER}'")
_key_var, LLM_BASE_URL, _default_model = PRESETS[LLM_PROVIDER]
LLM_MODEL = os.getenv("LLM_MODEL", _default_model)
LLM_API_KEY = os.getenv(_key_var)


def require_llm_key() -> None:
    if not LLM_API_KEY:
        raise RuntimeError(
            f"{_key_var} is not set (LLM_PROVIDER={LLM_PROVIDER}). "
            "Copy .env.example to .env and fill it in."
        )
