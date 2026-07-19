"""Models that this Board of Prompts deployment may offer to callers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.6-sol")


def available_models() -> tuple[str, ...]:
    """Return the deployment's opt-in model allowlist, with a safe fallback."""
    configured = os.getenv("OPENAI_ALLOWED_MODELS", DEFAULT_MODEL)
    models = tuple(dict.fromkeys(model.strip() for model in configured.split(",") if model.strip()))
    return tuple(dict.fromkeys((DEFAULT_MODEL, *models)))
