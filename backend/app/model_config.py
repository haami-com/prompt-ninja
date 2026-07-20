"""OpenRouter configuration and its cacheable model catalogue."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/")
DEFAULT_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "google/gemini-2.5-flash")
MODEL_CACHE_SECONDS = 300
_catalogue: tuple[dict[str, Any], ...] = ()
_catalogue_loaded_at = 0.0


def configured_models() -> tuple[str, ...]:
    """Return an optional model allowlist and a safe offline fallback."""
    configured = os.getenv("OPENROUTER_ALLOWED_MODELS", DEFAULT_MODEL)
    models = tuple(
        dict.fromkeys(model.strip() for model in configured.split(",") if model.strip())
    )
    return tuple(dict.fromkeys((DEFAULT_MODEL, *models)))


def openrouter_headers() -> dict[str, str]:
    """Build optional attribution and authentication headers for OpenRouter."""
    headers: dict[str, str] = {}
    if api_key := os.getenv("OPENROUTER_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    if referer := os.getenv("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = referer
    if title := os.getenv("OPENROUTER_APP_TITLE"):
        headers["X-OpenRouter-Title"] = title
    return headers


def _normalise_model(model: dict[str, Any]) -> dict[str, Any] | None:
    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    pricing = model.get("pricing")
    return {
        "id": model_id,
        "name": model.get("name") if isinstance(model.get("name"), str) else model_id,
        "context_length": model.get("context_length"),
        "pricing": pricing if isinstance(pricing, dict) else {},
        "supported_parameters": model.get("supported_parameters", []),
    }


async def model_catalogue() -> tuple[dict[str, Any], ...]:
    """Return models from OpenRouter, including the prices supplied by its API."""
    global _catalogue, _catalogue_loaded_at
    if _catalogue and time.monotonic() - _catalogue_loaded_at < MODEL_CACHE_SECONDS:
        return _catalogue
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{OPENROUTER_BASE_URL}/models", headers=openrouter_headers()
            )
            response.raise_for_status()
        payload = response.json()
        raw_models = payload.get("data", []) if isinstance(payload, dict) else []
        models = tuple(
            normalised
            for item in raw_models
            if isinstance(item, dict)
            if (normalised := _normalise_model(item)) is not None
        )
        if models:
            _catalogue = models
            _catalogue_loaded_at = time.monotonic()
    except (httpx.HTTPError, ValueError):
        # The configured fallback keeps local development usable during an outage.
        pass
    return _catalogue


async def available_models() -> tuple[dict[str, Any], ...]:
    """Return models permitted for this structured-output application."""
    configured = os.getenv("OPENROUTER_ALLOWED_MODELS")
    catalogue = await model_catalogue()
    if catalogue:
        if configured:
            allowed = set(configured_models())
            catalogue = tuple(model for model in catalogue if model["id"] in allowed)
        structured = tuple(
            model
            for model in catalogue
            if "structured_outputs" in model["supported_parameters"]
        )
        return structured or catalogue
    return tuple(
        {
            "id": model,
            "name": model,
            "context_length": None,
            "pricing": {},
            "supported_parameters": [],
        }
        for model in configured_models()
    )


async def is_available_model(model_id: str) -> bool:
    """Check an incoming selection against the live catalogue or configured fallback."""
    return model_id in {model["id"] for model in await available_models()}
