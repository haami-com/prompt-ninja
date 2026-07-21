"""Token usage and cost telemetry for completed prompt runs."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..model_config import available_models
from ..core import PromptRunEvent

PricingResolver = Callable[[str], Awaitable[dict[str, Any] | None]]
UsageSink = Callable[["RunUsage"], Any]


class RunUsage(BaseModel):
    """Normalized token usage and estimated OpenRouter cost for one run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    prompt_name: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    input_cost: float | None = Field(default=None, ge=0)
    output_cost: float | None = Field(default=None, ge=0)
    total_cost: float | None = Field(default=None, ge=0)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


async def openrouter_pricing(model_id: str) -> dict[str, Any] | None:
    for model in await available_models():
        if model.get("id") == model_id:
            pricing = model.get("pricing")
            return pricing if isinstance(pricing, dict) else None
    return None


class TokenUsageCostHook:
    """Record response token usage and estimate its cost without delaying the run."""

    def __init__(
        self,
        sink: UsageSink,
        pricing_resolver: PricingResolver = openrouter_pricing,
    ):
        self.sink = sink
        self.pricing_resolver = pricing_resolver
        self._pending_tasks: set[asyncio.Task[None]] = set()

    @property
    def pending_records(self) -> int:
        return len(self._pending_tasks)

    async def __call__(self, event: PromptRunEvent) -> None:
        if (
            event.type != "response"
            or event.input_tokens is None
            or event.output_tokens is None
        ):
            return
        task = asyncio.create_task(
            self._record(event),
            name=f"record-prompt-usage-{event.run_id}",
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._record_finished)

    async def drain(self) -> None:
        if self._pending_tasks:
            await asyncio.gather(*tuple(self._pending_tasks), return_exceptions=True)

    def _record_finished(self, task: asyncio.Task[None]) -> None:
        self._pending_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def _record(self, event: PromptRunEvent) -> None:
        pricing = await self.pricing_resolver(event.model)
        input_cost = _token_cost(event.input_tokens or 0, pricing, "prompt")
        output_cost = _token_cost(event.output_tokens or 0, pricing, "completion")
        total_cost = (
            input_cost + output_cost
            if input_cost is not None and output_cost is not None
            else None
        )
        usage = RunUsage(
            run_id=event.run_id,
            prompt_name=event.prompt_name,
            model=event.model,
            input_tokens=event.input_tokens or 0,
            output_tokens=event.output_tokens or 0,
            total_tokens=(
                event.total_tokens
                if event.total_tokens is not None
                else (event.input_tokens or 0) + (event.output_tokens or 0)
            ),
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
        )
        result = self.sink(usage)
        if inspect.isawaitable(result):
            await result


def _token_cost(
    tokens: int,
    pricing: dict[str, Any] | None,
    key: str,
) -> float | None:
    if not pricing or pricing.get(key) is None:
        return None
    try:
        return float(Decimal(tokens) * Decimal(str(pricing[key])))
    except (InvalidOperation, TypeError, ValueError):
        return None
