import asyncio

import pytest

from prompt_ninja.hooks.usage import RunUsage, TokenUsageCostHook
from prompt_ninja import PromptRunEvent


def usage_event(**overrides) -> PromptRunEvent:
    values = {
        "type": "response",
        "run_id": "run-2",
        "prompt_name": "creator-2",
        "provider": "openrouter",
        "model": "example/model",
        "system": "Create a prompt.",
        "user": "Summarize this report.",
        "output": {"draft": "A useful prompt."},
        "input_tokens": 1000,
        "output_tokens": 250,
        "total_tokens": 1250,
    }
    values.update(overrides)
    return PromptRunEvent(**values)


def test_usage_hook_records_tokens_and_cost_without_blocking():
    records = []

    async def run_hook():
        pricing_started = asyncio.Event()
        release_pricing = asyncio.Event()

        async def pricing(_model):
            pricing_started.set()
            await release_pricing.wait()
            return {"prompt": "0.000001", "completion": "0.000002"}

        hook = TokenUsageCostHook(records.append, pricing_resolver=pricing)
        await hook(usage_event())
        await pricing_started.wait()

        assert hook.pending_records == 1
        assert records == []

        release_pricing.set()
        await hook.drain()

    asyncio.run(run_hook())

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, RunUsage)
    assert record.input_tokens == 1000
    assert record.output_tokens == 250
    assert record.total_tokens == 1250
    assert record.input_cost == pytest.approx(0.001)
    assert record.output_cost == pytest.approx(0.0005)
    assert record.total_cost == pytest.approx(0.0015)


def test_usage_hook_ignores_events_without_response_usage():
    records = []

    async def pricing(_model):
        raise AssertionError("pricing should not be requested")

    hook = TokenUsageCostHook(records.append, pricing_resolver=pricing)

    async def run_events():
        await hook(usage_event(type="request", output=None))
        await hook(
            usage_event(input_tokens=None, output_tokens=None, total_tokens=None)
        )

    asyncio.run(run_events())

    assert records == []


def test_usage_hook_keeps_tokens_when_pricing_is_unavailable():
    records = []

    async def pricing(_model):
        return None

    hook = TokenUsageCostHook(records.append, pricing_resolver=pricing)

    async def run_hook():
        await hook(usage_event(total_tokens=None))
        await hook.drain()

    asyncio.run(run_hook())

    assert records[0].total_tokens == 1250
    assert records[0].total_cost is None
