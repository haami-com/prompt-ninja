import asyncio

import pytest

from prompt_ninja.hooks import EveryNRunEvalHook, RunEvaluation
from prompt_ninja.hooks.runtime import InMemoryEvaluationStore
from prompt_ninja import PromptRunEvent, TestJudgment as Judgment


def response_event(index: int) -> PromptRunEvent:
    return PromptRunEvent(
        type="response",
        run_id=f"run-{index}",
        prompt_name="weekly-summary",
        provider="openrouter",
        model="google/gemini-3.5-flash",
        system="Summarize the update without inventing facts.",
        user=f"Project update {index}",
        output=f"Summary {index}",
    )


def test_hook_evaluates_every_fifth_successful_response():
    judged = []
    evaluations = []

    async def judge(event):
        judged.append(event.run_id)
        return Judgment(score=0.8, rationale="Mostly follows the prompt.")

    hook = EveryNRunEvalHook(judge=judge, sink=evaluations.append, every=5)

    async def run_calls():
        for index in range(1, 11):
            await hook(response_event(index))
        await hook.drain()

    asyncio.run(run_calls())

    assert judged == ["run-5", "run-10"]
    assert [evaluation.run_id for evaluation in evaluations] == ["run-5", "run-10"]
    assert all(isinstance(evaluation, RunEvaluation) for evaluation in evaluations)
    assert evaluations[0].score == 0.8
    assert evaluations[0].input == "Project update 5"


def test_hook_does_not_wait_for_evaluation_to_finish():
    evaluations = []

    async def run_call():
        judge_started = asyncio.Event()
        release_judge = asyncio.Event()

        async def judge(_event):
            judge_started.set()
            await release_judge.wait()
            return Judgment(score=1.0, rationale="Follows the prompt.")

        hook = EveryNRunEvalHook(judge=judge, sink=evaluations.append, every=1)
        await hook(response_event(1))
        await judge_started.wait()

        assert hook.pending_evaluations == 1
        assert evaluations == []

        release_judge.set()
        await hook.drain()
        assert hook.pending_evaluations == 0

    asyncio.run(run_call())

    assert [evaluation.run_id for evaluation in evaluations] == ["run-1"]


def test_hook_ignores_request_and_error_events():
    evaluations = []

    async def judge(_event):
        return {"score": 1.0, "rationale": "Good."}

    hook = EveryNRunEvalHook(judge=judge, sink=evaluations.append, every=1)
    base = response_event(1).model_dump()

    async def emit_non_responses():
        await hook(PromptRunEvent(**{**base, "type": "request", "output": None}))
        await hook(
            PromptRunEvent(
                **{**base, "type": "error", "error": "failed", "output": None}
            )
        )

    asyncio.run(emit_non_responses())

    assert hook.completed_calls == 0
    assert evaluations == []


def test_hook_rejects_invalid_interval():
    with pytest.raises(ValueError, match="at least 1"):
        EveryNRunEvalHook(judge=lambda _: None, sink=lambda _: None, every=0)


def test_in_memory_store_is_bounded_and_returns_newest_first():
    store = InMemoryEvaluationStore(max_records=2)
    base = {
        "prompt_name": "creator_1",
        "model": "openai/gpt-5.6-luna",
        "system_prompt": "Create a prompt.",
        "input": "Summarize a report.",
        "output": "A generated prompt.",
        "score": 0.9,
        "rationale": "Strong result.",
    }
    for index in range(3):
        store(RunEvaluation(run_id=f"run-{index}", **base))

    assert [record.run_id for record in store.snapshot()] == ["run-2", "run-1"]
