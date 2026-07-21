"""Deterministically evaluate every Nth successful prompt execution."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..model_config import DEFAULT_JUDGE_MODEL
from ..prompt_catalog import PROMPTS
from ..core import (
    OpenRouterPromptClient,
    PromptRunEvent,
    PromptRuntimeOptions,
    TestJudgment,
)

RunJudge = Callable[[PromptRunEvent], Awaitable[TestJudgment | dict[str, Any]]]
EvaluationSink = Callable[["RunEvaluation"], Any]


class RunEvaluation(BaseModel):
    """The sampled run evidence and its normalized judge score."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    prompt_name: str
    model: str
    system_prompt: str
    input: str
    output: Any
    score: float = Field(ge=0, le=1)
    rationale: str
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OpenRouterRunJudge:
    """Judge one completed run with the versioned sampled-run eval prompt."""

    def __init__(
        self,
        prompt_client: OpenRouterPromptClient | None = None,
        model: str = DEFAULT_JUDGE_MODEL,
    ):
        self.prompt_client = prompt_client or OpenRouterPromptClient()
        self.model = model
        self.prompt = PROMPTS.sampled_run_judge

    async def __call__(self, event: PromptRunEvent) -> TestJudgment:
        if event.type != "response":
            raise ValueError("Only completed response events can be judged.")
        output = event.output
        if isinstance(output, BaseModel):
            output = output.model_dump(mode="json")
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False, default=str)
        prepared = self.prompt.prepare(
            {
                "prompt_name": event.prompt_name,
                "system_prompt": event.system,
                "input": event.user,
                "output": output,
            }
        )
        judgment = await self.prompt_client.execute(
            self.prompt,
            prepared,
            runtime=PromptRuntimeOptions(model=self.model),
            output_model=TestJudgment,
        )
        return (
            judgment
            if isinstance(judgment, TestJudgment)
            else TestJudgment.model_validate(judgment)
        )


class EveryNRunEvalHook:
    """Evaluate every Nth successful response and forward a structured record."""

    def __init__(
        self,
        judge: RunJudge,
        sink: EvaluationSink,
        every: int = 5,
    ):
        if every < 1:
            raise ValueError("every must be at least 1.")
        self.judge = judge
        self.sink = sink
        self.every = every
        self._completed_calls = 0
        self._counter_lock = asyncio.Lock()
        self._pending_tasks: set[asyncio.Task[None]] = set()

    @property
    def completed_calls(self) -> int:
        return self._completed_calls

    @property
    def pending_evaluations(self) -> int:
        return len(self._pending_tasks)

    async def __call__(self, event: PromptRunEvent) -> None:
        if event.type != "response":
            return
        async with self._counter_lock:
            self._completed_calls += 1
            should_evaluate = self._completed_calls % self.every == 0
        if not should_evaluate:
            return
        task = asyncio.create_task(
            self._evaluate(event),
            name=f"evaluate-prompt-run-{event.run_id}",
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._evaluation_finished)

    async def drain(self) -> None:
        if self._pending_tasks:
            await asyncio.gather(*tuple(self._pending_tasks), return_exceptions=True)

    def _evaluation_finished(self, task: asyncio.Task[None]) -> None:
        self._pending_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def _evaluate(self, event: PromptRunEvent) -> None:
        judgment = TestJudgment.model_validate(await self.judge(event))
        evaluation = RunEvaluation(
            run_id=event.run_id,
            prompt_name=event.prompt_name,
            model=event.model,
            system_prompt=event.system,
            input=event.user,
            output=event.output,
            score=judgment.score,
            rationale=judgment.rationale,
        )
        result = self.sink(evaluation)
        if inspect.isawaitable(result):
            await result
