"""Run an end-to-end synthetic self-test for a generated prompt."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

from .models import GeneratedPromptTestRequest, GeneratedPromptTestResult
from .prompt_ninja import PromptNinja


PROMPTS_DIRECTORY = Path(__file__).resolve().parents[1] / "prompts"


class PromptTestHarness:
    """Generates a fixture, runs a prompt, then judges the result semantically."""

    def __init__(self, client: Any | None = None):
        self.client = client
        if self.client is None and os.getenv("OPENAI_API_KEY"):
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI()
        self.fixture_generator = PromptNinja.from_file(PROMPTS_DIRECTORY / "test-case-generator.prompt.toml")
        self.judge = PromptNinja.from_file(PROMPTS_DIRECTORY / "test-judge.prompt.toml")

    @property
    def enabled(self) -> bool:
        return self.client is not None

    async def run(self, request: GeneratedPromptTestRequest) -> GeneratedPromptTestResult:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is required to run generated-prompt tests.")
        fixture = await self.fixture_generator.run_openai(
            {
                "goal": request.goal,
                "context": request.context,
                "user_expectation": request.expected_output,
            },
            client=self.client,
            model=request.judge_model,
        )
        output_format = fixture["output_format"]
        expected_schema = fixture["expected_schema"]
        if output_format not in {"text", "json"}:
            raise ValueError("Test fixture output_format must be 'text' or 'json'.")
        generated_prompt = PromptNinja(
            {
                "spec_version": "1.0",
                "prompt": {
                    "name": "generated_prompt_under_test",
                    "description": "The in-memory prompt produced by the council.",
                    "used_in": ["prompt-ninja-ui"],
                },
                "model": {"provider": "openai", "name": request.model},
                "template": {"system": request.final_prompt, "user": "{{input}}"},
                "variables": [{"name": "input", "type": "string", "required": True}],
                "output": (
                    {"format": "json", "schema": expected_schema}
                    if output_format == "json"
                    else {"format": "text"}
                ),
            },
            source="<generated prompt>",
        )
        actual = await generated_prompt.run_openai(
            {"input": fixture["input"]},
            client=self.client,
            model=request.model,
        )
        actual_output = json.dumps(actual, ensure_ascii=False) if isinstance(actual, (dict, list)) else actual
        verdict = await self.judge.run_openai(
            {
                "expected_output": fixture["expected_output"],
                "actual_output": actual_output,
            },
            client=self.client,
            model=request.judge_model,
        )
        score = float(verdict["score"])
        return GeneratedPromptTestResult(
            model=request.model,
            input=fixture["input"],
            expected_output=fixture["expected_output"],
            expected_schema=expected_schema,
            schema_valid=True,
            actual_output=actual_output,
            score=score,
            passed=score >= self.judge.spec.testing.pass_threshold,
            rationale=verdict["rationale"],
        )
