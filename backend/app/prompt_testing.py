"""Run an end-to-end synthetic self-test for a generated prompt."""

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import GeneratedPromptTestRequest, GeneratedPromptTestResult
from .prompt_ninja import OpenAIPromptClient, PromptNinja, PromptRuntimeOptions


PROMPTS_DIRECTORY = Path(__file__).resolve().parents[1] / "prompts"
_TEMPLATE_VARIABLE_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def fixture_values_for_prompt(prompt: PromptNinja, source_input: str) -> dict[str, Any]:
    """Create type-correct runtime values from one synthetic source fixture."""
    values_by_type = {
        "string": source_input,
        "integer": 1,
        "number": 1.0,
        "boolean": True,
        "array": [source_input],
        "object": {"input": source_input},
    }
    return {
        name: values_by_type[variable.type]
        for name, variable in prompt.variables.items()
        if variable.required or not variable.has_default
    }


class PromptTestHarness:
    """Generates a fixture, runs a prompt, then judges the result semantically."""

    def __init__(self, client: Any | None = None, prompt_client: OpenAIPromptClient | None = None):
        self.prompt_client = prompt_client
        if self.prompt_client is None and (client is not None or os.getenv("OPENAI_API_KEY")):
            self.prompt_client = OpenAIPromptClient(client)
        self.fixture_generator = PromptNinja.from_file(PROMPTS_DIRECTORY / "test-case-generator.prompt.toml")
        self.judge = PromptNinja.from_file(PROMPTS_DIRECTORY / "test-judge.prompt.toml")

    @property
    def enabled(self) -> bool:
        return self.prompt_client is not None

    async def _run_prompt(self, prompt: PromptNinja, values: dict[str, Any], model: str) -> Any:
        if self.prompt_client is None:
            raise RuntimeError("OPENAI_API_KEY is required to run generated-prompt tests.")
        prepared = prompt.prepare(values)
        result = await self.prompt_client.execute(
            prompt,
            prepared,
            runtime=PromptRuntimeOptions(model=model),
        )
        return result.model_dump() if isinstance(result, BaseModel) else result

    async def run(self, request: GeneratedPromptTestRequest) -> GeneratedPromptTestResult:
        if self.prompt_client is None:
            raise RuntimeError("OPENAI_API_KEY is required to run generated-prompt tests.")
        fixture = await self._run_prompt(
            self.fixture_generator,
            {
                "goal": request.goal,
                "context": request.context,
                "user_expectation": request.expected_output,
            },
            self.fixture_generator.spec.model.name,
        )
        fixture_output_format = fixture["output_format"]
        if fixture_output_format not in {"text", "json"}:
            raise ValueError("Test fixture output_format must be 'text' or 'json'.")
        if request.definition is not None:
            generated_prompt = PromptNinja(request.definition, source="<generated prompt test>")
        else:
            generated_variable_names = sorted(
                set(_TEMPLATE_VARIABLE_PATTERN.findall(request.final_prompt)) - {"input"}
            )
            generated_prompt = PromptNinja({
                "spec_version": "1.0",
                "prompt": {
                    "name": "generated_prompt_under_test",
                    "description": "The in-memory prompt produced by the Board of Prompts.",
                    "used_in": ["backend/app/prompt_testing.py"],
                },
                "model": {"provider": "openai", "name": request.model},
                "template": {"system": request.final_prompt, "user": "{{input}}"},
                "variables": [
                    {"name": "input", "type": "string", "required": True},
                    *[
                        {"name": name, "type": "string", "required": True}
                        for name in generated_variable_names
                    ],
                ],
                "output": (
                    "String"
                    if fixture_output_format == "text"
                    else "app.prompt_ninja.JsonObjectOutput"
                ),
            }, source="<generated prompt>")
        actual = await self._run_prompt(
            generated_prompt,
            fixture_values_for_prompt(generated_prompt, fixture["input"]),
            request.model,
        )
        actual_output = json.dumps(actual, ensure_ascii=False) if isinstance(actual, (dict, list)) else actual
        verdict = await self._run_prompt(
            self.judge,
            {
                "expected_output": fixture["expected_output"],
                "actual_output": actual_output,
            },
            request.judge_model,
        )
        score = float(verdict["score"])
        return GeneratedPromptTestResult(
            model=request.model,
            input=fixture["input"],
            expected_output=fixture["expected_output"],
            actual_output=actual_output,
            score=score,
            passed=score >= self.judge.spec.testing.pass_threshold,
            rationale=verdict["rationale"],
        )
