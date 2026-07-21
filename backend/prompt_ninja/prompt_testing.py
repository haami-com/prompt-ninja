"""Run an end-to-end synthetic self-test for a generated prompt."""

from __future__ import annotations

import os
import json
import re
from typing import Any

from pydantic import BaseModel

from .models import GeneratedPromptTestRequest, GeneratedPromptTestResult
from .prompt_catalog import PROMPTS
from .core import OpenRouterPromptClient, PromptNinja, PromptRuntimeOptions

_TEMPLATE_VARIABLE_PATTERN = re.compile(
    r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)(?:\s*\|\s*[A-Za-z_][A-Za-z0-9_]*)?\s*\}\}"
)


def fixture_values_for_prompt(prompt: PromptNinja, source_input: str) -> dict[str, Any]:
    """Create type-correct runtime values from one synthetic source fixture."""

    def value_for(variable: Any) -> Any:
        values_by_type = {
            "string": source_input,
            "integer": 1,
            "number": 1.0,
            "boolean": True,
            "json": {"input": source_input},
            "dict": {"input": source_input},
            "array": [source_input],
            "object": {"input": source_input},
            "date": "2026-01-01",
            "datetime": "2026-01-01T00:00:00",
            "dynamic": source_input,
        }
        if variable.list_item_type is not None:
            item = variable.model_copy(update={"type": variable.list_item_type})
            return [value_for(item)]
        if variable.model_class is not None:
            return variable.model_class.model_construct()
        return values_by_type[variable.type]

    return {
        name: value_for(variable)
        for name, variable in prompt.variables.items()
        if variable.required or not variable.has_default
    }


class PromptTestHarness:
    """Generates a fixture, runs a prompt, then judges the result semantically."""

    def __init__(
        self,
        client: Any | None = None,
        prompt_client: OpenRouterPromptClient | None = None,
    ):
        self.prompt_client = prompt_client
        if self.prompt_client is None and (
            client is not None or os.getenv("OPENROUTER_API_KEY")
        ):
            self.prompt_client = OpenRouterPromptClient(client)
        self.fixture_generator = PROMPTS.test_case_generator
        self.judge = PROMPTS.test_judge

    @property
    def enabled(self) -> bool:
        return self.prompt_client is not None

    async def _run_prompt(
        self, prompt: PromptNinja, values: dict[str, Any], model: str
    ) -> Any:
        if self.prompt_client is None:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required to run generated-prompt tests."
            )
        prepared = prompt.prepare(values)
        result = await self.prompt_client.execute(
            prompt,
            prepared,
            runtime=PromptRuntimeOptions(model=model),
        )
        return result.model_dump() if isinstance(result, BaseModel) else result

    async def run(
        self, request: GeneratedPromptTestRequest
    ) -> GeneratedPromptTestResult:
        if self.prompt_client is None:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required to run generated-prompt tests."
            )
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
            generated_prompt = PromptNinja(
                request.definition, source="<generated prompt test>"
            )
        else:
            generated_variable_names = sorted(
                set(_TEMPLATE_VARIABLE_PATTERN.findall(request.final_prompt))
                - {"input"}
            )
            generated_prompt = PromptNinja(
                {
                    "metadata": {
                        "spec_version": "1.2",
                        "name": "generated_prompt_under_test",
                        "description": "The in-memory prompt produced by the Board of Prompts.",
                        "used_by": ["backend/prompt_ninja/prompt_testing.py"],
                        "version": "1.0.0",
                        "output": (
                            "String"
                            if fixture_output_format == "text"
                            else "prompt_ninja.JsonObjectOutput"
                        ),
                    },
                    "llm_model": {"provider": "openrouter", "name": request.model},
                    "prompt": {"system": request.final_prompt, "user": "{{input}}"},
                    "variables": [
                        {
                            "name": "input",
                            "type": "string",
                            "description": "The source text supplied to the generated prompt.",
                            "required": True,
                        },
                        *[
                            {
                                "name": name,
                                "type": "string",
                                "description": "The %s supplied to the generated prompt."
                                % name.replace("_", " "),
                                "required": True,
                            }
                            for name in generated_variable_names
                        ],
                    ],
                },
                source="<generated prompt>",
            )
        generated_values = fixture_values_for_prompt(
            generated_prompt, fixture["input"]
        )
        tested_prompt = generated_prompt.prepare(generated_values)
        actual = await self._run_prompt(
            generated_prompt,
            generated_values,
            request.model,
        )
        actual_output = (
            json.dumps(actual, ensure_ascii=False)
            if isinstance(actual, (dict, list))
            else actual
        )
        verdict = await self._run_prompt(
            self.judge,
            {
                "prompt_system": tested_prompt.system,
                "prompt_user": tested_prompt.user,
                "test_input": json.dumps(generated_values, ensure_ascii=False),
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
