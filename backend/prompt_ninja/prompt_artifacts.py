"""Update prompt implementations while preserving their behavioral contracts."""

from __future__ import annotations

import tomllib
import json
from typing import Any

from pydantic import BaseModel

from .prompt_catalog import PROMPTS
from .core import (
    OpenRouterPromptClient,
    PromptNinja,
    PromptNinjaError,
    PromptRuntimeOptions,
    PromptTestReport,
)


def _contract(prompt: PromptNinja) -> dict[str, Any]:
    return {
        "testing": prompt.spec.testing.model_dump(mode="json"),
        "tests": [test.model_dump(mode="json") for test in prompt.tests],
    }


async def run_prompt_artifact_tests(
    prompt: PromptNinja,
    judge_model: str,
    *,
    prompt_client: OpenRouterPromptClient | None = None,
) -> PromptTestReport:
    """Run every embedded test and return complete semantic diagnostics."""
    if not prompt.tests:
        return PromptTestReport(prompt_name=prompt.name, results=())
    owned_client = prompt_client is None
    client = prompt_client or OpenRouterPromptClient()

    async def execute(prepared):
        return await client.execute(prompt, prepared)

    async def judge(test, actual):
        if isinstance(actual, BaseModel):
            actual = actual.model_dump(mode="json")
        tested_prompt = prompt.prepare(test.input)
        expected_output = test.expected_output
        if isinstance(expected_output, dict):
            expected_output = json.dumps(expected_output, ensure_ascii=False)
        prepared = PROMPTS.test_judge.prepare(
            {
                "prompt_system": tested_prompt.system,
                "prompt_user": tested_prompt.user,
                "test_input": json.dumps(test.input, ensure_ascii=False),
                "expected_output": expected_output,
                "actual_output": json.dumps(actual, ensure_ascii=False),
            }
        )
        return await client.execute(
            PROMPTS.test_judge,
            prepared,
            runtime=PromptRuntimeOptions(model=judge_model),
        )

    try:
        return await prompt.arun_tests(execute, judge=judge)
    finally:
        if owned_client:
            await client.aclose()


async def update_prompt_artifact(
    prompt: PromptNinja,
    feedback: str,
    model: str,
    *,
    prompt_client: OpenRouterPromptClient | None = None,
) -> PromptNinja:
    """Rewrite a prompt artifact without allowing its tests to change."""
    owned_client = prompt_client is None
    client = prompt_client or OpenRouterPromptClient()
    updater = PROMPTS.prompt_updater
    try:
        prepared = updater.prepare(
            {"prompt_toml": prompt.to_toml(), "feedback": feedback}
        )
        candidate_text = await client.execute(
            updater,
            prepared,
            runtime=PromptRuntimeOptions(model=model),
        )
    finally:
        if owned_client:
            await client.aclose()

    stripped = candidate_text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = "\n".join(stripped.splitlines()[1:-1]).strip()
    try:
        candidate = PromptNinja(
            tomllib.loads(stripped), source="<prompt update candidate>"
        )
    except (tomllib.TOMLDecodeError, PromptNinjaError) as exc:
        raise PromptNinjaError(
            "The updater returned an invalid prompt artifact: %s" % exc
        ) from exc
    if _contract(candidate) != _contract(prompt):
        raise PromptNinjaError(
            "The updater changed the protected semantic test contract. "
            "Edit tests explicitly before updating the prompt implementation."
        )
    return candidate