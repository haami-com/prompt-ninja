import asyncio

import pytest

from prompt_ninja.prompt_artifacts import update_prompt_artifact
from prompt_ninja import PromptNinja, PromptNinjaError


DEFINITION = {
    "metadata": {
        "spec_version": "1.2",
        "name": "summary",
        "description": "Summarizes updates.",
        "used_by": ["backend/tests/test_prompt_artifacts.py"],
        "version": "1.0.0",
        "output": "String",
    },
    "llm_model": {"provider": "openrouter", "name": "test/model"},
    "prompt": {"system": "Summarize accurately.", "user": "{{source}}"},
    "variables": [
        {
            "name": "source",
            "type": "string",
            "description": "Update to summarize.",
            "required": True,
        }
    ],
    "testing": {"pass_threshold": 0.9},
    "tests": [
        {
            "name": "preserves dates",
            "variable": {"source": "Launch is July 30."},
            "expected_output": "A summary that preserves July 30.",
        }
    ],
}


class FakePromptClient:
    def __init__(self, response):
        self.response = response

    async def execute(self, *_args, **_kwargs):
        return self.response


def test_update_preserves_tests_and_changes_prompt_implementation():
    original = PromptNinja(DEFINITION)
    response = original.to_toml().replace(
        "Summarize accurately.", "Summarize accurately and preserve every date."
    )

    updated = asyncio.run(
        update_prompt_artifact(
            original,
            "Preserve dates.",
            "test/model",
            prompt_client=FakePromptClient(response),
        )
    )

    assert updated.spec.prompt.system.endswith("preserve every date.")
    assert updated.spec.testing == original.spec.testing
    assert updated.tests == original.tests


def test_update_rejects_a_rewritten_test_contract():
    original = PromptNinja(DEFINITION)
    response = original.to_toml().replace(
        "A summary that preserves July 30.", "Any plausible summary."
    )

    with pytest.raises(PromptNinjaError, match="protected semantic test contract"):
        asyncio.run(
            update_prompt_artifact(
                original,
                "Make the test pass.",
                "test/model",
                prompt_client=FakePromptClient(response),
            )
        )