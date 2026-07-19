import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.prompt_compiler import CompiledPromptResult
from app.prompt_ninja import PromptFileSpec, PromptNinja


def canonical_definition():
    return {
        "spec_version": "1.0",
        "prompt": {
            "name": "code-review",
            "description": "Reviews a user specification.",
            "used_in": ["src/prompt_consumer.py"],
        },
        "model": {"provider": "openai", "name": "gpt-5.6-sol"},
        "template": {
            "system": "Review the specification and return advice.",
            "user": "{{USER_SPECIFICATION}}",
        },
        "variables": [{
            "name": "USER_SPECIFICATION",
            "type": "string",
            "required": True,
            "description": "The specification to review.",
        }],
        "output": "String",
        "testing": {"pass_threshold": 0.95},
        "tests": [{
            "name": "summarize_complex_spec",
            "input": {"USER_SPECIFICATION": "Do not mutate the input."},
            "expected_output": "A review that preserves the constraint.",
        }],
    }


def test_compiler_structured_output_is_a_nested_pydantic_model():
    result = CompiledPromptResult.model_validate({
        "definition": canonical_definition(),
    })

    assert isinstance(result.definition, PromptFileSpec)
    assert result.definition.variables[0].name == "USER_SPECIFICATION"
    assert result.definition.output == "String"


def test_compiler_does_not_normalize_an_incompatible_schema_dialect():
    malformed = canonical_definition()
    malformed["variables"] = {
        "USER_SPECIFICATION": {"type": "string"},
    }
    malformed["output"] = {
        "type": "string",
        "description": "A concise review.",
    }
    malformed["tests"][0]["variables"] = malformed["tests"][0].pop("input")

    with pytest.raises(ValidationError):
        CompiledPromptResult.model_validate({"definition": malformed})


def test_prompt_toml_resolves_its_declared_pydantic_output_model():
    compiler_prompt = PromptNinja.from_file(
        "prompts/prompt-compiler.prompt.toml"
    )

    assert compiler_prompt.output_model is CompiledPromptResult
    assert compiler_prompt.output_format == "json"


def test_compiler_passes_its_toml_declared_model_to_responses_parse():
    compiler_prompt = PromptNinja.from_file(
        "prompts/prompt-compiler.prompt.toml"
    )

    class FakeResponses:
        async def parse(self, **request):
            self.request = request
            return SimpleNamespace(output_parsed=CompiledPromptResult(
                definition=canonical_definition(),
            ))

    responses = FakeResponses()
    result = asyncio.run(compiler_prompt.run_openai(
        {
            "goal": "Review a specification",
            "model": "gpt-5.6-sol",
            "requirements": {},
            "candidate_prompt": "Review the specification.",
            "test_result": {},
        },
        client=SimpleNamespace(responses=responses),
    ))

    assert isinstance(result, CompiledPromptResult)
    assert responses.request["text_format"] is CompiledPromptResult
    assert "text" not in responses.request
