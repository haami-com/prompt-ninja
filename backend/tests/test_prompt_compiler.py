import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from prompt_ninja.prompt_catalog import PROMPTS
from prompt_ninja.prompt_compiler import (
    CompiledOutputModel,
    CompiledPromptDefinition,
    CompiledPromptResult,
    build_compiled_output_model,
)


def canonical_definition():
    return {
        "metadata": {
            "spec_version": "1.2",
            "name": "code-review",
            "description": "Reviews a user specification.",
            "used_by": ["src/prompt_consumer.py"],
            "version": "1.0.0",
            "output": "String",
        },
        "llm_model": {"provider": "openrouter", "name": "google/gemini-2.5-flash"},
        "prompt": {
            "system": "Review the specification and return advice.",
            "user": "{{USER_SPECIFICATION}}",
        },
        "variables": [
            {
                "name": "USER_SPECIFICATION",
                "type": "string",
                "required": True,
                "description": "The specification to review.",
            }
        ],
        "testing": {"pass_threshold": 0.95},
        "tests": [
            {
                "name": "summarize_complex_spec",
                "variable": {"USER_SPECIFICATION": "Do not mutate the input."},
                "expected_output": "A review that preserves the constraint.",
            }
        ],
    }


def test_compiler_structured_output_is_a_nested_pydantic_model():
    result = CompiledPromptResult.model_validate(
        {
            "definition": canonical_definition(),
            "output_model": None,
        }
    )

    assert isinstance(result.definition, CompiledPromptDefinition)
    assert result.definition.variables[0].name == "USER_SPECIFICATION"
    assert result.definition.output == "String"


def test_compiler_schema_omits_free_form_test_variable_objects():
    schema = CompiledPromptResult.model_json_schema()

    assert "tests" not in schema["$defs"]["CompiledPromptDefinition"]["properties"]

    def assert_strict_objects(value):
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value.get("additionalProperties") is False
            for child in value.values():
                assert_strict_objects(child)
        elif isinstance(value, list):
            for child in value:
                assert_strict_objects(child)

    assert_strict_objects(schema)


def test_compiler_does_not_normalize_an_incompatible_schema_dialect():
    malformed = canonical_definition()
    malformed["variables"] = {
        "USER_SPECIFICATION": {"type": "string"},
    }
    malformed["metadata"]["output"] = {
        "type": "string",
        "description": "A concise review.",
    }

    with pytest.raises(ValidationError):
        CompiledPromptResult.model_validate({"definition": malformed})


def test_compiler_discards_only_an_invalid_model_generated_default():
    definition = canonical_definition()
    definition["variables"][0]["default"] = False

    result = CompiledPromptResult.model_validate(
        {"definition": definition, "output_model": None}
    )

    assert not result.definition.variables[0].has_default


def test_prompt_toml_resolves_its_declared_pydantic_output_model():
    compiler_prompt = PROMPTS.prompt_compiler

    assert compiler_prompt.output_model is CompiledPromptResult
    assert compiler_prompt.output_format == "json"


def test_compiler_passes_its_toml_declared_model_to_responses_parse():
    compiler_prompt = PROMPTS.prompt_compiler

    class FakeResponses:
        async def parse(self, **request):
            self.request = request
            return SimpleNamespace(
                output_parsed=CompiledPromptResult(
                    definition=canonical_definition(),
                    output_model=None,
                )
            )

    responses = FakeResponses()
    result = asyncio.run(
        compiler_prompt.run_openrouter(
            {
                "goal": "Review a specification",
                "model": "gpt-5.6-sol",
                "requirements": {},
                "candidate_prompt": "Review the specification.",
                "test_result": {},
            },
            client=SimpleNamespace(responses=responses),
        )
    )

    assert isinstance(result, CompiledPromptResult)
    assert responses.request["text_format"] is CompiledPromptResult
    assert "text" not in responses.request


def test_compiler_accepts_a_concrete_generated_output_model():
    definition = canonical_definition()
    definition["metadata"]["used_by"] = []
    definition["metadata"]["output"] = "prompt_ninja.JsonObjectOutput"

    result = CompiledPromptResult.model_validate(
        {
            "definition": definition,
            "output_model": {
                "class_name": "TicketTriageOutput",
                "fields": [
                    {
                        "name": "category",
                        "type": "string",
                        "description": "Assigned support category.",
                    },
                    {
                        "name": "needs_escalation",
                        "type": "boolean",
                        "description": "Whether the ticket needs escalation.",
                    },
                ],
            },
        }
    )

    assert result.definition.metadata.used_by == []
    assert result.output_model is not None
    assert result.output_model.class_name == "TicketTriageOutput"
    assert result.output_model.fields[1].type == "boolean"


def test_compiler_defaults_unknown_consumers_to_empty():
    definition = canonical_definition()
    definition["metadata"].pop("used_by")

    result = CompiledPromptResult.model_validate(
        {"definition": definition, "output_model": None}
    )

    assert result.definition.metadata.used_by == []


def test_compiler_rejects_output_fields_that_cannot_be_generated_as_python():
    with pytest.raises(ValidationError, match="Python keywords"):
        CompiledOutputModel.model_validate(
            {
                "class_name": "InvalidOutput",
                "fields": [
                    {
                        "name": "class",
                        "type": "string",
                        "description": "An invalid Python field name.",
                    }
                ],
            }
        )


def test_compiled_output_model_rejects_incompatible_self_test_json():
    model = build_compiled_output_model(
        CompiledOutputModel.model_validate(
            {
                "class_name": "TicketTriageOutput",
                "fields": [
                    {
                        "name": "category",
                        "type": "string",
                        "description": "Assigned support category.",
                    },
                    {
                        "name": "needs_escalation",
                        "type": "boolean",
                        "description": "Whether escalation is needed.",
                    },
                ],
            }
        )
    )

    with pytest.raises(ValidationError, match="needs_escalation"):
        model.model_validate_json('{"category":"billing"}')
