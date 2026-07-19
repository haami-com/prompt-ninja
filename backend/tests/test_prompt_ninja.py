import asyncio
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, ValidationError

import pytest

from app.prompt_ninja import (
    BigIntOutput,
    JsonObjectOutput,
    PromptFileSpec,
    PromptNinja,
    PromptNinjaError,
    PromptRenderError,
    PromptRuntimeOptions,
    PromptValidationError,
    SamplingRunHook,
)

PROMPT = """
spec_version = "1.0"
output = "app.prompt_ninja.JsonObjectOutput"

[prompt]
name = "hello"
description = "Greets a person"
used_in = ["backend/tests/test_prompt_ninja.py"]

[model]
provider = "openai"
name = "gpt-5.6"

[template]
system = "Return JSON only."
user = "Hello {{name}}. Enabled: {{enabled}}."

[[variables]]
name = "name"
type = "string"
required = true

[[variables]]
name = "enabled"
type = "boolean"
required = false
default = true

[[tests]]
name = "Ada greeting"

[tests.input]
name = "Ada"

[tests.expected]
result = "Hello, Ada!"
"""


def write_prompt(tmp_path, content=PROMPT):
    path = tmp_path / "hello.prompt.toml"
    path.write_text(content)
    return path


def test_load_render_and_run_embedded_tests(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))
    assert isinstance(prompt.spec, PromptFileSpec)
    assert prompt.output_model is JsonObjectOutput
    prepared = prompt.prepare({"name": "Ada"})
    assert prepared.provider == "openai"
    assert prepared.model == "gpt-5.6"
    assert prepared.user == "Hello Ada. Enabled: true."

    report = prompt.run_tests(
        lambda _: '{"result": "Hello, Ada!", "meta": {"enabled": true}}'
    )
    assert report.passed
    assert report.results[0].actual.root == {
        "result": "Hello, Ada!",
        "meta": {"enabled": True},
    }


def test_required_and_unknown_variables_are_rejected(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))
    with pytest.raises(PromptRenderError, match="Required variable"):
        prompt.prepare({})
    with pytest.raises(PromptRenderError, match="Undeclared"):
        prompt.prepare({"name": "Ada", "other": "nope"})


def test_system_overrides_render_the_prompt_variables(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))

    prepared = prompt.prepare(
        {"name": "Ada"},
        system_override="Write a friendly greeting for {{name}}.",
    )

    assert prepared.system == "Write a friendly greeting for Ada."


def test_pydantic_output_and_expected_output_are_enforced(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))
    definition = prompt.spec.model_dump(by_alias=True, exclude_none=True)
    definition["output"] = "app.models.GreetingResult"
    definition["tests"] = []

    with pytest.raises(ValidationError, match="result"):
        PromptNinja(definition).validate_output("{}")

    failing = prompt.run_tests(
        lambda _: {"result": "Different", "meta": {"enabled": True}}
    )
    assert not failing.passed
    assert failing.results[0].error == "Output did not contain the expected values."


def test_undeclared_template_variable_is_rejected(tmp_path):
    content = PROMPT.replace("{{enabled}}", "{{missing}}")
    with pytest.raises(PromptValidationError, match=r"no \[\[variables\]\] definition"):
        PromptNinja.from_file(write_prompt(tmp_path, content))


def test_required_variable_missing_from_template_is_rejected(tmp_path):
    content = PROMPT.replace(
        "Hello {{name}}. Enabled: {{enabled}}.", "Enabled: {{enabled}}."
    )
    with pytest.raises(
        PromptValidationError, match="required variables are not referenced"
    ):
        PromptNinja.from_file(write_prompt(tmp_path, content))


def test_unknown_toml_fields_are_rejected(tmp_path):
    content = PROMPT.replace('name = "gpt-5.6"', 'name = "gpt-5.6"\nunsupported = true')
    with pytest.raises(PromptValidationError, match="Extra inputs are not permitted"):
        PromptNinja.from_file(write_prompt(tmp_path, content))


def test_used_in_requires_repository_relative_file_paths(tmp_path):
    content = PROMPT.replace(
        'used_in = ["backend/tests/test_prompt_ninja.py"]',
        'used_in = ["prompt-ninja"]',
    )

    with pytest.raises(PromptValidationError, match="repository-relative file paths"):
        PromptNinja.from_file(write_prompt(tmp_path, content))


def test_validation_resolves_declared_output_model_paths(tmp_path):
    definition = PromptNinja.from_file(write_prompt(tmp_path)).spec.model_dump(
        by_alias=True,
        exclude_none=True,
    )
    definition["output"] = "app.models.GreetingResult"
    definition["tests"] = []

    prompt = PromptNinja(definition)

    assert prompt.output_model.__name__ == "GreetingResult"


def test_validation_rejects_missing_or_non_pydantic_output_models(tmp_path):
    definition = PromptNinja.from_file(write_prompt(tmp_path)).spec.model_dump(
        by_alias=True,
        exclude_none=True,
    )
    definition["tests"] = []
    definition["output"] = "app.models.OutputModelThatDoesNotExist"

    with pytest.raises(PromptValidationError, match="could not be imported"):
        PromptNinja(definition)

    definition["output"] = "app.models.DEFAULT_MODEL"
    with pytest.raises(
        PromptValidationError, match="must resolve to a Pydantic BaseModel"
    ):
        PromptNinja(definition)


def test_validate_rechecks_a_mutated_output_model_path(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))
    prompt.definition["output"] = "app.models.OutputModelThatDoesNotExist"

    with pytest.raises(PromptValidationError, match="could not be imported"):
        prompt.validate()


def test_included_prompt_file_has_a_passing_test():
    path = Path(__file__).resolve().parents[1] / "prompts" / "greeting.prompt.toml"
    prompt = PromptNinja.from_file(path)

    async def executor(_):
        return {"result": "Hello, Ada!"}

    async def judge(test, actual):
        assert test.expected_output
        assert actual.result == "Hello, Ada!"
        return {
            "score": 0.96,
            "rationale": "The response satisfies the greeting contract.",
        }

    report = asyncio.run(prompt.arun_tests(executor, judge=judge))
    assert report.passed
    assert report.results[0].score == 0.96


def test_validated_prompt_round_trips_through_toml_with_its_full_spec(tmp_path):
    source = Path(__file__).resolve().parents[1] / "prompts" / "greeting.prompt.toml"
    original = PromptNinja.from_file(source)
    exported_path = tmp_path / "round-trip.prompt.toml"
    exported_path.write_text(original.to_toml())
    restored = PromptNinja.from_file(exported_path)

    assert restored.spec.model_dump(by_alias=True) == original.spec.model_dump(
        by_alias=True
    )


def test_toml_round_trip_preserves_every_variable_type_and_embedded_test(tmp_path):
    definition = {
        "spec_version": "1.0",
        "prompt": {
            "name": "typed",
            "description": "Exercises every variable type.",
            "used_in": ["backend/tests/test_prompt_ninja.py"],
        },
        "model": {"provider": "openai", "name": "gpt-5.6-sol"},
        "template": {
            "system": "Process typed input.",
            "user": "{{text}} {{count}} {{ratio}} {{enabled}} {{tags}} {{metadata}}",
        },
        "variables": [
            {"name": "text", "type": "string", "required": True},
            {"name": "count", "type": "integer", "required": True},
            {"name": "ratio", "type": "number", "required": True},
            {"name": "enabled", "type": "boolean", "required": True},
            {"name": "tags", "type": "array", "required": True},
            {"name": "metadata", "type": "object", "required": True},
        ],
        "output": "app.prompt_ninja.JsonObjectOutput",
        "testing": {"pass_threshold": 0.9},
        "tests": [
            {
                "name": "typed fixture",
                "input": {
                    "text": "hello",
                    "count": 2,
                    "ratio": 0.5,
                    "enabled": True,
                    "tags": ["one"],
                    "metadata": {"source": "test"},
                },
                "expected": {"items": ["hello"]},
            }
        ],
    }
    original = PromptNinja(definition)
    path = tmp_path / "typed-round-trip.prompt.toml"
    path.write_text(original.to_toml())
    restored = PromptNinja.from_file(path)

    assert restored.spec.model_dump(by_alias=True) == original.spec.model_dump(
        by_alias=True
    )


def test_natural_language_test_uses_configured_pass_threshold(tmp_path):
    content = PROMPT.replace(
        "[[tests]]",
        "[testing]\npass_threshold = 0.97\n\n[[tests]]",
    )
    content = content.replace(
        'name = "Ada greeting"\n\n[tests.input]',
        'name = "Ada greeting"\nexpected_output = "A correct greeting for Ada."\n\n[tests.input]',
    )
    content = content.replace('\n[tests.expected]\nresult = "Hello, Ada!"', "")
    prompt = PromptNinja.from_file(write_prompt(tmp_path, content))

    async def executor(_):
        return {"result": "Hello, Ada!", "meta": {"enabled": True}}

    async def judge(_, __):
        return {"score": 0.96, "rationale": "Almost correct."}

    report = asyncio.run(prompt.arun_tests(executor, judge=judge))
    assert not report.passed
    assert report.results[0].score == 0.96


def test_async_execution_and_openai_adapter_use_the_prompt_contract(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))

    async def executor(_):
        return {"result": "Hello, Ada!", "meta": {"enabled": True}}

    assert (
        asyncio.run(prompt.arun({"name": "Ada"}, executor)).root["result"]
        == "Hello, Ada!"
    )

    class FakeResponses:
        async def create(self, **request):
            self.request = request
            return SimpleNamespace(
                output_text='{"result":"Hello, Ada!","meta":{"enabled":true}}'
            )

    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    assert asyncio.run(prompt.run_openai({"name": "Ada"}, client=client)).root[
        "meta"
    ] == {"enabled": True}
    assert responses.request["model"] == "gpt-5.6"
    assert (
        responses.request["instructions"]
        == "Return JSON only.\n\nReturn a valid JSON object."
    )
    assert (
        responses.request["input"]
        == "Hello Ada. Enabled: true.\n\nReturn a valid JSON object."
    )
    assert "text" not in responses.request
    assert responses.request["store"] is False


def test_json_output_adds_an_explicit_json_instruction_when_template_omits_one(
    tmp_path,
):
    prompt = PromptNinja.from_file(
        write_prompt(
            tmp_path,
            PROMPT.replace(
                'system = "Return JSON only."', 'system = "Follow the output schema."'
            ),
        )
    )

    class FakeResponses:
        async def create(self, **request):
            self.request = request
            return SimpleNamespace(
                output_text='{"result":"Hello, Ada!","meta":{"enabled":true}}'
            )

    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    asyncio.run(prompt.run_openai({"name": "Ada"}, client=client))

    assert (
        responses.request["instructions"]
        == "Follow the output schema.\n\nReturn a valid JSON object."
    )
    assert (
        responses.request["input"]
        == "Hello Ada. Enabled: true.\n\nReturn a valid JSON object."
    )


def test_installed_openai_client_exposes_the_responses_api():
    """Integration check for the SDK surface used by OpenAIPromptClient.

    Constructing this client with a placeholder key does not issue a network
    request, but catches an environment that has not been synced after an SDK
    upgrade.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="test-key")
    assert client.responses is not None
    assert callable(client.responses.create)


def test_openai_adapter_explains_when_a_client_lacks_responses_api(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))
    client = SimpleNamespace()

    with pytest.raises(
        PromptNinjaError,
        match="does not support the Responses API.*uv sync.*restart the service",
    ):
        asyncio.run(prompt.run_openai({"name": "Ada"}, client=client))


def test_runtime_overrides_and_sampling_hooks_capture_a_complete_run(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))

    class FakeResponses:
        async def create(self, **request):
            self.request = request
            return SimpleNamespace(
                output_text='{"result":"Hello, Ada!","meta":{"enabled":true}}'
            )

    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    events = []
    hook = SamplingRunHook(events.append, sample_rate=1.0)

    asyncio.run(
        prompt.run_openai(
            {"name": "Ada"},
            client=client,
            runtime=PromptRuntimeOptions(model="gpt-5.6-sol"),
            hooks=(hook,),
        )
    )

    assert responses.request["model"] == "gpt-5.6-sol"
    assert "temperature" not in responses.request
    assert "max_output_tokens" not in responses.request
    assert [event.type for event in events] == ["request", "response"]
    assert events[0].user == "Hello Ada. Enabled: true.\n\nReturn a valid JSON object."
    assert events[1].output.root["result"] == "Hello, Ada!"


def test_openai_client_can_return_a_typed_output_model(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))

    class GreetingOutput(BaseModel):
        result: str
        meta: dict[str, bool]

    class FakeResponses:
        async def create(self, **_):
            return SimpleNamespace(
                output_text='{"result":"Hello, Ada!","meta":{"enabled":true}}'
            )

    client = SimpleNamespace(responses=FakeResponses())
    result = asyncio.run(
        prompt.run_openai({"name": "Ada"}, client=client, output_model=GreetingOutput)
    )
    assert isinstance(result, GreetingOutput)
    assert result.meta == {"enabled": True}


def test_openai_client_prefers_responses_parse_for_a_typed_output_model(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))

    class GreetingOutput(BaseModel):
        result: str
        meta: dict[str, bool]

    class FakeResponses:
        async def parse(self, **request):
            self.request = request
            return SimpleNamespace(
                output_parsed=GreetingOutput(
                    result="Hello, Ada!",
                    meta={"enabled": True},
                )
            )

        async def create(self, **_request):
            raise AssertionError(
                "create should not be used for typed structured output"
            )

    def redundant_validation(_):
        raise AssertionError("parsed Pydantic output should not be validated twice")

    prompt.validate_output = redundant_validation
    responses = FakeResponses()
    result = asyncio.run(
        prompt.run_openai(
            {"name": "Ada"},
            client=SimpleNamespace(responses=responses),
            output_model=GreetingOutput,
        )
    )

    assert isinstance(result, GreetingOutput)
    assert result.result == "Hello, Ada!"
    assert responses.request["text_format"] is GreetingOutput
    assert "text" not in responses.request


def test_bigint_output_uses_a_typed_root_model_with_responses_parse(tmp_path):
    definition = PromptNinja.from_file(write_prompt(tmp_path)).spec.model_dump(
        by_alias=True,
        exclude_none=True,
    )
    definition["output"] = "BigInt"
    definition["tests"] = []
    prompt = PromptNinja(definition)

    class FakeResponses:
        async def parse(self, **request):
            self.request = request
            return SimpleNamespace(output_parsed=BigIntOutput(42))

    responses = FakeResponses()
    result = asyncio.run(
        prompt.run_openai(
            {"name": "Ada"},
            client=SimpleNamespace(responses=responses),
        )
    )

    assert result == 42
    assert responses.request["text_format"] is BigIntOutput
