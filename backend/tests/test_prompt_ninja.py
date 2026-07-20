import asyncio
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, ValidationError

import pytest

from app.prompt_ninja import (
    BigIntOutput,
    JsonObjectOutput,
    PromptCollection,
    PromptFileSpec,
    PromptNinja,
    PromptNinjaError,
    PromptRenderError,
    PromptRuntimeOptions,
    PromptValidationError,
    SamplingRunHook,
)
from app.models import Person

PROMPT = """
[metadata]
spec_version = "1.2"
name = "hello"
description = "Greets a person"
used_by = ["backend/tests/test_prompt_ninja.py"]
version = "1.0.0"
output = "app.prompt_ninja.JsonObjectOutput"

[llm_model]
provider = "openrouter"
name = "google/gemini-2.5-flash"

[prompt]
system = "Provide a friendly greeting."
user = "Hello {{name}}. Enabled: {{enabled}}."

[[variables]]
name = "name"
type = "string"
description = "The person to greet."
required = true

[[variables]]
name = "enabled"
type = "boolean"
description = "Whether the greeting feature is enabled."
required = false
default = true

[[tests]]
name = "Ada greeting"
variable.name = "Ada"
expected_output = "A friendly greeting to Ada in a JSON object with a result string."
"""


def write_prompt(tmp_path, content=PROMPT):
    path = tmp_path / "hello.prompt.toml"
    path.write_text(content)
    return path


def prompt_with_name(name: str) -> str:
    return PROMPT.replace('name = "hello"', f'name = "{name}"', 1)


def test_prompt_collection_eagerly_loads_prompts_with_mapping_and_dot_access(tmp_path):
    write_prompt(tmp_path)
    (tmp_path / "brief-enhancer.prompt.toml").write_text(
        prompt_with_name("brief-enhancer")
    )

    prompts = PromptCollection(dir=tmp_path)

    assert prompts.names == ("brief-enhancer", "hello")
    assert prompts["brief-enhancer"] is prompts.brief_enhancer
    assert prompts["hello"] is prompts.hello
    assert isinstance(prompts.hello, PromptNinja)

    (tmp_path / "later.prompt.toml").write_text(prompt_with_name("later"))
    assert "later" not in prompts
    assert len(prompts) == 2


def test_prompt_collection_rejects_ambiguous_dot_access_aliases(tmp_path):
    write_prompt(tmp_path, prompt_with_name("brief-enhancer"))
    (tmp_path / "brief_enhancer.prompt.toml").write_text(
        prompt_with_name("brief_enhancer")
    )

    with pytest.raises(PromptValidationError, match="share dot-access alias"):
        PromptCollection(dir=tmp_path)


def test_prompt_collection_rejects_names_reserved_by_its_dot_access_api(tmp_path):
    write_prompt(tmp_path, prompt_with_name("items"))

    with pytest.raises(PromptValidationError, match="reserved by PromptCollection"):
        PromptCollection(dir=tmp_path)


def test_load_render_and_run_embedded_tests(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))
    assert isinstance(prompt.spec, PromptFileSpec)
    assert prompt.output_model is JsonObjectOutput
    prepared = prompt.prepare({"name": "Ada"})
    assert prepared.provider == "openrouter"
    assert prepared.model == "google/gemini-2.5-flash"
    assert prepared.user == "Hello Ada. Enabled: True."

    async def judge(_, __):
        return {"score": 1.0, "rationale": "Matches the greeting contract."}

    async def executor(_):
        return {"result": "Hello, Ada!", "meta": {"enabled": True}}

    report = asyncio.run(prompt.arun_tests(executor, judge=judge))
    assert report.passed


def test_async_tests_report_each_completed_case_to_a_progress_callback(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))
    started = []
    completed = []

    async def executor(_):
        return {"result": "Hello, Ada!"}

    async def judge(_, __):
        return {"score": 1.0, "rationale": "Matches the greeting contract."}

    report = asyncio.run(
        prompt.arun_tests(
            executor,
            judge=judge,
            on_start=started.append,
            on_result=completed.append,
        )
    )

    assert report.passed
    assert [test.name for test in started] == ["Ada greeting"]
    assert [result.name for result in completed] == ["Ada greeting"]


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

    assert prepared.system.startswith("Write a friendly greeting for Ada.\n\n")
    assert "metadata.output = 'app.prompt_ninja.JsonObjectOutput'" in prepared.system


def test_pydantic_output_and_expected_output_are_enforced(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))
    definition = prompt.spec.model_dump(by_alias=True, exclude_none=True)
    definition["metadata"]["output"] = "app.models.GreetingResult"
    definition["tests"] = []

    with pytest.raises(ValidationError, match="result"):
        PromptNinja(definition).validate_output("{}")

    async def executor(_):
        return {"result": "Different", "meta": {"enabled": True}}

    async def judge(_, __):
        return {"score": 0.0, "rationale": "The greeting is wrong."}

    failing = asyncio.run(prompt.arun_tests(executor, judge=judge))
    assert not failing.passed
    assert failing.results[0].score == 0.0


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
    content = PROMPT.replace(
        'name = "google/gemini-2.5-flash"',
        'name = "google/gemini-2.5-flash"\nunsupported = true',
    )
    with pytest.raises(PromptValidationError, match="Extra inputs are not permitted"):
        PromptNinja.from_file(write_prompt(tmp_path, content))


def test_used_by_requires_repository_relative_file_paths(tmp_path):
    content = PROMPT.replace(
        'used_by = ["backend/tests/test_prompt_ninja.py"]',
        'used_by = ["prompt-ninja"]',
    )

    with pytest.raises(PromptValidationError, match="repository-relative file paths"):
        PromptNinja.from_file(write_prompt(tmp_path, content))


def test_validation_resolves_declared_output_model_paths(tmp_path):
    definition = PromptNinja.from_file(write_prompt(tmp_path)).spec.model_dump(
        by_alias=True,
        exclude_none=True,
    )
    definition["metadata"]["output"] = "app.models.GreetingResult"
    definition["tests"] = []

    prompt = PromptNinja(definition)

    assert prompt.output_model.__name__ == "GreetingResult"


def test_validation_rejects_missing_or_non_pydantic_output_models(tmp_path):
    definition = PromptNinja.from_file(write_prompt(tmp_path)).spec.model_dump(
        by_alias=True,
        exclude_none=True,
    )
    definition["tests"] = []
    definition["metadata"]["output"] = "app.models.OutputModelThatDoesNotExist"

    with pytest.raises(PromptValidationError, match="could not be imported"):
        PromptNinja(definition)

    definition["metadata"]["output"] = "app.models.DEFAULT_MODEL"
    with pytest.raises(
        PromptValidationError, match="must resolve to a Pydantic BaseModel"
    ):
        PromptNinja(definition)


def test_validate_rechecks_a_mutated_output_model_path(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))
    prompt.definition["metadata"]["output"] = "app.models.OutputModelThatDoesNotExist"

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


def test_every_checked_in_prompt_has_two_semantic_test_cases():
    prompts_directory = Path(__file__).resolve().parents[1] / "prompts"

    for path in sorted(prompts_directory.glob("*.prompt.toml")):
        prompt = PromptNinja.from_file(path)
        assert len(prompt.tests) == 2, path.name
        assert all(test.expected_output.strip() for test in prompt.tests), path.name


def test_validated_prompt_round_trips_through_toml_with_its_full_spec(tmp_path):
    source = Path(__file__).resolve().parents[1] / "prompts" / "greeting.prompt.toml"
    original = PromptNinja.from_file(source)
    exported_path = tmp_path / "round-trip.prompt.toml"
    serialized = original.to_toml()
    exported_path.write_text(serialized)
    restored = PromptNinja.from_file(exported_path)

    assert "[metadata]" in serialized
    assert "[llm_model]" in serialized
    assert "[prompt]" in serialized
    assert 'variable.name = "Ada"' in serialized
    assert "expected_output =" in serialized
    assert restored.spec.model_dump(by_alias=True) == original.spec.model_dump(
        by_alias=True
    )


def test_toml_round_trip_preserves_every_variable_type_and_embedded_test(tmp_path):
    definition = {
        "metadata": {
            "spec_version": "1.2",
            "name": "typed",
            "description": "Exercises every variable type.",
            "used_by": ["backend/tests/test_prompt_ninja.py"],
            "version": "1.0.0",
            "output": "app.prompt_ninja.JsonObjectOutput",
        },
        "llm_model": {"provider": "openrouter", "name": "google/gemini-2.5-flash"},
        "prompt": {
            "system": "Process typed input.",
            "user": "{{text}} {{count}} {{ratio}} {{enabled}} {{tags}} {{metadata}} {{due_date}}",
        },
        "variables": [
            {
                "name": "text",
                "type": "string",
                "description": "Text to process.",
                "required": True,
            },
            {
                "name": "count",
                "type": "integer",
                "description": "Item count.",
                "required": True,
            },
            {
                "name": "ratio",
                "type": "number",
                "description": "A numeric ratio.",
                "required": True,
            },
            {
                "name": "enabled",
                "type": "boolean",
                "description": "Whether processing is enabled.",
                "required": True,
            },
            {
                "name": "tags",
                "type": "array",
                "description": "Associated tags.",
                "required": True,
            },
            {
                "name": "metadata",
                "type": "object",
                "description": "Associated metadata.",
                "required": True,
            },
            {
                "name": "due_date",
                "type": "date",
                "description": "The date associated with this input.",
                "required": True,
            },
        ],
        "testing": {"pass_threshold": 0.9},
        "tests": [
            {
                "name": "typed fixture",
                "variable": {
                    "text": "hello",
                    "count": 2,
                    "ratio": 0.5,
                    "enabled": True,
                    "tags": ["one"],
                    "metadata": {"source": "test"},
                    "due_date": date(2026, 7, 20),
                },
                "expected_output": "An object containing hello in its items list.",
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
    assert (
        "2026-07-20"
        in original.prepare(
            {
                "text": "hello",
                "count": 2,
                "ratio": 0.5,
                "enabled": True,
                "tags": ["one"],
                "metadata": {"source": "test"},
                "due_date": date(2026, 7, 20),
            }
        ).user
    )


def test_typed_variables_coerce_render_and_construct_pydantic_models(tmp_path):
    content = """
[metadata]
spec_version = "1.2"
name = "typed-rendering"
description = "Renders declared Python-like input types."
used_by = ["backend/tests/test_prompt_ninja.py"]
version = "1.0.0"
output = "String"

[llm_model]
provider = "openrouter"
name = "google/gemini-2.5-flash"

[prompt]
user = "{{payload}} | {{tags}} | {{when}} | {{person}} | {{person | json}} | {{tags | repr}}"

[[variables]]
name = "payload"
type = "json"
description = "Arbitrary JSON data."
required = true

[[variables]]
name = "tags"
type = "list[str]"
description = "Labels to include."
required = true

[[variables]]
name = "when"
type = "datetime"
description = "The timestamp for this operation."
required = true

[[variables]]
name = "person"
type = "app.models.Person"
description = "The person receiving the response."
required = true

[[tests]]
name = "constructs a person from TOML data"
variable.payload = { source = "test" }
variable.tags = ["Ada", "Lin"]
variable.when = "2026-07-20T09:30:00"
variable.person = { name = "Ada", role = "Engineer" }
expected_output = "Uses the supplied typed values correctly."
"""
    prompt = PromptNinja.from_file(write_prompt(tmp_path, content))

    assert isinstance(prompt.spec.tests[0].variable["person"], Person)
    prepared = prompt.prepare(
        {
            "payload": {"source": "runtime"},
            "tags": ["Ada", "Lin"],
            "when": "2026-07-20T09:30:00",
            "person": {"name": "Ada", "role": "Engineer"},
        }
    )

    assert prepared.user == (
        '{"source": "runtime"} | Ada, Lin | 2026-07-20 09:30:00 | '
        "Person(name='Ada', role='Engineer') | "
        '{"name": "Ada", "role": "Engineer"} | [\'Ada\', \'Lin\']'
    )


def test_typed_variable_errors_explain_missing_and_received_values(tmp_path):
    content = PROMPT.replace('type = "string"', 'type = "list[int]"', 1)
    content = content.replace('variable.name = "Ada"', "variable.name = [1]")
    prompt = PromptNinja.from_file(write_prompt(tmp_path, content))

    with pytest.raises(
        PromptRenderError, match=r"name.*type list\[integer\].*not provided"
    ):
        prompt.prepare({})
    with pytest.raises(
        PromptRenderError, match=r"name.*expects list\[integer\]; received list"
    ):
        prompt.prepare({"name": ["not-an-integer"]})


def test_datetime_and_dynamic_types_are_supported(tmp_path):
    definition = PromptNinja.from_file(write_prompt(tmp_path)).spec.model_dump(
        by_alias=True, exclude_none=True
    )
    definition["prompt"]["user"] = "{{occurred_at}} {{anything}}"
    definition["variables"] = [
        {
            "name": "occurred_at",
            "type": "datetime",
            "description": "When the event occurred.",
            "required": True,
        },
        {
            "name": "anything",
            "type": "dynamic",
            "description": "An intentionally unconstrained value.",
            "required": True,
        },
    ]
    definition["tests"] = []
    prompt = PromptNinja(definition)

    assert (
        prompt.prepare(
            {"occurred_at": datetime(2026, 7, 20, 9, 30), "anything": {"ok": True}}
        ).user
        == "2026-07-20 09:30:00 {'ok': True}"
    )


def test_natural_language_test_uses_configured_pass_threshold(tmp_path):
    content = PROMPT.replace(
        "[[tests]]",
        "[testing]\npass_threshold = 0.97\n\n[[tests]]",
    )
    content = content.replace(
        'expected_output = "A friendly greeting to Ada in a JSON object with a result string."',
        'expected_output = "A correct greeting for Ada."',
    )
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
    assert asyncio.run(prompt.run_openrouter({"name": "Ada"}, client=client)).root[
        "meta"
    ] == {"enabled": True}
    assert responses.request["model"] == "google/gemini-2.5-flash"
    assert responses.request["instructions"].startswith(
        "Provide a friendly greeting.\n\n"
    )
    assert (
        "metadata.output = 'app.prompt_ninja.JsonObjectOutput'"
        in responses.request["instructions"]
    )
    assert responses.request["input"] == "Hello Ada. Enabled: True."
    assert "text" not in responses.request
    assert responses.request["store"] is False


def test_output_contract_is_injected_when_template_omits_output_instructions(
    tmp_path,
):
    prompt = PromptNinja.from_file(
        write_prompt(
            tmp_path,
            PROMPT.replace(
                'system = "Provide a friendly greeting."',
                'system = "Follow the task instructions."',
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
    asyncio.run(prompt.run_openrouter({"name": "Ada"}, client=client))

    assert responses.request["instructions"].startswith(
        "Follow the task instructions.\n\nOutput contract"
    )
    assert '"type": "object"' in responses.request["instructions"]
    assert responses.request["input"] == "Hello Ada. Enabled: True."


def test_installed_openai_compatible_client_exposes_the_responses_api():
    """Integration check for the SDK surface used by OpenRouterPromptClient.

    Constructing this client with a placeholder key does not issue a network
    request, but catches an environment that has not been synced after an SDK
    upgrade.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="test-key")
    assert client.responses is not None
    assert callable(client.responses.create)


def test_openrouter_adapter_explains_when_a_client_lacks_responses_api(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))
    client = SimpleNamespace()

    with pytest.raises(
        PromptNinjaError,
        match="does not support the Responses API.*uv sync.*restart the service",
    ):
        asyncio.run(prompt.run_openrouter({"name": "Ada"}, client=client))


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
        prompt.run_openrouter(
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
    assert events[0].user == "Hello Ada. Enabled: True."
    assert events[1].output.root["result"] == "Hello, Ada!"


def test_openrouter_client_can_return_a_typed_output_model(tmp_path):
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
        prompt.run_openrouter(
            {"name": "Ada"}, client=client, output_model=GreetingOutput
        )
    )
    assert isinstance(result, GreetingOutput)
    assert result.meta == {"enabled": True}


def test_openrouter_client_prefers_responses_parse_for_a_typed_output_model(tmp_path):
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
        prompt.run_openrouter(
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
    definition["metadata"]["output"] = "BigInt"
    definition["tests"] = []
    prompt = PromptNinja(definition)

    class FakeResponses:
        async def parse(self, **request):
            self.request = request
            return SimpleNamespace(output_parsed=BigIntOutput(42))

    responses = FakeResponses()
    result = asyncio.run(
        prompt.run_openrouter(
            {"name": "Ada"},
            client=SimpleNamespace(responses=responses),
        )
    )

    assert result == 42
    assert responses.request["text_format"] is BigIntOutput
