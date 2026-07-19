import asyncio
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

import pytest

from app.prompt_ninja import (
    PromptFileSpec,
    PromptNinja,
    PromptOutputError,
    PromptRenderError,
    PromptRuntimeOptions,
    PromptValidationError,
    SamplingRunHook,
)


PROMPT = """
spec_version = "1.0"

[prompt]
name = "hello"
description = "Greets a person"
used_in = ["tests"]

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

[output]
format = "json"

[output.schema]
type = "object"
required = ["result", "meta"]

[output.schema.properties.result]
type = "string"

[output.schema.properties.meta]
type = "object"
required = ["enabled"]

[output.schema.properties.meta.properties.enabled]
type = "boolean"

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
    assert prompt.spec.output.output_schema is not None
    prepared = prompt.prepare({"name": "Ada"})
    assert prepared.provider == "openai"
    assert prepared.model == "gpt-5.6"
    assert prepared.user == "Hello Ada. Enabled: true."

    report = prompt.run_tests(lambda _: '{"result": "Hello, Ada!", "meta": {"enabled": true}}')
    assert report.passed
    assert report.results[0].actual == {"result": "Hello, Ada!", "meta": {"enabled": True}}


def test_required_and_unknown_variables_are_rejected(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))
    with pytest.raises(PromptRenderError, match="Required variable"):
        prompt.prepare({})
    with pytest.raises(PromptRenderError, match="Undeclared"):
        prompt.prepare({"name": "Ada", "other": "nope"})


def test_schema_and_expected_output_are_enforced(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))
    with pytest.raises(PromptOutputError, match="missing required fields"):
        prompt.validate_output('{"result": "Hello"}')

    failing = prompt.run_tests(lambda _: {"result": "Different", "meta": {"enabled": True}})
    assert not failing.passed
    assert failing.results[0].error == "Output did not contain the expected values."


def test_undeclared_template_variable_is_rejected(tmp_path):
    content = PROMPT.replace("{{enabled}}", "{{missing}}")
    with pytest.raises(PromptValidationError, match=r"no \[\[variables\]\] definition"):
        PromptNinja.from_file(write_prompt(tmp_path, content))


def test_unknown_toml_fields_are_rejected(tmp_path):
    content = PROMPT.replace('name = "gpt-5.6"', 'name = "gpt-5.6"\nunsupported = true')
    with pytest.raises(PromptValidationError, match="Extra inputs are not permitted"):
        PromptNinja.from_file(write_prompt(tmp_path, content))


def test_included_prompt_file_has_a_passing_test():
    path = Path(__file__).resolve().parents[1] / "prompts" / "greeting.prompt.toml"
    prompt = PromptNinja.from_file(path)

    async def executor(_):
        return {"result": "Hello, Ada!"}

    async def judge(test, actual):
        assert test.expected_output
        assert actual["result"] == "Hello, Ada!"
        return {"score": 0.96, "rationale": "The response satisfies the greeting contract."}

    report = asyncio.run(prompt.arun_tests(executor, judge=judge))
    assert report.passed
    assert report.results[0].score == 0.96


def test_natural_language_test_uses_configured_pass_threshold(tmp_path):
    content = PROMPT.replace('[output]\nformat = "json"', '[output]\nformat = "json"\n\n[testing]\npass_threshold = 0.97')
    content = content.replace('name = "Ada greeting"\n\n[tests.input]', 'name = "Ada greeting"\nexpected_output = "A correct greeting for Ada."\n\n[tests.input]')
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

    assert asyncio.run(prompt.arun({"name": "Ada"}, executor))["result"] == "Hello, Ada!"

    class FakeCompletions:
        async def create(self, **request):
            self.request = request
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"result":"Hello, Ada!","meta":{"enabled":true}}'))])

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    assert asyncio.run(prompt.run_openai({"name": "Ada"}, client=client))["meta"] == {"enabled": True}
    assert completions.request["model"] == "gpt-5.6"
    assert completions.request["response_format"] == {"type": "json_object"}


def test_runtime_overrides_and_sampling_hooks_capture_a_complete_run(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))

    class FakeCompletions:
        async def create(self, **request):
            self.request = request
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"result":"Hello, Ada!","meta":{"enabled":true}}'))])

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    events = []
    hook = SamplingRunHook(events.append, sample_rate=1.0)

    asyncio.run(prompt.run_openai(
        {"name": "Ada"},
        client=client,
        runtime=PromptRuntimeOptions(model="gpt-5.6-sol", temperature=0.3, max_output_tokens=42),
        hooks=(hook,),
    ))

    assert completions.request["model"] == "gpt-5.6-sol"
    assert completions.request["temperature"] == 0.3
    assert completions.request["max_tokens"] == 42
    assert [event.type for event in events] == ["request", "response"]
    assert events[0].user == "Hello Ada. Enabled: true."
    assert events[1].output["result"] == "Hello, Ada!"


def test_openai_client_can_return_a_typed_output_model(tmp_path):
    prompt = PromptNinja.from_file(write_prompt(tmp_path))

    class GreetingOutput(BaseModel):
        result: str
        meta: dict[str, bool]

    class FakeCompletions:
        async def create(self, **_):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"result":"Hello, Ada!","meta":{"enabled":true}}'))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    result = asyncio.run(prompt.run_openai({"name": "Ada"}, client=client, output_model=GreetingOutput))
    assert isinstance(result, GreetingOutput)
    assert result.meta == {"enabled": True}
