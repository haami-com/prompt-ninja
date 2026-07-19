"""Load, validate, render, and test versioned ``*.prompt.toml`` files.

TOML is parsed into a typed Pydantic model before it is used. This keeps the
file format's structural rules close to the format itself, while
``PromptNinja`` owns only runtime work: rendering, executing, and checking a
model response.
"""

from __future__ import annotations

import inspect
import json
import random
import re
import tomllib
import uuid
from pathlib import Path
from collections.abc import Awaitable
from typing import Any, Callable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


SUPPORTED_SPEC_VERSION = "1.0"
VariableType = Literal["string", "integer", "number", "boolean", "array", "object"]
OutputFormat = Literal["text", "json"]
_VARIABLE_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class PromptNinjaError(ValueError):
    """Base error for an invalid prompt or prompt operation."""


class PromptValidationError(PromptNinjaError):
    """Raised when a prompt file does not satisfy the prompt-file specification."""


class PromptRenderError(PromptNinjaError):
    """Raised when variables cannot be rendered into a prompt template."""


class PromptOutputError(PromptNinjaError):
    """Raised when a model response does not match the declared output contract."""


class SpecModel(BaseModel):
    """Base model that rejects misspelled or unsupported TOML fields."""

    model_config = ConfigDict(extra="forbid")


class PromptMetadata(SpecModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    used_in: list[str] = Field(min_length=1)

    @field_validator("name", "description")
    @classmethod
    def non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("used_in")
    @classmethod
    def non_blank_usage_names(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("must contain only non-blank application names")
        return value


class ModelConfig(SpecModel):
    provider: str = Field(min_length=1)
    name: str = Field(min_length=1)
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)

    @field_validator("provider", "name")
    @classmethod
    def non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("temperature", "max_output_tokens", mode="before")
    @classmethod
    def numbers_cannot_be_booleans(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("must be a number, not a boolean")
        return value


class TemplateSpec(SpecModel):
    system: str = ""
    user: str = ""

    @model_validator(mode="after")
    def has_a_message(self) -> "TemplateSpec":
        if not self.system.strip() and not self.user.strip():
            raise ValueError("must contain a non-empty system or user template")
        return self

    @property
    def referenced_variables(self) -> set[str]:
        return set(_VARIABLE_PATTERN.findall(self.system)) | set(_VARIABLE_PATTERN.findall(self.user))


class VariableSpec(SpecModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    type: VariableType
    required: bool
    description: str | None = None
    default: Any = None

    @property
    def has_default(self) -> bool:
        return "default" in self.model_fields_set

    def accepts(self, value: Any) -> bool:
        checks = {
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "array": isinstance(value, list),
            "object": isinstance(value, dict),
        }
        return checks[self.type]

    @model_validator(mode="after")
    def default_matches_type(self) -> "VariableSpec":
        if self.has_default and not self.accepts(self.default):
            raise ValueError("default must have type %s" % self.type)
        return self


class OutputSchema(SpecModel):
    """The supported, intentionally small JSON Schema subset."""

    type: VariableType
    required: list[str] = Field(default_factory=list)
    properties: dict[str, "OutputSchema"] = Field(default_factory=dict)
    items: "OutputSchema | None" = None

    @model_validator(mode="after")
    def structure_matches_type(self) -> "OutputSchema":
        if self.required and self.type != "object":
            raise ValueError("required is only valid for object schemas")
        if self.properties and self.type != "object":
            raise ValueError("properties is only valid for object schemas")
        if self.items is not None and self.type != "array":
            raise ValueError("items is only valid for array schemas")
        return self

    def validate_output(self, value: Any, location: str = "output") -> None:
        checks = {
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "array": isinstance(value, list),
            "object": isinstance(value, dict),
        }
        if not checks[self.type]:
            raise PromptOutputError("%s must be a %s." % (location, self.type))
        if self.type == "object":
            missing = [name for name in self.required if name not in value]
            if missing:
                raise PromptOutputError("%s is missing required fields: %s." % (location, ", ".join(missing)))
            for name, schema in self.properties.items():
                if name in value:
                    schema.validate_output(value[name], "%s.%s" % (location, name))
        elif self.type == "array" and self.items is not None:
            for index, item in enumerate(value):
                self.items.validate_output(item, "%s[%d]" % (location, index))

    def validate_expected(self, value: Any, location: str) -> None:
        """Validate an assertion without requiring every output field to be asserted."""
        if self.type != "object":
            self.validate_output(value, location)
            return
        if not isinstance(value, dict):
            raise PromptValidationError("%s must be an object." % location)
        unknown = sorted(set(value) - set(self.properties))
        if unknown:
            raise PromptValidationError(
                "%s contains fields not declared by output.schema: %s." % (location, ", ".join(unknown))
            )
        for name, expected in value.items():
            self.properties[name].validate_expected(expected, "%s.%s" % (location, name))


class OutputSpec(SpecModel):
    format: OutputFormat
    output_schema: OutputSchema | None = Field(default=None, alias="schema")

    @model_validator(mode="after")
    def schema_requires_json(self) -> "OutputSpec":
        if self.output_schema is not None and self.format != "json":
            raise ValueError("schema is only valid when format is 'json'")
        return self


class PromptTestCase(SpecModel):
    name: str | None = Field(default=None, min_length=1)
    input: dict[str, Any]
    expected_output: str | None = None
    # Kept for existing prompt files; prefer expected_output for semantic LLM judging.
    expected: dict[str, Any] | None = None

    @field_validator("name")
    @classmethod
    def non_blank_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def has_an_expectation(self) -> "PromptTestCase":
        if self.expected_output is None and self.expected is None:
            raise ValueError("must define expected_output or expected")
        if self.expected_output is not None and not self.expected_output.strip():
            raise ValueError("expected_output must not be blank")
        return self


class TestingSpec(SpecModel):
    pass_threshold: float = Field(default=0.95, ge=0, le=1)


class PromptFileSpec(SpecModel):
    """Typed representation of the Prompt Ninja 1.0 TOML document."""

    spec_version: Literal[SUPPORTED_SPEC_VERSION]
    prompt: PromptMetadata
    model: ModelConfig
    template: TemplateSpec
    variables: list[VariableSpec] = Field(default_factory=list)
    output: OutputSpec
    testing: TestingSpec = Field(default_factory=TestingSpec)
    tests: list[PromptTestCase] = Field(default_factory=list)

    @property
    def variables_by_name(self) -> dict[str, VariableSpec]:
        return {variable.name: variable for variable in self.variables}

    @model_validator(mode="after")
    def validate_cross_field_rules(self) -> "PromptFileSpec":
        variables = self.variables_by_name
        if len(variables) != len(self.variables):
            raise ValueError("variable names must be unique")
        undeclared = sorted(self.template.referenced_variables - set(variables))
        if undeclared:
            raise ValueError("template variables have no [[variables]] definition: %s" % ", ".join(undeclared))
        for test in self.tests:
            unknown = sorted(set(test.input) - set(variables))
            if unknown:
                raise ValueError("test %r uses undeclared variables: %s" % (test.name, ", ".join(unknown)))
            for name, value in test.input.items():
                if not variables[name].accepts(value):
                    raise ValueError("test %r input %r must have type %s" % (test.name, name, variables[name].type))
            if test.expected is not None and self.output.output_schema is not None:
                self.output.output_schema.validate_expected(test.expected, "test %r expected" % test.name)
        return self


OutputSchema.model_rebuild()


class RuntimeModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class PreparedPrompt(RuntimeModel):
    """The fully rendered messages and configured model for one prompt run."""

    name: str
    provider: str
    model: str
    system: str
    user: str
    temperature: float | None = None
    max_output_tokens: int | None = None


class PromptRuntimeOptions(RuntimeModel):
    """Per-run overrides without mutating the versioned prompt definition."""

    model: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_output_tokens: int | None = Field(default=None, gt=0)


class PromptRunEvent(RuntimeModel):
    """A hook payload containing exactly what was sent or received in one run."""

    type: Literal["request", "response", "error"]
    run_id: str
    prompt_name: str
    provider: str
    model: str
    system: str
    user: str
    temperature: float | None = None
    max_output_tokens: int | None = None
    output: Any = None
    error: str | None = None


class TestJudgment(RuntimeModel):
    score: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class PromptTestResult(RuntimeModel):
    name: str
    passed: bool
    expected: dict[str, Any] | str
    actual: Any = None
    score: float | None = None
    rationale: str | None = None
    error: str | None = None


class PromptTestReport(RuntimeModel):
    prompt_name: str
    results: tuple[PromptTestResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)


PromptExecutor = Callable[[PreparedPrompt], Any]
AsyncPromptExecutor = Callable[[PreparedPrompt], Awaitable[Any]]
AsyncTestJudge = Callable[[PromptTestCase, Any], Awaitable[dict[str, Any]]]
PromptRunHook = Callable[[PromptRunEvent], Any]


class SamplingRunHook:
    """Forward a stable sample of complete runs to a storage or auto-fix sink."""

    def __init__(self, sink: PromptRunHook, sample_rate: float = 0.1, random_value: Callable[[], float] = random.random):
        if not 0 <= sample_rate <= 1:
            raise ValueError("sample_rate must be between 0 and 1.")
        self.sink = sink
        self.sample_rate = sample_rate
        self.random_value = random_value
        self._sampled_runs: set[str] = set()

    async def __call__(self, event: PromptRunEvent) -> None:
        if event.type == "request" and self.random_value() < self.sample_rate:
            self._sampled_runs.add(event.run_id)
        if event.run_id not in self._sampled_runs:
            return
        result = self.sink(event)
        if inspect.isawaitable(result):
            await result
        if event.type in {"response", "error"}:
            self._sampled_runs.discard(event.run_id)


class OpenAIPromptClient:
    """OpenAI-backed Prompt Ninja client with runtime controls and run hooks."""

    def __init__(self, client: Any | None = None):
        self.client = client

    async def execute(
        self,
        prompt: "PromptNinja",
        prepared: PreparedPrompt,
        *,
        runtime: PromptRuntimeOptions | None = None,
        output_model: type[BaseModel] | None = None,
        hooks: tuple[PromptRunHook, ...] = (),
    ) -> Any:
        if prepared.provider != "openai":
            raise PromptNinjaError("OpenAI execution requires model.provider = 'openai'.")
        if self.client is None:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI()
        options = runtime or PromptRuntimeOptions()
        model = options.model or prepared.model
        temperature = options.temperature if options.temperature is not None else prepared.temperature
        max_output_tokens = options.max_output_tokens or prepared.max_output_tokens
        run_id = str(uuid.uuid4())
        event_data = {
            "run_id": run_id,
            "prompt_name": prepared.name,
            "provider": prepared.provider,
            "model": model,
            "system": prepared.system,
            "user": prepared.user,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        await prompt._emit_hooks(hooks, PromptRunEvent(type="request", **event_data))
        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": prepared.system},
                {"role": "user", "content": prepared.user},
            ],
        }
        if prompt.spec.output.format == "json":
            request["response_format"] = {"type": "json_object"}
        if temperature is not None:
            request["temperature"] = temperature
        if max_output_tokens is not None:
            request["max_tokens"] = max_output_tokens
        try:
            response = await self.client.chat.completions.create(**request)
            output = prompt.validate_output(response.choices[0].message.content or "")
            if output_model is not None:
                output = output_model.model_validate(output)
        except Exception as exc:
            await prompt._emit_hooks(hooks, PromptRunEvent(type="error", error=str(exc), **event_data))
            raise
        await prompt._emit_hooks(hooks, PromptRunEvent(type="response", output=output, **event_data))
        return output


class PromptNinja:
    """A validated prompt definition backed by a ``*.prompt.toml`` file."""

    def __init__(self, definition: Mapping[str, Any], source: str = "<memory>"):
        self.definition = dict(definition)
        self.source = source
        self.spec = self._parse_definition(self.definition)

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptNinja":
        """Load a prompt definition from a file with the required extension."""
        prompt_path = Path(path)
        if not prompt_path.name.endswith(".prompt.toml"):
            raise PromptValidationError("Prompt files must use the .prompt.toml extension.")
        try:
            with prompt_path.open("rb") as prompt_file:
                definition = tomllib.load(prompt_file)
        except OSError as exc:
            raise PromptNinjaError("Unable to read prompt file %s: %s" % (prompt_path, exc)) from exc
        except tomllib.TOMLDecodeError as exc:
            raise PromptValidationError("Invalid TOML in %s: %s" % (prompt_path, exc)) from exc
        return cls(definition, source=str(prompt_path))

    @property
    def name(self) -> str:
        return self.spec.prompt.name

    @property
    def tests(self) -> list[PromptTestCase]:
        return self.spec.tests

    @property
    def variables(self) -> dict[str, VariableSpec]:
        return self.spec.variables_by_name

    def validate(self) -> None:
        """Re-validate the original TOML data after an external mutation."""
        self.spec = self._parse_definition(self.definition)

    def prepare(self, values: Mapping[str, Any], system_override: str | None = None) -> PreparedPrompt:
        """Validate variables and render system/user message templates."""
        if not isinstance(values, Mapping):
            raise PromptRenderError("Prompt inputs must be a mapping of variable names to values.")
        variables = self.variables
        unknown = sorted(set(values) - set(variables))
        if unknown:
            raise PromptRenderError("Undeclared prompt variables were provided: %s." % ", ".join(unknown))
        resolved = dict(values)
        for name, variable in variables.items():
            if name not in resolved and variable.has_default:
                resolved[name] = variable.default
            if name not in resolved and variable.required:
                raise PromptRenderError("Required variable %r was not provided." % name)
            if name in resolved and not variable.accepts(resolved[name]):
                raise PromptRenderError("Variable %r must have type %s." % (name, variable.type))

        return PreparedPrompt(
            name=self.name,
            provider=self.spec.model.provider,
            model=self.spec.model.name,
            system=system_override if system_override and system_override.strip() else self._render(self.spec.template.system, resolved),
            user=self._render(self.spec.template.user, resolved),
            temperature=self.spec.model.temperature,
            max_output_tokens=self.spec.model.max_output_tokens,
        )

    def run(self, values: Mapping[str, Any], executor: PromptExecutor) -> Any:
        """Prepare the prompt, execute it through ``executor``, and validate its output."""
        if not callable(executor):
            raise PromptNinjaError("executor must be a callable that accepts a PreparedPrompt.")
        return self.validate_output(executor(self.prepare(values)))

    def run_tests(self, executor: PromptExecutor) -> PromptTestReport:
        """Execute legacy exact-match test cases against an injected executor."""
        results: list[PromptTestResult] = []
        for index, test in enumerate(self.tests, start=1):
            name = test.name or "test %d" % index
            expected = test.expected_output or test.expected or {}
            try:
                if test.expected is None:
                    raise PromptNinjaError("Test %r uses expected_output and requires an async LLM judge." % name)
                actual = self.run(test.input, executor)
                if not self._contains_expected(actual, test.expected):
                    raise PromptOutputError("Output did not contain the expected values.")
                results.append(PromptTestResult(name=name, passed=True, expected=expected, actual=actual, score=1.0))
            except Exception as exc:  # A report should include every failing prompt case.
                results.append(PromptTestResult(name=name, passed=False, expected=expected, error=str(exc)))
        return PromptTestReport(prompt_name=self.name, results=tuple(results))

    async def arun(self, values: Mapping[str, Any], executor: AsyncPromptExecutor) -> Any:
        """Async counterpart to :meth:`run` for provider-backed execution."""
        if not callable(executor):
            raise PromptNinjaError("executor must be an async callable that accepts a PreparedPrompt.")
        return self.validate_output(await executor(self.prepare(values)))

    async def arun_tests(self, executor: AsyncPromptExecutor, judge: AsyncTestJudge | None = None) -> PromptTestReport:
        """Run embedded tests, using ``judge`` for natural-language expectations."""
        results: list[PromptTestResult] = []
        for index, test in enumerate(self.tests, start=1):
            name = test.name or "test %d" % index
            expected = test.expected_output or test.expected or {}
            try:
                actual = await self.arun(test.input, executor)
                if test.expected_output is not None:
                    if judge is None:
                        raise PromptNinjaError("Test %r uses expected_output and requires an LLM judge." % name)
                    verdict = TestJudgment.model_validate(await judge(test, actual))
                    results.append(PromptTestResult(
                        name=name,
                        passed=verdict.score >= self.spec.testing.pass_threshold,
                        expected=expected,
                        actual=actual,
                        score=verdict.score,
                        rationale=verdict.rationale,
                    ))
                elif not self._contains_expected(actual, test.expected):
                    raise PromptOutputError("Output did not contain the expected values.")
                else:
                    results.append(PromptTestResult(name=name, passed=True, expected=expected, actual=actual, score=1.0))
            except Exception as exc:  # A report should include every failing prompt case.
                results.append(PromptTestResult(name=name, passed=False, expected=expected, error=str(exc)))
        return PromptTestReport(prompt_name=self.name, results=tuple(results))

    async def run_openai(
        self,
        values: Mapping[str, Any],
        *,
        client: Any | None = None,
        model: str | None = None,
        runtime: PromptRuntimeOptions | None = None,
        system_override: str | None = None,
        output_model: type[BaseModel] | None = None,
        hooks: tuple[PromptRunHook, ...] = (),
    ) -> Any:
        """Render, execute, parse, and validate an OpenAI-backed prompt file."""
        prepared = self.prepare(values, system_override=system_override)
        return await self.execute_openai(prepared, client=client, model=model, runtime=runtime, output_model=output_model, hooks=hooks)

    def openai_client(self, client: Any | None = None) -> OpenAIPromptClient:
        """Create a provider client that can be shared across many prompt runs."""
        return OpenAIPromptClient(client)

    async def execute_openai(
        self,
        prepared: PreparedPrompt,
        *,
        client: Any | None = None,
        model: str | None = None,
        runtime: PromptRuntimeOptions | None = None,
        output_model: type[BaseModel] | None = None,
        hooks: tuple[PromptRunHook, ...] = (),
    ) -> Any:
        """Execute a prepared OpenAI prompt with optional runtime overrides and hooks."""
        options = runtime or PromptRuntimeOptions()
        if model is not None:
            options = options.model_copy(update={"model": model})
        return await OpenAIPromptClient(client).execute(self, prepared, runtime=options, output_model=output_model, hooks=hooks)

    def validate_output(self, output: Any) -> Any:
        """Parse and validate a provider response against the output declaration."""
        if self.spec.output.format == "text":
            if not isinstance(output, str):
                raise PromptOutputError("Text output must be a string.")
            return output
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError as exc:
                raise PromptOutputError("JSON output was not valid JSON: %s" % exc.msg) from exc
        if self.spec.output.output_schema is not None:
            self.spec.output.output_schema.validate_output(output)
        return output

    def _parse_definition(self, definition: Mapping[str, Any]) -> PromptFileSpec:
        try:
            return PromptFileSpec.model_validate(definition)
        except ValidationError as exc:
            raise PromptValidationError("Invalid prompt specification in %s:\n%s" % (self.source, exc)) from exc

    @staticmethod
    async def _emit_hooks(hooks: tuple[PromptRunHook, ...], event: PromptRunEvent) -> None:
        """Keep observability non-blocking: a failing hook never fails a model run."""
        for hook in hooks:
            try:
                result = hook(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                continue

    @staticmethod
    def _render(template: str, values: Mapping[str, Any]) -> str:
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in values:
                raise PromptRenderError("No value was provided for template variable %r." % name)
            value = values[name]
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            if isinstance(value, bool):
                return str(value).lower()
            return str(value)

        return _VARIABLE_PATTERN.sub(replace, template)

    @staticmethod
    def _contains_expected(actual: Any, expected: Any) -> bool:
        if isinstance(expected, dict):
            return isinstance(actual, dict) and all(
                key in actual and PromptNinja._contains_expected(actual[key], value)
                for key, value in expected.items()
            )
        if isinstance(expected, list):
            return isinstance(actual, list) and actual == expected
        return actual == expected
