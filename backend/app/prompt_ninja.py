"""Load, validate, render, and test versioned ``*.prompt.toml`` files.

TOML is parsed into a typed Pydantic model before it is used. This keeps the
file format's structural rules close to the format itself, while
``PromptNinja`` owns only runtime work: rendering, executing, and checking a
model response.
"""

from __future__ import annotations

import inspect
import importlib
import json
import random
import re
import tomllib
import uuid
from pathlib import Path, PurePosixPath
from collections.abc import Awaitable
from typing import Any, Callable, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationError,
    field_validator,
    model_validator,
)

SUPPORTED_SPEC_VERSION = "1.0"
VariableType = Literal["string", "integer", "number", "boolean", "array", "object"]
_VARIABLE_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_TOML_BARE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class PromptNinjaError(ValueError):
    """Base error for an invalid prompt or prompt operation."""


class PromptValidationError(PromptNinjaError):
    """Raised when a prompt file does not satisfy the prompt-file specification."""


class PromptRenderError(PromptNinjaError):
    """Raised when variables cannot be rendered into a prompt template."""


class PromptOutputError(PromptNinjaError):
    """Raised when a model response does not match the declared output contract."""


def _toml_key(key: str) -> str:
    return (
        key
        if _TOML_BARE_KEY_PATTERN.fullmatch(key)
        else json.dumps(key, ensure_ascii=False)
    )


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{ "
            + ", ".join(
                "%s = %s" % (_toml_key(str(key)), _toml_value(item))
                for key, item in value.items()
            )
            + " }"
        )
    raise PromptNinjaError("Cannot serialize %r to TOML." % (type(value).__name__,))


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
        for item in value:
            path = PurePosixPath(item)
            if (
                not item.strip()
                or path.is_absolute()
                or ".." in path.parts
                or not path.suffix
            ):
                raise ValueError("must contain only repository-relative file paths")
        return value


class ModelConfig(SpecModel):
    provider: str = Field(min_length=1)
    name: str = Field(min_length=1)

    @field_validator("provider", "name")
    @classmethod
    def non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
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
        return set(_VARIABLE_PATTERN.findall(self.system)) | set(
            _VARIABLE_PATTERN.findall(self.user)
        )


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


class BigIntOutput(RootModel[int]):
    """Structured-output wrapper for Prompt Ninja's BigInt declaration."""


class JsonObjectOutput(RootModel[dict[str, Any]]):
    """Generic structured output for an object without a domain model."""


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
    output: str
    testing: TestingSpec = Field(default_factory=TestingSpec)
    tests: list[PromptTestCase] = Field(default_factory=list)

    @property
    def variables_by_name(self) -> dict[str, VariableSpec]:
        return {variable.name: variable for variable in self.variables}

    @field_validator("output")
    @classmethod
    def validate_output_declaration(cls, value: str) -> str:
        if value.casefold() in {"string", "bigint"}:
            return value
        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+",
            value,
        ):
            raise ValueError("must be String, BigInt, or a dotted Pydantic model path")
        return value

    @model_validator(mode="after")
    def validate_cross_field_rules(self) -> "PromptFileSpec":
        variables = self.variables_by_name
        if len(variables) != len(self.variables):
            raise ValueError("variable names must be unique")
        undeclared = sorted(self.template.referenced_variables - set(variables))
        if undeclared:
            raise ValueError(
                "template variables have no [[variables]] definition: %s"
                % ", ".join(undeclared)
            )
        missing_required = sorted(
            name
            for name, variable in variables.items()
            if variable.required and name not in self.template.referenced_variables
        )
        if missing_required:
            raise ValueError(
                "required variables are not referenced by the template: %s"
                % ", ".join(missing_required)
            )
        for test in self.tests:
            unknown = sorted(set(test.input) - set(variables))
            if unknown:
                raise ValueError(
                    "test %r uses undeclared variables: %s"
                    % (test.name, ", ".join(unknown))
                )
            for name, value in test.input.items():
                if not variables[name].accepts(value):
                    raise ValueError(
                        "test %r input %r must have type %s"
                        % (test.name, name, variables[name].type)
                    )
        return self


class RuntimeModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class PreparedPrompt(RuntimeModel):
    """The fully rendered messages and configured model for one prompt run."""

    name: str
    provider: str
    model: str
    system: str
    user: str


class PromptRuntimeOptions(RuntimeModel):
    """Per-run overrides without mutating the versioned prompt definition."""

    model: str | None = None


class PromptRunEvent(RuntimeModel):
    """A hook payload containing exactly what was sent or received in one run."""

    type: Literal["request", "response", "error"]
    run_id: str
    prompt_name: str
    provider: str
    model: str
    system: str
    user: str
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


def _parse_prompt_spec(definition: Mapping[str, Any], source: str) -> PromptFileSpec:
    try:
        return PromptFileSpec.model_validate(definition)
    except ValidationError as exc:
        raise PromptValidationError(
            "Invalid prompt specification in %s:\n%s" % (source, exc)
        ) from exc


def _resolve_output_model(output: str) -> type[BaseModel] | None:
    if output.casefold() in {"string", "bigint"}:
        return None
    try:
        module_name, attribute_name = output.rsplit(".", 1)
        model = getattr(importlib.import_module(module_name), attribute_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise PromptValidationError(
            "Output model %r could not be imported." % output
        ) from exc
    if (
        not isinstance(model, type)
        or not issubclass(model, BaseModel)
        or model is BaseModel
    ):
        raise PromptValidationError(
            "Output model %r must resolve to a Pydantic BaseModel class." % output
        )
    return model


def _output_format(output: str) -> str:
    return {
        "string": "text",
        "bigint": "integer",
    }.get(output.casefold(), "json")


def _output_json_schema(
    output: str,
    output_model: type[BaseModel] | None,
) -> dict[str, Any]:
    output_type = _output_format(output)
    if output_type == "text":
        return {"type": "string"}
    if output_type == "integer":
        return {"type": "integer"}
    return output_model.model_json_schema() if output_model else {"type": "object"}


def _validate_output(
    declaration: str,
    output_model: type[BaseModel] | None,
    value: Any,
) -> Any:
    output_type = _output_format(declaration)
    if output_type == "text":
        if not isinstance(value, str):
            raise PromptOutputError("String output must be a string.")
        return value
    if output_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise PromptOutputError("BigInt output must be an integer.")
        return value
    if output_model is not None:
        if isinstance(value, output_model):
            return value
        if isinstance(value, str):
            return output_model.model_validate_json(value)
        return output_model.model_validate(value)
    raise PromptOutputError("Structured output requires a Pydantic model.")


def _render_template(template: str, values: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise PromptRenderError(
                "No value was provided for template variable %r." % name
            )
        value = values[name]
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    return _VARIABLE_PATTERN.sub(replace, template)


def _prepare_prompt(
    spec: PromptFileSpec,
    values: Mapping[str, Any],
    system_override: str | None,
) -> PreparedPrompt:
    if not isinstance(values, Mapping):
        raise PromptRenderError(
            "Prompt inputs must be a mapping of variable names to values."
        )
    variables = spec.variables_by_name
    unknown = sorted(set(values) - set(variables))
    if unknown:
        raise PromptRenderError(
            "Undeclared prompt variables were provided: %s." % ", ".join(unknown)
        )
    resolved = dict(values)
    for name, variable in variables.items():
        if name not in resolved and variable.has_default:
            resolved[name] = variable.default
        if name not in resolved and variable.required:
            raise PromptRenderError("Required variable %r was not provided." % name)
        if name in resolved and not variable.accepts(resolved[name]):
            raise PromptRenderError(
                "Variable %r must have type %s." % (name, variable.type)
            )

    return PreparedPrompt(
        name=spec.prompt.name,
        provider=spec.model.provider,
        model=spec.model.name,
        system=_render_template(
            (
                system_override
                if system_override and system_override.strip()
                else spec.template.system
            ),
            resolved,
        ),
        user=_render_template(spec.template.user, resolved),
    )


def _contains_expected(actual: Any, expected: Any) -> bool:
    if isinstance(actual, BaseModel):
        actual = actual.model_dump()
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_expected(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and actual == expected
    return actual == expected


async def _emit_hooks(
    hooks: tuple[PromptRunHook, ...],
    event: PromptRunEvent,
) -> None:
    """Keep observability non-blocking: a failing hook never fails a model run."""
    for hook in hooks:
        try:
            result = hook(event)
            if inspect.isawaitable(result):
                await result
        except Exception:
            continue


def _run_prompt_tests(
    prompt: "PromptNinja",
    executor: PromptExecutor,
) -> PromptTestReport:
    results: list[PromptTestResult] = []
    for index, test in enumerate(prompt.tests, start=1):
        name = test.name or "test %d" % index
        expected = test.expected_output or test.expected or {}
        try:
            if test.expected is None:
                raise PromptNinjaError(
                    "Test %r uses expected_output and requires an async LLM judge."
                    % name
                )
            actual = prompt.run(test.input, executor)
            if not _contains_expected(actual, test.expected):
                raise PromptOutputError("Output did not contain the expected values.")
            results.append(
                PromptTestResult(
                    name=name,
                    passed=True,
                    expected=expected,
                    actual=actual,
                    score=1.0,
                )
            )
        except Exception as exc:  # A report should include every failing case.
            results.append(
                PromptTestResult(
                    name=name,
                    passed=False,
                    expected=expected,
                    error=str(exc),
                )
            )
    return PromptTestReport(prompt_name=prompt.name, results=tuple(results))


async def _run_prompt_tests_async(
    prompt: "PromptNinja",
    executor: AsyncPromptExecutor,
    judge: AsyncTestJudge | None,
) -> PromptTestReport:
    results: list[PromptTestResult] = []
    for index, test in enumerate(prompt.tests, start=1):
        name = test.name or "test %d" % index
        expected = test.expected_output or test.expected or {}
        try:
            actual = await prompt.arun(test.input, executor)
            if test.expected_output is not None:
                if judge is None:
                    raise PromptNinjaError(
                        "Test %r uses expected_output and requires an LLM judge." % name
                    )
                verdict = TestJudgment.model_validate(await judge(test, actual))
                results.append(
                    PromptTestResult(
                        name=name,
                        passed=verdict.score >= prompt.spec.testing.pass_threshold,
                        expected=expected,
                        actual=actual,
                        score=verdict.score,
                        rationale=verdict.rationale,
                    )
                )
            elif not _contains_expected(actual, test.expected):
                raise PromptOutputError("Output did not contain the expected values.")
            else:
                results.append(
                    PromptTestResult(
                        name=name,
                        passed=True,
                        expected=expected,
                        actual=actual,
                        score=1.0,
                    )
                )
        except Exception as exc:  # A report should include every failing case.
            results.append(
                PromptTestResult(
                    name=name,
                    passed=False,
                    expected=expected,
                    error=str(exc),
                )
            )
    return PromptTestReport(prompt_name=prompt.name, results=tuple(results))


class SamplingRunHook:
    """Forward a stable sample of complete runs to a storage or auto-fix sink."""

    def __init__(
        self,
        sink: PromptRunHook,
        sample_rate: float = 0.1,
        random_value: Callable[[], float] = random.random,
    ):
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
            raise PromptNinjaError(
                "OpenAI execution requires model.provider = 'openai'."
            )
        if self.client is None:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI()
        options = runtime or PromptRuntimeOptions()
        model = options.model or prepared.model
        instructions = prepared.system
        input_text = prepared.user
        if prompt.output_format == "json":
            instructions = f"{instructions.rstrip()}\n\nReturn a valid JSON object."
            input_text = f"{input_text.rstrip()}\n\nReturn a valid JSON object."
        run_id = str(uuid.uuid4())
        event_data = {
            "run_id": run_id,
            "prompt_name": prepared.name,
            "provider": prepared.provider,
            "model": model,
            "system": instructions,
            "user": input_text,
        }
        await _emit_hooks(hooks, PromptRunEvent(type="request", **event_data))
        request: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_text,
            "store": False,
        }
        try:
            responses_api = getattr(self.client, "responses", None)
            if responses_api is None:
                raise PromptNinjaError(
                    "The configured OpenAI client does not support the Responses API. "
                    "Run `uv sync` from the backend directory and restart the service "
                    "to install the pinned openai SDK."
                )
            effective_output_model = (
                BigIntOutput
                if prompt.output_format == "integer"
                else output_model or prompt.output_model
            )
            parse_response = getattr(responses_api, "parse", None)
            if effective_output_model is not None and callable(parse_response):
                response = await parse_response(
                    **request,
                    text_format=effective_output_model,
                )
                output = response.output_parsed
                if output is None:
                    raise PromptOutputError(
                        "The Responses API returned no parsed structured output."
                    )
                if prompt.output_format == "integer":
                    output = output.root
            else:
                response = await responses_api.create(**request)
                output_text = response.output_text or ""
                if effective_output_model is None:
                    output = prompt.validate_output(output_text)
                else:
                    output = effective_output_model.model_validate_json(output_text)
                    if prompt.output_format == "integer":
                        output = output.root
        except Exception as exc:
            await _emit_hooks(
                hooks,
                PromptRunEvent(type="error", error=str(exc), **event_data),
            )
            raise
        await _emit_hooks(
            hooks,
            PromptRunEvent(type="response", output=output, **event_data),
        )
        return output


class PromptNinja:
    """A validated prompt definition backed by a ``*.prompt.toml`` file."""

    def __init__(self, definition: Mapping[str, Any], source: str = "<memory>"):
        self.definition = dict(definition)
        self.source = source
        self.spec = _parse_prompt_spec(self.definition, self.source)
        self._output_model = _resolve_output_model(self.spec.output)

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptNinja":
        """Load a prompt definition from a file with the required extension."""
        prompt_path = Path(path)
        if not prompt_path.name.endswith(".prompt.toml"):
            raise PromptValidationError(
                "Prompt files must use the .prompt.toml extension."
            )
        try:
            with prompt_path.open("rb") as prompt_file:
                definition = tomllib.load(prompt_file)
        except OSError as exc:
            raise PromptNinjaError(
                "Unable to read prompt file %s: %s" % (prompt_path, exc)
            ) from exc
        except tomllib.TOMLDecodeError as exc:
            raise PromptValidationError(
                "Invalid TOML in %s: %s" % (prompt_path, exc)
            ) from exc
        return cls(definition, source=str(prompt_path))

    def to_toml(self) -> str:
        """Serialize this validated prompt definition as a TOML document."""
        definition = self.spec.model_dump(by_alias=True, exclude_none=True)
        return (
            "\n".join(
                "%s = %s" % (_toml_key(key), _toml_value(value))
                for key, value in definition.items()
            )
            + "\n"
        )

    @property
    def name(self) -> str:
        return self.spec.prompt.name

    @property
    def tests(self) -> list[PromptTestCase]:
        return self.spec.tests

    @property
    def variables(self) -> dict[str, VariableSpec]:
        return self.spec.variables_by_name

    @property
    def output_format(self) -> str:
        return _output_format(self.spec.output)

    @property
    def output_model(self) -> type[BaseModel] | None:
        return self._output_model

    @property
    def output_json_schema(self) -> dict[str, Any]:
        return _output_json_schema(self.spec.output, self.output_model)

    def validate(self) -> None:
        """Re-validate the original TOML data after an external mutation."""
        self.spec = _parse_prompt_spec(self.definition, self.source)
        self._output_model = _resolve_output_model(self.spec.output)

    def prepare(
        self, values: Mapping[str, Any], system_override: str | None = None
    ) -> PreparedPrompt:
        """Validate variables and render system/user message templates."""
        return _prepare_prompt(self.spec, values, system_override)

    def run(self, values: Mapping[str, Any], executor: PromptExecutor) -> Any:
        """Prepare the prompt, execute it through ``executor``, and validate its output."""
        if not callable(executor):
            raise PromptNinjaError(
                "executor must be a callable that accepts a PreparedPrompt."
            )
        return self.validate_output(executor(self.prepare(values)))

    def run_tests(self, executor: PromptExecutor) -> PromptTestReport:
        """Execute legacy exact-match test cases against an injected executor."""
        return _run_prompt_tests(self, executor)

    async def arun(
        self, values: Mapping[str, Any], executor: AsyncPromptExecutor
    ) -> Any:
        """Async counterpart to :meth:`run` for provider-backed execution."""
        if not callable(executor):
            raise PromptNinjaError(
                "executor must be an async callable that accepts a PreparedPrompt."
            )
        return self.validate_output(await executor(self.prepare(values)))

    async def arun_tests(
        self, executor: AsyncPromptExecutor, judge: AsyncTestJudge | None = None
    ) -> PromptTestReport:
        """Run embedded tests, using ``judge`` for natural-language expectations."""
        return await _run_prompt_tests_async(self, executor, judge)

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
        options = runtime or PromptRuntimeOptions()
        if model is not None:
            options = options.model_copy(update={"model": model})
        return await OpenAIPromptClient(client).execute(
            self,
            prepared,
            runtime=options,
            output_model=output_model,
            hooks=hooks,
        )

    def validate_output(self, output: Any) -> Any:
        """Parse and validate a provider response against the output declaration."""
        return _validate_output(self.spec.output, self.output_model, output)
