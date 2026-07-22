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
import keyword
import os
import random
import re
import sys
import textwrap
import tomllib
import uuid
from datetime import date, datetime
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

SUPPORTED_SPEC_VERSION = "1.2"
OPENROUTER_API_ATTEMPTS = 3
_BUILTIN_VARIABLE_TYPES = {
    "string",
    "integer",
    "number",
    "boolean",
    "json",
    "dict",
    "date",
    "datetime",
    "dynamic",
    # Backwards-compatible aliases for prompt files written before spec 1.2.
    "array",
    "object",
}
_VARIABLE_TYPE_ALIASES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}
_LIST_TYPE_PATTERN = re.compile(r"^list\[(.+)\]$")
_VARIABLE_PATTERN = re.compile(
    r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)(?:\s*\|\s*([A-Za-z_][A-Za-z0-9_]*))?\s*\}\}"
)
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


_TOML_LINE_WIDTH = 120


def _wrap_long_lines(text: str, width: int = _TOML_LINE_WIDTH) -> str:
    """Visually rewrap lines longer than `width` without changing the value.

    A line broken here is rejoined with a TOML line-ending backslash: TOML
    discards a `\\` at the end of a line along with the newline and any
    leading whitespace that follows, so the extra breaks are purely a file
    presentation choice and the parsed string is unchanged. Lines that were
    already separated by a real newline in the source value stay separated by
    a real newline here too.
    """
    rendered_lines = []
    for line in text.split("\n"):
        if len(line) <= width:
            rendered_lines.append(line)
            continue
        # Reserve 2 columns for the trailing " \" continuation marker every
        # wrapped chunk but the last one gets, so no rendered line exceeds width.
        chunks = (
            textwrap.wrap(
                line, width=width - 2, break_long_words=False, break_on_hyphens=False
            )
            or [""]
        )
        rendered_lines.append(" \\\n".join(chunks))
    return "\n".join(rendered_lines)


def _toml_multiline_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"""', '""\\"')
    return '"""\n%s"""' % _wrap_long_lines(escaped)


def _toml_scalar(value: Any, *, prefix: str = "") -> str:
    """Serialize a top-level `key = value` line, wrapping long strings for readability.

    `prefix` is the literal text ("key = ") that will precede the returned value on
    the same line, so the width check reflects the actual rendered line length.
    """
    if isinstance(value, str):
        single_line = _toml_value(value)
        if "\n" in value or len(prefix) + len(single_line) > _TOML_LINE_WIDTH:
            return _toml_multiline_string(value)
        return single_line
    return _toml_value(value)


def _toml_value(value: Any) -> str:
    if isinstance(value, BaseModel):
        return _toml_value(value.model_dump(mode="json"))
    if isinstance(value, datetime):
        return json.dumps(value.isoformat())
    if isinstance(value, date):
        return value.isoformat()
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
    spec_version: Literal[SUPPORTED_SPEC_VERSION]
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    used_by: list[str] = Field(default_factory=list)
    version: str = Field(min_length=1)
    output: str

    @field_validator("name", "description")
    @classmethod
    def non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("used_by")
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


class LLMModelConfig(SpecModel):
    provider: str = Field(min_length=1)
    name: str = Field(min_length=1)

    @field_validator("provider", "name")
    @classmethod
    def non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class PromptSpec(SpecModel):
    system: str = ""
    user: str = ""

    @model_validator(mode="after")
    def has_a_message(self) -> "TemplateSpec":
        if not self.system.strip() and not self.user.strip():
            raise ValueError("must contain a non-empty system or user template")
        return self

    @property
    def referenced_variables(self) -> set[str]:
        return {
            match.group(1)
            for template in (self.system, self.user)
            for match in _VARIABLE_PATTERN.finditer(template)
        }


def _import_project_module(module_name: str):
    project_root = str(Path.cwd())
    added_project_root = project_root not in sys.path
    if added_project_root:
        sys.path.insert(0, project_root)
    try:
        return importlib.import_module(module_name)
    finally:
        if added_project_root:
            sys.path.remove(project_root)


def _resolve_variable_model(declaration: str) -> type[BaseModel] | None:
    """Resolve a dotted variable declaration only when it names a Pydantic model."""
    if "." not in declaration:
        return None
    try:
        module_name, attribute_name = declaration.rsplit(".", 1)
        model = getattr(_import_project_module(module_name), attribute_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValueError(
            "Pydantic model %r could not be imported" % declaration
        ) from exc
    if (
        not isinstance(model, type)
        or not issubclass(model, BaseModel)
        or model is BaseModel
    ):
        raise ValueError(
            "Pydantic model %r must resolve to a BaseModel class" % declaration
        )
    return model


def _json_ready(value: Any) -> Any:
    """Make supported runtime values safe for deterministic JSON rendering."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("%s is not JSON serializable" % type(value).__name__)


def _format_json(value: Any) -> str:
    return json.dumps(_json_ready(value), ensure_ascii=False)


class VariableSpec(SpecModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    type: str = Field(min_length=1)
    required: bool
    description: str = Field(min_length=1)
    default: Any = None

    @property
    def has_default(self) -> bool:
        return "default" in self.model_fields_set

    @field_validator("type")
    @classmethod
    def declared_type_is_supported(cls, value: str) -> str:
        value = _VARIABLE_TYPE_ALIASES.get(value, value)
        if value in _BUILTIN_VARIABLE_TYPES:
            return value
        nested = _LIST_TYPE_PATTERN.fullmatch(value)
        if nested:
            item_type = _VARIABLE_TYPE_ALIASES.get(nested.group(1), nested.group(1))
            if not item_type or item_type == value:
                raise ValueError(
                    "list type must name an item type, for example list[string]"
                )
            canonical_item_type = cls.declared_type_is_supported(item_type)
            return "list[%s]" % canonical_item_type
        model = _resolve_variable_model(value)
        if model is None:
            raise ValueError(
                "must be a supported built-in type, list[TYPE], or dotted Pydantic model path"
            )
        return value

    @property
    def model_class(self) -> type[BaseModel] | None:
        return _resolve_variable_model(self.type)

    @property
    def list_item_type(self) -> str | None:
        match = _LIST_TYPE_PATTERN.fullmatch(self.type)
        return match.group(1) if match else None

    def _coerce(self, declaration: str, value: Any) -> Any:
        if declaration == "dynamic":
            return value
        if declaration in {"json", "object"}:
            _json_ready(value)
            return value
        if declaration == "dict":
            if not isinstance(value, dict):
                raise ValueError("must be a dict")
            return value
        if declaration == "array":
            if not isinstance(value, list):
                raise ValueError("must be a list")
            return value
        nested = _LIST_TYPE_PATTERN.fullmatch(declaration)
        if nested:
            if not isinstance(value, list):
                raise ValueError("must be a list")
            return [self._coerce(nested.group(1), item) for item in value]
        if declaration == "string":
            if not isinstance(value, str):
                raise ValueError("must be a string")
            return value
        if declaration == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("must be an integer")
            return value
        if declaration == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("must be a number")
            return value
        if declaration == "boolean":
            if not isinstance(value, bool):
                raise ValueError("must be a boolean")
            return value
        if declaration == "date":
            if isinstance(value, datetime):
                raise ValueError("must be a date, not a datetime")
            if isinstance(value, date):
                return value
            if isinstance(value, str):
                try:
                    return date.fromisoformat(value)
                except ValueError as exc:
                    raise ValueError("must be an ISO 8601 date") from exc
            raise ValueError("must be a date or ISO 8601 date string")
        if declaration == "datetime":
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value)
                except ValueError as exc:
                    raise ValueError("must be an ISO 8601 datetime") from exc
            raise ValueError("must be a datetime or ISO 8601 datetime string")
        model = _resolve_variable_model(declaration)
        if isinstance(value, model):
            return value
        try:
            return model.model_validate(value)
        except ValidationError as exc:
            raise ValueError(
                "model validation failed: %s" % exc.errors(include_url=False)
            ) from exc

    def coerce(self, value: Any) -> Any:
        """Validate and normalize a runtime value before it reaches a template."""
        return self._coerce(self.type, value)

    def accepts(self, value: Any) -> bool:
        try:
            self.coerce(value)
        except ValueError:
            return False
        return True

    def render(self, value: Any, filter_name: str | None = None) -> str:
        if filter_name == "str":
            return str(value)
        if filter_name == "repr":
            return repr(value)
        if filter_name == "json":
            try:
                return _format_json(value)
            except ValueError as exc:
                raise PromptRenderError(
                    "Variable %r cannot be rendered with the json filter: %s"
                    % (self.name, exc)
                ) from exc
        if filter_name == "csv":
            if not isinstance(value, list):
                raise PromptRenderError(
                    "Variable %r uses the csv filter but its value is %s, not a list."
                    % (self.name, type(value).__name__)
                )
            return ", ".join(str(item) for item in value)
        if filter_name is not None:
            raise PromptRenderError(
                "Variable %r uses unsupported template filter %r."
                % (self.name, filter_name)
            )
        if self.type in {"json", "dict", "object"}:
            try:
                return _format_json(value)
            except ValueError as exc:
                raise PromptRenderError(
                    "Variable %r cannot be rendered as JSON: %s" % (self.name, exc)
                ) from exc
        if self.list_item_type is not None or self.type == "array":
            return ", ".join(str(item) for item in value)
        if self.model_class is not None:
            return repr(value)
        return str(value)

    @model_validator(mode="after")
    def default_matches_type(self) -> "VariableSpec":
        if self.has_default:
            try:
                self.default = self.coerce(self.default)
            except ValueError as exc:
                raise ValueError(
                    "default must have type %s: %s" % (self.type, exc)
                ) from exc
        return self


class BigIntOutput(RootModel[int]):
    """Structured-output wrapper for Prompt Ninja's BigInt declaration."""


class JsonObjectOutput(RootModel[dict[str, Any]]):
    """Generic structured output for an object without a domain model."""


class JsonArrayOutput(RootModel[list[Any]]):
    """Generic structured output for a top-level JSON array without a domain model."""


class PromptTestCase(SpecModel):
    name: str | None = Field(default=None, min_length=1)
    variable: dict[str, Any]
    expected_output: str | dict[str, Any]

    @field_validator("expected_output")
    @classmethod
    def expected_output_is_not_empty(
        cls, value: str | dict[str, Any]
    ) -> str | dict[str, Any]:
        if isinstance(value, str) and not value.strip():
            raise ValueError("must not be blank")
        if isinstance(value, dict) and not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("name")
    @classmethod
    def non_blank_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @property
    def input(self) -> dict[str, Any]:
        """Runtime test values retained for the execution API."""
        return self.variable


class TestingSpec(SpecModel):
    pass_threshold: float = Field(default=0.95, ge=0, le=1)


class PromptFileSpec(SpecModel):
    """Typed representation of the human-readable Prompt Ninja TOML document."""

    metadata: PromptMetadata
    llm_model: LLMModelConfig
    prompt: PromptSpec
    variables: list[VariableSpec] = Field(default_factory=list)
    testing: TestingSpec = Field(default_factory=TestingSpec)
    tests: list[PromptTestCase] = Field(default_factory=list)

    @property
    def spec_version(self) -> str:
        return self.metadata.spec_version

    @property
    def model(self) -> LLMModelConfig:
        """Compatibility view for the runtime's model resolution."""
        return self.llm_model

    @property
    def template(self) -> PromptSpec:
        """Compatibility view for template rendering."""
        return self.prompt

    @property
    def output(self) -> str:
        """Compatibility view for output validation."""
        return self.metadata.output

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
            unknown = sorted(set(test.variable) - set(variables))
            if unknown:
                raise ValueError(
                    "test %r uses undeclared variables: %s"
                    % (test.name, ", ".join(unknown))
                )
            for name, value in test.variable.items():
                try:
                    test.variable[name] = variables[name].coerce(value)
                except ValueError as exc:
                    raise ValueError(
                        "test %r variable %r expects %s; received %s: %s"
                        % (
                            test.name,
                            name,
                            variables[name].type,
                            type(value).__name__,
                            exc,
                        )
                    ) from exc
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
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class TestJudgment(RuntimeModel):
    score: float = Field(
        ge=0, le=1, description="Semantic correctness score from 0.0 to 1.0."
    )
    rationale: str = Field(
        min_length=1, description="Brief explanation supporting the score."
    )
    prompt_suggestion: str | None = Field(
        default=None,
        description="Actionable prompt change when the prompt caused the failure.",
    )
    test_suggestion: str | None = Field(
        default=None,
        description="Actionable test-case change when the expectation is unclear or flawed.",
    )


class PromptTestResult(RuntimeModel):
    name: str
    passed: bool
    expected: dict[str, Any] | str
    input: dict[str, Any] | None = None
    actual: Any = None
    score: float | None = None
    rationale: str | None = None
    prompt_suggestion: str | None = None
    test_suggestion: str | None = None
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
PromptTestCaseCallback = Callable[[PromptTestCase], Any]
PromptTestResultCallback = Callable[[PromptTestResult], Any]
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
        model = getattr(_import_project_module(module_name), attribute_name)
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


def _output_instruction(
    declaration: str,
    output_model: type[BaseModel] | None,
) -> str:
    """Turn metadata.output into the instruction every model invocation receives."""
    output_type = _output_format(declaration)
    if output_type == "text":
        return (
            "Output contract (metadata.output = %r): return plain text only."
            % declaration
        )
    if output_type == "integer":
        return (
            "Output contract (metadata.output = %r): return one JSON integer only."
            % declaration
        )
    schema = json.dumps(
        _output_json_schema(declaration, output_model),
        ensure_ascii=False,
        sort_keys=True,
    )
    return (
        "Output contract (metadata.output = %r): return only JSON that "
        "validates against this schema:\n%s" % (declaration, schema)
    )


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


def _validate_test_expectations(
    spec: PromptFileSpec, output_model: type[BaseModel] | None, source: str
) -> None:
    for index, test in enumerate(spec.tests, start=1):
        if not isinstance(test.expected_output, dict):
            continue
        name = test.name or "test %d" % index
        if output_model is None:
            raise PromptValidationError(
                "Test %r in %s has an object expected_output, but metadata.output "
                "does not declare a Pydantic model." % (name, source)
            )
        try:
            output_model.model_validate(test.expected_output)
        except ValidationError as exc:
            raise PromptValidationError(
                "Test %r expected_output does not match %s:\n%s"
                % (name, spec.output, exc)
            ) from exc


def _render_template(
    template: str,
    values: Mapping[str, Any],
    variables: Mapping[str, VariableSpec],
) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise PromptRenderError(
                "No value was provided for template variable %r." % name
            )
        return variables[name].render(values[name], match.group(2))

    return _VARIABLE_PATTERN.sub(replace, template)


def _prepare_prompt(
    spec: PromptFileSpec,
    values: Mapping[str, Any],
    system_override: str | None,
    output_model: type[BaseModel] | None,
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
            raise PromptRenderError(
                "Required variable %r (type %s) was not provided. "
                "Supply it at runtime or declare a default." % (name, variable.type)
            )
        if name in resolved:
            received = resolved[name]
            try:
                resolved[name] = variable.coerce(received)
            except ValueError as exc:
                raise PromptRenderError(
                    "Variable %r expects %s; received %s: %s"
                    % (name, variable.type, type(received).__name__, exc)
                ) from exc

    rendered_system = _render_template(
        (
            system_override
            if system_override and system_override.strip()
            else spec.template.system
        ),
        resolved,
        variables,
    )
    output_instruction = _output_instruction(spec.output, output_model)
    system = (
        "%s\n\n%s" % (rendered_system.rstrip(), output_instruction)
        if rendered_system.strip()
        else output_instruction
    )

    return PreparedPrompt(
        name=spec.metadata.name,
        provider=spec.model.provider,
        model=spec.model.name,
        system=system,
        user=_render_template(spec.template.user, resolved, variables),
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
        expected = test.expected_output
        try:
            raise PromptNinjaError(
                "Test %r has semantic expected_output and requires an async LLM judge."
                % name
            )
        except Exception as exc:  # A report should include every failing case.
            results.append(
                PromptTestResult(
                    name=name,
                    passed=False,
                    expected=expected,
                    input=test.input,
                    error=str(exc),
                )
            )
    return PromptTestReport(prompt_name=prompt.name, results=tuple(results))


async def _run_prompt_tests_async(
    prompt: "PromptNinja",
    executor: AsyncPromptExecutor,
    judge: AsyncTestJudge | None,
    on_start: PromptTestCaseCallback | None = None,
    on_result: PromptTestResultCallback | None = None,
) -> PromptTestReport:
    results: list[PromptTestResult] = []
    for index, test in enumerate(prompt.tests, start=1):
        name = test.name or "test %d" % index
        expected = test.expected_output
        if on_start is not None:
            callback_result = on_start(test)
            if inspect.isawaitable(callback_result):
                await callback_result
        try:
            actual = await prompt.arun(test.input, executor)
            if judge is None:
                raise PromptNinjaError(
                    "Test %r uses expected_output and requires an LLM judge." % name
                )
            verdict = TestJudgment.model_validate(await judge(test, actual))
            result = PromptTestResult(
                name=name,
                passed=verdict.score >= prompt.spec.testing.pass_threshold,
                expected=expected,
                input=test.input,
                actual=actual,
                score=verdict.score,
                rationale=verdict.rationale,
                prompt_suggestion=verdict.prompt_suggestion,
                test_suggestion=verdict.test_suggestion,
            )
        except Exception as exc:  # A report should include every failing case.
            result = PromptTestResult(
                name=name,
                passed=False,
                expected=expected,
                input=test.input,
                error=str(exc),
            )
        results.append(result)
        if on_result is not None:
            callback_result = on_result(result)
            if inspect.isawaitable(callback_result):
                await callback_result
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


class OpenRouterPromptClient:
    """OpenRouter-backed Prompt Ninja client with runtime controls and run hooks."""

    def __init__(self, client: Any | None = None):
        self.client = client
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close the internally created OpenAI client before its event loop exits."""
        if not self._owns_client or self.client is None:
            return
        close = getattr(self.client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        self.client = None

    async def execute(
        self,
        prompt: "PromptNinja",
        prepared: PreparedPrompt,
        *,
        runtime: PromptRuntimeOptions | None = None,
        output_model: type[BaseModel] | None = None,
        hooks: tuple[PromptRunHook, ...] = (),
    ) -> Any:
        if prepared.provider != "openrouter":
            raise PromptNinjaError(
                "OpenRouter execution requires model.provider = 'openrouter'."
            )
        if self.client is None:
            from openai import AsyncOpenAI
            from .model_config import OPENROUTER_BASE_URL, openrouter_headers

            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise PromptNinjaError("OPENROUTER_API_KEY is required to run prompts.")
            headers = openrouter_headers()
            headers.pop("Authorization", None)
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
                default_headers=headers,
                max_retries=OPENROUTER_API_ATTEMPTS - 1,
            )
        options = runtime or PromptRuntimeOptions()
        model = options.model or prepared.model
        instructions = prepared.system
        input_text = prepared.user
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
                    "The configured OpenRouter client does not support the Responses API. "
                    "Run `uv sync` from the backend directory and restart the service "
                    "to install the pinned openai SDK."
                )
            effective_output_model = (
                BigIntOutput
                if prompt.output_format == "integer"
                else output_model or prompt.output_model
            )
            parse_response = getattr(responses_api, "parse", None)
            provider_can_parse_model = (
                effective_output_model is not None
                and effective_output_model is not JsonObjectOutput
                and effective_output_model is not JsonArrayOutput
            )
            if provider_can_parse_model and callable(parse_response):
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
            PromptRunEvent(
                type="response",
                output=output,
                input_tokens=getattr(
                    getattr(response, "usage", None), "input_tokens", None
                ),
                output_tokens=getattr(
                    getattr(response, "usage", None), "output_tokens", None
                ),
                total_tokens=getattr(
                    getattr(response, "usage", None), "total_tokens", None
                ),
                **event_data,
            ),
        )
        return output


class PromptNinja:
    """A validated prompt definition backed by a ``*.prompt.toml`` file."""

    def __init__(self, definition: Mapping[str, Any], source: str = "<memory>"):
        self.definition = dict(definition)
        self.source = source
        self.spec = _parse_prompt_spec(self.definition, self.source)
        self._output_model = _resolve_output_model(self.spec.output)
        _validate_test_expectations(self.spec, self._output_model, self.source)

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
        """Serialize this validated prompt definition using readable TOML tables."""
        definition = self.spec.model_dump(by_alias=True, exclude_none=True)

        def assignment(key: str, value: Any, key_prefix: str = "") -> str:
            toml_key = key_prefix + _toml_key(key)
            return "%s = %s" % (toml_key, _toml_scalar(value, prefix=toml_key + " = "))

        def table(name: str, values: Mapping[str, Any]) -> list[str]:
            return [
                f"[{name}]",
                *[assignment(key, value) for key, value in values.items()],
            ]

        lines = [
            *table("metadata", definition["metadata"]),
            "",
            *table("llm_model", definition["llm_model"]),
            "",
            *table("prompt", definition["prompt"]),
        ]
        for variable in definition["variables"]:
            lines.extend(["", "[[variables]]"])
            lines.extend(
                assignment(key, value) for key, value in variable.items()
            )
        if testing := definition.get("testing"):
            lines.extend(["", *table("testing", testing)])
        for test in definition["tests"]:
            lines.extend(["", "[[tests]]"])
            lines.extend(
                assignment(key, value)
                for key, value in test.items()
                if key != "variable"
            )
            lines.extend(
                assignment(key, value, key_prefix="variable.")
                for key, value in test["variable"].items()
            )
        return "\n".join(lines) + "\n"

    @property
    def name(self) -> str:
        return self.spec.metadata.name

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
        _validate_test_expectations(self.spec, self._output_model, self.source)

    def prepare(
        self, values: Mapping[str, Any], system_override: str | None = None
    ) -> PreparedPrompt:
        """Validate variables and render system/user message templates."""
        return _prepare_prompt(
            self.spec,
            values,
            system_override,
            self.output_model,
        )

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
        self,
        executor: AsyncPromptExecutor,
        judge: AsyncTestJudge | None = None,
        on_start: PromptTestCaseCallback | None = None,
        on_result: PromptTestResultCallback | None = None,
    ) -> PromptTestReport:
        """Run embedded tests and optionally notify after every completed case."""
        return await _run_prompt_tests_async(self, executor, judge, on_start, on_result)

    async def run_openrouter(
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
        """Render, execute, parse, and validate an OpenRouter-backed prompt file."""
        prepared = self.prepare(values, system_override=system_override)
        options = runtime or PromptRuntimeOptions()
        if model is not None:
            options = options.model_copy(update={"model": model})
        prompt_client = OpenRouterPromptClient(client)
        try:
            return await prompt_client.execute(
                self,
                prepared,
                runtime=options,
                output_model=output_model,
                hooks=hooks,
            )
        finally:
            await prompt_client.aclose()

    def validate_output(self, output: Any) -> Any:
        """Parse and validate a provider response against the output declaration."""
        return _validate_output(self.spec.output, self.output_model, output)


class PromptCollection(Mapping[str, PromptNinja]):
    """An eagerly loaded view of all ``*.prompt.toml`` files in a directory.

    Exact metadata names are available by index; dot access uses a Python-safe
    alias, replacing separators such as ``-`` with ``_``.
    """

    def __init__(self, dir: str | Path):
        self.directory = Path(dir)
        if not self.directory.is_dir():
            raise PromptNinjaError(
                "Prompt collection directory %s does not exist or is not a directory."
                % self.directory
            )
        paths = sorted(self.directory.glob("*.prompt.toml"))
        if not paths:
            raise PromptNinjaError(
                "No *.prompt.toml files found in %s." % self.directory
            )
        prompts = [PromptNinja.from_file(path) for path in paths]
        by_name = {prompt.name: prompt for prompt in prompts}
        if len(by_name) != len(prompts):
            raise PromptValidationError(
                "Prompt collection %s contains duplicate metadata.name values."
                % self.directory
            )
        aliases: dict[str, PromptNinja] = {}
        for prompt in prompts:
            alias = self._attribute_alias(prompt.name)
            if alias is None:
                continue
            if hasattr(type(self), alias):
                raise PromptValidationError(
                    "Prompt name %r cannot use dot-access alias %r because it is reserved "
                    "by PromptCollection." % (prompt.name, alias)
                )
            existing = aliases.get(alias)
            if existing is not None:
                raise PromptValidationError(
                    "Prompt names %r and %r share dot-access alias %r."
                    % (existing.name, prompt.name, alias)
                )
            aliases[alias] = prompt
        self._by_name = by_name
        self._aliases = aliases

    @staticmethod
    def _attribute_alias(name: str) -> str | None:
        alias = re.sub(r"[^A-Za-z0-9_]", "_", name)
        if not alias or not alias.isidentifier() or keyword.iskeyword(alias):
            return None
        return alias

    def __getitem__(self, name: str) -> PromptNinja:
        return self._by_name[name]

    def __iter__(self):
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    def __getattr__(self, name: str) -> PromptNinja:
        try:
            return self._aliases[name]
        except KeyError as exc:
            raise AttributeError(
                "PromptCollection has no prompt alias %r." % name
            ) from exc

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self._aliases))

    @property
    def names(self) -> tuple[str, ...]:
        """Exact metadata names in deterministic filename order."""
        return tuple(self._by_name)
