"""Typed structured-output model for the board's prompt compiler."""

import keyword
import re
from datetime import date, datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
    model_validator,
)

from .core import (
    LLMModelConfig,
    PromptMetadata,
    PromptSpec,
    TestingSpec,
)


class CompiledVariableSpec(BaseModel):
    """Compiler-facing variable declaration without an untyped default value."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    type: str = Field(min_length=1)
    required: bool
    description: str = Field(min_length=1)

    @property
    def has_default(self) -> bool:
        return False


class CompiledPromptDefinition(BaseModel):
    """Strict-output-safe prompt definition; runtime tests are attached locally."""

    model_config = ConfigDict(extra="forbid")

    metadata: PromptMetadata
    llm_model: LLMModelConfig
    prompt: PromptSpec
    variables: list[CompiledVariableSpec] = Field(default_factory=list)
    testing: TestingSpec = Field(default_factory=TestingSpec)

    @property
    def output(self) -> str:
        return self.metadata.output


class CompiledOutputField(BaseModel):
    """Portable field declaration used to generate a companion Pydantic model."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    type: Literal[
        "string",
        "integer",
        "number",
        "boolean",
        "date",
        "datetime",
        "list[string]",
        "list[integer]",
        "list[number]",
        "list[boolean]",
        "list[date]",
        "list[datetime]",
    ]
    description: str = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def name_is_not_a_keyword(cls, value: str) -> str:
        if keyword.iskeyword(value):
            raise ValueError("output model field names cannot be Python keywords")
        return value


class CompiledOutputModel(BaseModel):
    """Concrete flat output model that can be written into the user's project."""

    model_config = ConfigDict(extra="forbid")

    class_name: str = Field(pattern=r"^[A-Z][A-Za-z0-9]*$")
    fields: list[CompiledOutputField] = Field(min_length=1)

    @model_validator(mode="after")
    def has_unique_fields(self) -> "CompiledOutputModel":
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("output model field names must be unique")
        return self


_OUTPUT_FIELD_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "date": date,
    "datetime": datetime,
    "list[string]": list[str],
    "list[integer]": list[int],
    "list[number]": list[float],
    "list[boolean]": list[bool],
    "list[date]": list[date],
    "list[datetime]": list[datetime],
}


def build_compiled_output_model(spec: CompiledOutputModel) -> type[BaseModel]:
    """Create the runtime equivalent of a compiler-declared output model."""
    return create_model(
        spec.class_name,
        __config__=ConfigDict(extra="forbid"),
        **{
            field.name: (
                _OUTPUT_FIELD_TYPES[field.type],
                Field(description=field.description),
            )
            for field in spec.fields
        },
    )


class CompiledPromptResult(BaseModel):
    """Structured-output boundary for the board's compiler model."""

    model_config = ConfigDict(extra="forbid")

    definition: CompiledPromptDefinition = Field(
        description="The complete, validated Prompt Ninja 1.2 prompt definition."
    )
    output_model: CompiledOutputModel | None = Field(
        description=(
            "A concrete Pydantic output model for structured JSON, or null for "
            "String and BigInt outputs."
        )
    )

    @field_validator("definition", mode="before")
    @classmethod
    def discard_invalid_model_generated_defaults(cls, value: Any) -> Any:
        """Recover only from a bad fallback while preserving all other validation."""
        if not isinstance(value, dict) or not isinstance(value.get("variables"), list):
            return value
        definition = dict(value)
        # Test inputs are arbitrary key/value maps, which cannot be represented by
        # OpenAI strict structured outputs. The board attaches its tested fixture
        # after compilation instead.
        definition.pop("tests", None)
        variables: list[Any] = []
        for raw_variable in value["variables"]:
            if not isinstance(raw_variable, dict):
                variables.append(raw_variable)
                continue
            variable = dict(raw_variable)
            variable.pop("default", None)
            variables.append(variable)
        definition["variables"] = variables
        return definition


_VARIABLE_TOKEN_PATTERN = re.compile(
    r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)(\s*\|\s*[A-Za-z_][A-Za-z0-9_]*)?\s*\}\}"
)


def reconcile_variable_casing(definition: dict[str, Any]) -> dict[str, Any]:
    """Repair template references that differ from their declaration only by case.

    The compiler model occasionally declares a variable in one case (e.g.
    RELEASE_NOTES) while its own template references the same variable in
    another (release_notes). Prompt Ninja variable names are case-sensitive, so
    that bare mismatch fails validation even though the intent is unambiguous;
    conform the template to the declared name whenever they differ only by case.
    """
    variables = definition.get("variables")
    prompt = definition.get("prompt")
    if not isinstance(variables, list) or not isinstance(prompt, dict):
        return definition
    declared_names = [
        variable["name"]
        for variable in variables
        if isinstance(variable, dict) and isinstance(variable.get("name"), str)
    ]

    def fix_case(template: str) -> str:
        def replace(match: re.Match[str]) -> str:
            token_name = match.group(1)
            for declared in declared_names:
                if declared != token_name and declared.casefold() == token_name.casefold():
                    return match.group(0).replace(token_name, declared, 1)
            return match.group(0)

        return _VARIABLE_TOKEN_PATTERN.sub(replace, template)

    fixed_prompt = dict(prompt)
    for field in ("system", "user"):
        value = fixed_prompt.get(field)
        if isinstance(value, str):
            fixed_prompt[field] = fix_case(value)
    return {**definition, "prompt": fixed_prompt}
