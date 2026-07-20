"""Typed structured-output model for the board's prompt compiler."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .prompt_ninja import PromptFileSpec, VariableSpec


class CompiledPromptResult(BaseModel):
    """Structured-output boundary for the board's compiler model."""

    model_config = ConfigDict(extra="forbid")

    definition: PromptFileSpec = Field(
        description="The complete, validated Prompt Ninja 1.2 prompt definition."
    )

    @field_validator("definition", mode="before")
    @classmethod
    def discard_invalid_model_generated_defaults(cls, value: Any) -> Any:
        """Recover only from a bad fallback while preserving all other validation."""
        if not isinstance(value, dict) or not isinstance(value.get("variables"), list):
            return value
        definition = dict(value)
        variables: list[Any] = []
        for raw_variable in value["variables"]:
            if not isinstance(raw_variable, dict) or "default" not in raw_variable:
                variables.append(raw_variable)
                continue
            variable = dict(raw_variable)
            without_default = dict(variable)
            without_default.pop("default")
            try:
                VariableSpec.model_validate(variable)
            except ValidationError:
                try:
                    VariableSpec.model_validate(without_default)
                except ValidationError:
                    variables.append(variable)
                else:
                    variables.append(without_default)
            else:
                variables.append(variable)
        definition["variables"] = variables
        return definition
