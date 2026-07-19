"""Typed structured-output model for the board's prompt compiler."""

from pydantic import BaseModel, ConfigDict, Field

from .prompt_ninja import PromptFileSpec


class CompiledPromptFileSpec(PromptFileSpec):
    """Compiler output always uses the scalar TOML output declaration."""

    output: str = Field(
        pattern=(
            r"^(?:String|BigInt|"
            r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)$"
        )
    )


class CompiledPromptResult(BaseModel):
    """Structured-output boundary for the board's compiler model."""

    model_config = ConfigDict(extra="forbid")

    definition: CompiledPromptFileSpec
