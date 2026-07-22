import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .model_config import DEFAULT_MODEL


class Brief(BaseModel):
    outcome: str = Field(min_length=8, max_length=4000)
    context: str = Field(default="", max_length=2000)
    source_text: str = Field(default="", max_length=30000)
    expected_output: str = Field(default="", max_length=12000)
    constraints: str = Field(default="", max_length=6000)


class BriefEnhancementResult(BaseModel):
    enhanced_request: str = Field(
        min_length=8,
        max_length=6000,
        description="A polished standalone version of the user's request.",
    )
    outcome: str = Field(
        min_length=8,
        max_length=4000,
        description="The primary result the prompt should achieve.",
    )
    context: str = Field(
        default="",
        max_length=2000,
        description="Where or how the prompt will be used, or an empty string.",
    )
    expected_output: str = Field(
        default="",
        max_length=12000,
        description="The desired response shape and quality, or an empty string.",
    )
    constraints: str = Field(
        default="",
        max_length=6000,
        description="All explicit guardrails, or an empty string.",
    )
    file_references: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Referenced labels such as File #1; empty when none are used.",
    )


class CreatorDraftResult(BaseModel):
    draft: str = Field(description="The proposed production prompt.")
    rationale: str = Field(description="A concise explanation of the design choices.")


class JudgeSynthesisResult(BaseModel):
    final_prompt: str = Field(
        description="The final production-ready prompt synthesized from the proposals."
    )
    decision_summary: str = Field(
        description="A concise rationale without hidden chain-of-thought."
    )


class GeneratedTestCaseResult(BaseModel):
    input: str = Field(
        description="Representative source material or user input for the generated prompt."
    )
    expected_output: str = Field(
        description="Natural-language criteria for a correct result, not an exact answer."
    )
    output_format: Literal["text", "json_object", "json_array"] = Field(
        description=(
            "Whether the generated prompt should produce plain text, a top-level "
            "JSON object, or a top-level JSON array."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_output_format(cls, data: Any) -> Any:
        """Repair the model's occasional bare "json" into a concrete category.

        Despite an explicit prompt instruction against it, some models still
        default to the generic legacy term instead of choosing json_object vs
        json_array. Infer the right one from the fixture's own expected_output
        text rather than fail the whole run over a single stale keyword.
        """
        if not isinstance(data, dict) or data.get("output_format") != "json":
            return data
        expected_output = data.get("expected_output", "")
        is_list_shaped = isinstance(expected_output, str) and bool(
            re.search(r"\b(array|list)\b", expected_output, re.IGNORECASE)
        )
        return {
            **data,
            "output_format": "json_array" if is_list_shaped else "json_object",
        }


class GreetingResult(BaseModel):
    result: str = Field(description="The concise greeting.")


class Person(BaseModel):
    """Example domain model available to prompt variable declarations and tests."""

    name: str
    role: str = ""


class AgentMessage(BaseModel):
    stage: str
    agent: str
    status: Literal["started", "complete", "error"]
    title: str
    summary: str = ""
    payload: dict = Field(default_factory=dict)


class PromptSpec(BaseModel):
    goal: str
    inputs: list[str]
    output_contract: str
    constraints: list[str]
    assumptions: list[str]


class RequirementsResult(BaseModel):
    goal: str = Field(description="The user's intended outcome.")
    inputs: list[str] = Field(description="Information required to complete the task.")
    output_contract: str = Field(description="The desired response shape and quality.")
    constraints: list[str] = Field(
        description="Explicit guardrails that must be followed."
    )
    assumptions: list[str] = Field(description="Assumptions or ambiguities to surface.")
    risks: list[str] = Field(
        description="Risks or missing details that could affect the result."
    )


class CouncilResult(BaseModel):
    final_prompt: str
    prompt_spec: PromptSpec
    prompt_definition: dict = Field(default_factory=dict)
    output_model: dict = Field(default_factory=dict)
    prompt_test: dict = Field(default_factory=dict)
    agents: list[AgentMessage]
    creators: list[dict] = Field(default_factory=list)
    judge_model: str = ""
    judge_summary: str = ""
    prompt_trace: dict = Field(default_factory=dict)


class GeneratedPromptTestRequest(BaseModel):
    final_prompt: str = Field(min_length=1, max_length=24000)
    goal: str = Field(min_length=8, max_length=4000)
    context: str = Field(default="", max_length=2000)
    expected_output: str = Field(default="", max_length=12000)
    model: str
    judge_model: str = DEFAULT_MODEL
    definition: dict | None = None


class GeneratedPromptTestResult(BaseModel):
    model: str
    input: str
    expected_output: str
    actual_output: str
    score: float = Field(ge=0, le=1)
    passed: bool
    rationale: str


class PromptExportRequest(BaseModel):
    final_prompt: str = Field(default="", max_length=24000)
    goal: str = Field(min_length=8, max_length=4000)
    model: str = DEFAULT_MODEL
    definition: dict | None = None


class PromptArtifactTestRequest(BaseModel):
    definition: dict
    judge_model: str = DEFAULT_MODEL


class PromptArtifactUpdateRequest(PromptArtifactTestRequest):
    feedback: str = Field(min_length=3, max_length=12000)
    model: str = DEFAULT_MODEL


class PromptArtifactReport(BaseModel):
    prompt_name: str
    passed: bool
    results: list[dict]


class PromptArtifactUpdateResult(BaseModel):
    definition: dict
    report: PromptArtifactReport
