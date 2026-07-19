from typing import Literal

from pydantic import BaseModel, Field

from .model_config import DEFAULT_MODEL


class Brief(BaseModel):
    outcome: str = Field(min_length=8, max_length=4000)
    context: str = Field(default="", max_length=2000)
    source_text: str = Field(default="", max_length=30000)
    expected_output: str = Field(default="", max_length=12000)
    constraints: str = Field(default="", max_length=6000)


class BriefEnhancementResult(BaseModel):
    enhanced_request: str = Field(min_length=8, max_length=6000)
    outcome: str = Field(min_length=8, max_length=4000)
    context: str = Field(default="", max_length=2000)
    expected_output: str = Field(default="", max_length=12000)
    constraints: str = Field(default="", max_length=6000)
    file_references: list[str] = Field(default_factory=list, max_length=5)


class CreatorDraftResult(BaseModel):
    draft: str
    rationale: str


class JudgeSynthesisResult(BaseModel):
    final_prompt: str
    decision_summary: str


class GeneratedTestCaseResult(BaseModel):
    input: str
    expected_output: str
    output_format: Literal["text", "json"]


class GreetingResult(BaseModel):
    result: str


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
    goal: str
    inputs: list[str]
    output_contract: str
    constraints: list[str]
    assumptions: list[str]
    risks: list[str]


class CouncilResult(BaseModel):
    final_prompt: str
    prompt_spec: PromptSpec
    prompt_definition: dict = Field(default_factory=dict)
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
