from typing import Literal

from pydantic import BaseModel, Field


class Brief(BaseModel):
    outcome: str = Field(min_length=8, max_length=4000)
    context: str = Field(default="", max_length=2000)
    source_text: str = Field(default="", max_length=30000)
    expected_output: str = Field(default="", max_length=12000)
    constraints: str = Field(default="", max_length=6000)


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
    judge_model: str = "gpt-5.6-terra"


class GeneratedPromptTestResult(BaseModel):
    model: str
    input: str
    expected_output: str
    expected_schema: dict
    schema_valid: bool
    actual_output: str
    score: float = Field(ge=0, le=1)
    passed: bool
    rationale: str


class PromptExportRequest(BaseModel):
    final_prompt: str = Field(min_length=1, max_length=24000)
    goal: str = Field(min_length=8, max_length=4000)
    model: str = "gpt-5.6-terra"
