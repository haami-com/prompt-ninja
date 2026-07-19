from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dotenv import load_dotenv

from .models import AgentMessage, Brief, CouncilResult, PromptSpec, RequirementsResult
from .prompt_ninja import OpenAIPromptClient, PreparedPrompt, PromptNinja, PromptRuntimeOptions

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

PROMPTS_DIRECTORY = Path(__file__).resolve().parents[1] / "prompts"
CREATOR_PROMPT_FILES = tuple(PROMPTS_DIRECTORY / f"creator-{slot}.prompt.toml" for slot in range(1, 4))
JUDGE_PROMPT_FILE = PROMPTS_DIRECTORY / "judge.prompt.toml"
REQUIREMENTS_PROMPT_FILE = PROMPTS_DIRECTORY / "requirements.prompt.toml"


def _load_prompt(path: Path) -> PromptNinja:
    return PromptNinja.from_file(path)


def default_agent_instructions() -> dict[str, list[str] | str]:
    """Expose TOML-defined defaults for the optional prompt editor."""
    creator_instructions = [_load_prompt(path).spec.template.system for path in CREATOR_PROMPT_FILES]
    return {"creators": creator_instructions, "judge": _load_prompt(JUDGE_PROMPT_FILE).spec.template.system}


class PromptCouncil:
    def __init__(self, creator_models: list[str] | None = None, judge_model: str | None = None, creator_prompts: list[str] | None = None, judge_prompt: str | None = None):
        self.prompt_client = None
        if os.getenv("OPENAI_API_KEY"):
            self.prompt_client = OpenAIPromptClient()
        self.creator_models = creator_models or ["gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.5"]
        self.judge_model = judge_model or "gpt-5.6-terra"
        self.requirements_prompt_spec = _load_prompt(REQUIREMENTS_PROMPT_FILE)
        self.creator_prompt_specs = [_load_prompt(path) for path in CREATOR_PROMPT_FILES]
        self.judge_prompt_spec = _load_prompt(JUDGE_PROMPT_FILE)
        overrides = creator_prompts or []
        self.creator_prompts = [overrides[index] if index < len(overrides) else "" for index in range(3)]
        self.judge_prompt = judge_prompt or ""

    async def run_prompt(
        self,
        prompt: PromptNinja,
        runtime_variables: dict,
        *,
        model: str | None = None,
        system_override: str = "",
        output_model: type[BaseModel] | None = None,
    ) -> tuple[dict, PreparedPrompt]:
        """Run one TOML-defined stage with only its runtime values and model override."""
        prepared = prompt.prepare(runtime_variables, system_override=system_override)
        if not self.prompt_client:
            raise RuntimeError("The TOML-defined prompt requires OPENAI_API_KEY to run.")
        runtime = PromptRuntimeOptions(model=model) if model else None
        output = await self.prompt_client.execute(prompt, prepared, runtime=runtime, output_model=output_model)
        return output.model_dump() if isinstance(output, BaseModel) else output, prepared

    async def stream(self, brief: Brief) -> AsyncIterator[AgentMessage | CouncilResult]:
        agents: list[AgentMessage] = []
        prompt_trace = {"requirements": {}, "creators": [], "judge": {}}

        async def run_started(stage: str, agent: str, title: str):
            message = AgentMessage(stage=stage, agent=agent, status="started", title=title)
            agents.append(message)
            return message

        yield await run_started("requirements", "Requirements", "Requirements analyst")
        requirements, requirements_prompt = await self.run_prompt(
            self.requirements_prompt_spec,
            {"brief": brief.model_dump(), "council_context": {}},
            output_model=RequirementsResult,
        )
        prompt_trace["requirements"] = {"system_prompt": requirements_prompt.system, "input_context": requirements_prompt.user}
        yield AgentMessage(stage="requirements", agent="Requirements", status="complete", title="Requirements analyst", summary="Mapped the requested outcome into an output contract.", payload=requirements)
        for slot, model in enumerate(self.creator_models[:3], start=1):
            yield await run_started(f"creator_{slot}", f"Creator {slot}", "Prompt creator")

        async def run_creator(slot: int, model: str):
            creator_context = {"requirements": requirements, "creator_slot": slot}
            output, prompt = await self.run_prompt(
                self.creator_prompt_specs[slot - 1],
                {"brief": brief.model_dump(), "council_context": creator_context},
                model=model,
                system_override=self.creator_prompts[slot - 1],
            )
            return slot, model, output, prompt

        creator_runs = await asyncio.gather(*[
            run_creator(slot, model) for slot, model in enumerate(self.creator_models[:3], start=1)
        ])
        creator_outputs = []
        for slot, model, output, prompt in creator_runs:
            creator_outputs.append(output)
            prompt_trace["creators"].append({"slot": slot, "model": model, "system_prompt": prompt.system, "input_context": prompt.user})
            yield AgentMessage(stage=f"creator_{slot}", agent=f"Creator {slot}", status="complete", title=f"Prompt creator · {model}", summary=output.get("rationale", "Created a TOML-defined prompt proposal."), payload=output)

        yield await run_started("synthesis", "Judge", "Final judge")
        judge_context = {"requirements": requirements, "creators": creator_outputs}
        judged, judge_prompt = await self.run_prompt(
            self.judge_prompt_spec,
            {"brief": brief.model_dump(), "council_context": judge_context},
            model=self.judge_model,
            system_override=self.judge_prompt,
        )
        prompt_trace["judge"] = {"model": self.judge_model, "system_prompt": judge_prompt.system, "input_context": judge_prompt.user}
        spec = PromptSpec.model_validate({
            "goal": requirements.get("goal", brief.outcome),
            "inputs": requirements.get("inputs", []),
            "output_contract": requirements.get("output_contract", brief.expected_output),
            "constraints": requirements.get("constraints", []),
            "assumptions": requirements.get("assumptions", []),
        })
        yield AgentMessage(stage="synthesis", agent="Judge", status="complete", title=f"Final judge · {self.judge_model}", summary=judged.get("decision_summary", "Compared creator proposals and selected the strongest result."), payload={"decision_summary": judged.get("decision_summary", ""), "model": self.judge_model})
        yield CouncilResult(final_prompt=judged["final_prompt"], prompt_spec=spec, agents=agents, creators=creator_outputs, judge_model=self.judge_model, judge_summary=judged.get("decision_summary", ""), prompt_trace=prompt_trace)
