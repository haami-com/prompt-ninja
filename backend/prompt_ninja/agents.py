from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from .model_config import load_environment
from .models import (
    AgentMessage,
    Brief,
    CouncilResult,
    GeneratedPromptTestRequest,
    PromptSpec,
)
from .core import (
    OpenRouterPromptClient,
    PreparedPrompt,
    PromptNinja,
    PromptRunHook,
    PromptRuntimeOptions,
)
from .prompt_catalog import PROMPTS
from .prompt_compiler import (
    CompiledOutputModel,
    CompiledPromptResult,
    build_compiled_output_model,
    reconcile_variable_casing,
)
from .prompt_testing import PromptTestHarness, fixture_values_for_prompt

load_environment()


def _editor_metadata(prompt: PromptNinja) -> dict[str, Any]:
    referenced = prompt.spec.template.referenced_variables
    variables = [
        {
            "name": variable.name,
            "type": variable.type,
            "required": variable.required,
            "present_in_template": variable.name in referenced,
        }
        for variable in prompt.spec.variables
    ]
    missing = [
        variable["name"]
        for variable in variables
        if variable["required"] and not variable["present_in_template"]
    ]
    return {
        "variables": variables,
        "valid": not missing,
        "missing_variables": missing,
    }


def default_agent_instructions() -> dict[str, Any]:
    """Expose TOML-defined defaults for the optional prompt editor."""
    creator_prompts = [PROMPTS.creator_1, PROMPTS.creator_2, PROMPTS.creator_3]
    judge_prompt = PROMPTS.judge
    return {
        "creators": [prompt.spec.template.system for prompt in creator_prompts],
        "judge": judge_prompt.spec.template.system,
        "metadata": {
            "creators": [_editor_metadata(prompt) for prompt in creator_prompts],
            "judge": _editor_metadata(judge_prompt),
        },
    }


class PromptCouncil:
    def __init__(
        self,
        creator_models: list[str] | None = None,
        judge_model: str | None = None,
        creator_prompts: list[str] | None = None,
        judge_prompt: str | None = None,
        creator_1_hooks: tuple[PromptRunHook, ...] = (),
        creator_2_hooks: tuple[PromptRunHook, ...] = (),
    ):
        self.prompt_client = None
        if os.getenv("OPENROUTER_API_KEY"):
            self.prompt_client = OpenRouterPromptClient()
        self.requirements_prompt_spec = PROMPTS.requirements
        self.creator_prompt_specs = [
            PROMPTS.creator_1,
            PROMPTS.creator_2,
            PROMPTS.creator_3,
        ]
        self.judge_prompt_spec = PROMPTS.judge
        self.compiler_prompt_spec = PROMPTS.prompt_compiler
        self.creator_models = creator_models or [
            prompt.spec.model.name for prompt in self.creator_prompt_specs
        ]
        self.judge_model = judge_model or self.judge_prompt_spec.spec.model.name
        overrides = creator_prompts or []
        self.creator_prompts = [
            overrides[index] if index < len(overrides) else "" for index in range(3)
        ]
        self.judge_prompt = judge_prompt or ""
        self.creator_1_hooks = creator_1_hooks
        self.creator_2_hooks = creator_2_hooks

    async def run_prompt(
        self,
        prompt: PromptNinja,
        runtime_variables: dict,
        *,
        model: str | None = None,
        system_override: str = "",
        output_model: type[BaseModel] | None = None,
        hooks: tuple[PromptRunHook, ...] = (),
    ) -> tuple[dict, PreparedPrompt]:
        """Run one TOML-defined stage with only its runtime values and model override."""
        prepared = prompt.prepare(runtime_variables, system_override=system_override)
        if not self.prompt_client:
            raise RuntimeError(
                "The TOML-defined prompt requires OPENROUTER_API_KEY to run."
            )
        runtime = PromptRuntimeOptions(model=model) if model else None
        effective_output_model = output_model or prompt.output_model
        execute_options = {
            "runtime": runtime,
            "output_model": effective_output_model,
        }
        if hooks:
            execute_options["hooks"] = hooks
        output = await self.prompt_client.execute(prompt, prepared, **execute_options)
        if isinstance(output, BaseModel):
            return (
                output.model_dump(
                    exclude_none=isinstance(output, CompiledPromptResult)
                ),
                prepared,
            )
        return output, prepared

    async def stream(self, brief: Brief) -> AsyncIterator[AgentMessage | CouncilResult]:
        agents: list[AgentMessage] = []
        prompt_trace = {"requirements": {}, "creators": [], "judge": {}, "compiler": {}}

        async def run_started(stage: str, agent: str, title: str):
            message = AgentMessage(
                stage=stage, agent=agent, status="started", title=title
            )
            agents.append(message)
            return message

        yield await run_started("requirements", "Requirements", "Requirements analyst")
        requirements, requirements_prompt = await self.run_prompt(
            self.requirements_prompt_spec,
            {"brief": brief.model_dump(), "council_context": {}},
            model=self.requirements_prompt_spec.spec.model.name,
        )
        prompt_trace["requirements"] = {
            "system_prompt": requirements_prompt.system,
            "input_context": requirements_prompt.user,
        }
        yield AgentMessage(
            stage="requirements",
            agent="Requirements",
            status="complete",
            title="Requirements analyst",
            summary="Mapped the requested outcome into an output contract.",
            payload=requirements,
        )
        for slot, model in enumerate(self.creator_models[:3], start=1):
            yield await run_started(
                f"creator_{slot}", f"Creator {slot}", "Prompt creator"
            )

        async def run_creator(slot: int, model: str):
            creator_context = {"requirements": requirements, "creator_slot": slot}
            creator_hooks = {
                1: self.creator_1_hooks,
                2: self.creator_2_hooks,
            }.get(slot, ())
            output, prompt = await self.run_prompt(
                self.creator_prompt_specs[slot - 1],
                {"brief": brief.model_dump(), "council_context": creator_context},
                model=model,
                system_override=self.creator_prompts[slot - 1],
                hooks=creator_hooks,
            )
            return slot, model, output, prompt

        creator_runs = await asyncio.gather(
            *[
                run_creator(slot, model)
                for slot, model in enumerate(self.creator_models[:3], start=1)
            ]
        )
        creator_outputs = []
        for slot, model, output, prompt in creator_runs:
            creator_outputs.append({"model": model, **output})
            prompt_trace["creators"].append(
                {
                    "slot": slot,
                    "model": model,
                    "system_prompt": prompt.system,
                    "input_context": prompt.user,
                }
            )
            yield AgentMessage(
                stage=f"creator_{slot}",
                agent=f"Creator {slot}",
                status="complete",
                title=f"Prompt creator · {model}",
                summary=output.get(
                    "rationale", "Created a TOML-defined prompt proposal."
                ),
                payload=output,
            )

        yield await run_started("synthesis", "Judge", "Final judge")
        judge_context = {"requirements": requirements, "creators": creator_outputs}
        judged, judge_prompt = await self.run_prompt(
            self.judge_prompt_spec,
            {"brief": brief.model_dump(), "council_context": judge_context},
            model=self.judge_model,
            system_override=self.judge_prompt,
        )
        prompt_trace["judge"] = {
            "model": self.judge_model,
            "system_prompt": judge_prompt.system,
            "input_context": judge_prompt.user,
        }
        yield AgentMessage(
            stage="synthesis",
            agent="Judge",
            status="complete",
            title=f"Final judge · {self.judge_model}",
            summary=judged.get(
                "decision_summary",
                "Compared creator proposals and selected the strongest result.",
            ),
            payload={
                "final_prompt": judged["final_prompt"],
                "decision_summary": judged.get("decision_summary", ""),
                "model": self.judge_model,
            },
        )
        yield await run_started(
            "validation", "Prompt validation", "Self-test and compiler"
        )
        candidate_test = await PromptTestHarness(prompt_client=self.prompt_client).run(
            GeneratedPromptTestRequest(
                final_prompt=judged["final_prompt"],
                goal=brief.outcome,
                context=brief.context,
                expected_output=brief.expected_output,
                model=self.judge_model,
            )
        )
        compiled, compiler_prompt = await self.run_prompt(
            self.compiler_prompt_spec,
            {
                "goal": brief.outcome,
                "model": self.judge_model,
                "requirements": requirements,
                "candidate_prompt": judged["final_prompt"],
                "test_result": candidate_test.model_dump(),
            },
        )
        compiled_prompt = PromptNinja(
            reconcile_variable_casing(compiled["definition"]),
            source="<board compiler>",
        )
        expected_output: str | dict = candidate_test.expected_output
        if compiled.get("output_model"):
            output_model = build_compiled_output_model(
                CompiledOutputModel.model_validate(compiled["output_model"])
            )
            expected_output = output_model.model_validate_json(
                candidate_test.actual_output
            ).model_dump(mode="json")
        if not compiled_prompt.tests:
            compiled_definition = compiled_prompt.spec.model_dump(
                by_alias=True, exclude_none=True
            )
            compiled_definition["tests"] = [
                {
                    "name": "Generated self-test fixture",
                    "variable": fixture_values_for_prompt(
                        compiled_prompt, candidate_test.input
                    ),
                    "expected_output": expected_output,
                }
            ]
            compiled_prompt = PromptNinja(
                compiled_definition, source="<board compiler>"
            )
        prompt_test = await PromptTestHarness(prompt_client=self.prompt_client).run(
            GeneratedPromptTestRequest(
                final_prompt=compiled_prompt.spec.template.system,
                goal=brief.outcome,
                context=brief.context,
                expected_output=brief.expected_output,
                model=self.judge_model,
                definition=compiled_prompt.spec.model_dump(
                    by_alias=True, exclude_none=True
                ),
            ),
            reuse_fixture={
                "input": candidate_test.input,
                "expected_output": candidate_test.expected_output,
            },
        )
        prompt_trace["compiler"] = {
            "model": self.compiler_prompt_spec.spec.model.name,
            "system_prompt": compiler_prompt.system,
            "input_context": compiler_prompt.user,
        }
        spec = PromptSpec.model_validate(
            {
                "goal": requirements.get("goal", brief.outcome),
                "inputs": requirements.get("inputs", []),
                "output_contract": requirements.get(
                    "output_contract", brief.expected_output
                ),
                "constraints": requirements.get("constraints", []),
                "assumptions": requirements.get("assumptions", []),
            }
        )
        yield AgentMessage(
            stage="validation",
            agent="Prompt validation",
            status="complete",
            title="Self-test and compiler",
            summary="Validated the compiled Prompt Ninja definition.",
            payload=prompt_test.model_dump(),
        )
        yield CouncilResult(
            final_prompt=compiled_prompt.spec.template.system,
            prompt_spec=spec,
            prompt_definition=compiled_prompt.spec.model_dump(
                by_alias=True, exclude_none=True
            ),
            output_model=compiled.get("output_model") or {},
            prompt_test=prompt_test.model_dump(),
            agents=agents,
            creators=creator_outputs,
            judge_model=self.judge_model,
            judge_summary=judged.get("decision_summary", ""),
            prompt_trace=prompt_trace,
        )
