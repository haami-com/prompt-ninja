import asyncio

import pytest

from app.agents import (
    CREATOR_PROMPT_FILES,
    JUDGE_PROMPT_FILE,
    REQUIREMENTS_PROMPT_FILE,
    PromptCouncil,
    default_agent_instructions,
)
from app.main import app
from app.models import Brief
from app.prompt_catalog import PROMPTS
from app.prompt_ninja import PromptNinja
from fastapi.testclient import TestClient


def test_council_prompt_files_render_with_the_runtime_context():
    brief = Brief(outcome="Create an accurate summary")
    for path in (*CREATOR_PROMPT_FILES, JUDGE_PROMPT_FILE):
        prompt = PromptNinja.from_file(path)
        prepared = prompt.prepare(
            {"brief": brief.model_dump(), "council_context": {"requirements": {}}}
        )
        assert prepared.system
        assert "Create an accurate summary" in prepared.user


def test_requirements_prompt_injects_array_valued_output_fields():
    prompt = PromptNinja.from_file(REQUIREMENTS_PROMPT_FILE)
    prepared = prompt.prepare({"brief": {}, "council_context": {}})

    properties = prompt.output_json_schema["properties"]
    assert properties["inputs"]["type"] == "array"
    assert properties["constraints"]["type"] == "array"
    assert "Output contract" in prepared.system
    assert '"inputs"' in prepared.system


def test_council_uses_toml_defaults_and_allows_an_override():
    class FakePromptClient:
        async def execute(self, _, prepared, runtime=None, output_model=None):
            return {"draft": "A proposal", "rationale": "A rationale"}

    council = PromptCouncil()
    assert council.requirements_prompt_spec is PROMPTS.requirements
    assert council.creator_prompt_specs == [
        PROMPTS.creator_1,
        PROMPTS.creator_2,
        PROMPTS.creator_3,
    ]
    assert council.judge_prompt_spec is PROMPTS.judge
    assert council.compiler_prompt_spec is PROMPTS.prompt_compiler
    council.prompt_client = FakePromptClient()
    brief = Brief(outcome="Create an accurate summary")
    _, default = asyncio.run(
        council.run_prompt(
            council.creator_prompt_specs[0],
            {"brief": brief.model_dump(), "council_context": {}},
        )
    )
    _, override = asyncio.run(
        council.run_prompt(
            council.creator_prompt_specs[0],
            {"brief": brief.model_dump(), "council_context": {}},
            system_override="Custom creator instruction",
        )
    )

    assert "Creator 1" in default.system
    assert override.system.startswith("Custom creator instruction\n\nOutput contract")
    assert (
        default_agent_instructions()["judge"]
        == council.judge_prompt_spec.spec.template.system
    )
    assert council.creator_models == [
        prompt.spec.model.name for prompt in council.creator_prompt_specs
    ]
    assert council.judge_model == council.judge_prompt_spec.spec.model.name


def test_prompt_editor_metadata_lists_and_validates_runtime_variables():
    defaults = default_agent_instructions()

    for metadata in [*defaults["metadata"]["creators"], defaults["metadata"]["judge"]]:
        assert metadata["valid"]
        assert metadata["missing_variables"] == []
        assert {variable["name"] for variable in metadata["variables"]} == {
            "brief",
            "council_context",
        }
        assert all(variable["required"] for variable in metadata["variables"])
        assert all(
            variable["present_in_template"] for variable in metadata["variables"]
        )


def test_council_passes_a_selected_model_as_a_prompt_runtime_override():
    runtimes = []

    class FakePromptClient:
        async def execute(self, _, prepared, runtime=None, output_model=None):
            runtimes.append(runtime)
            return {"draft": "A proposal", "rationale": "A rationale"}

    council = PromptCouncil()
    council.prompt_client = FakePromptClient()
    brief = Brief(outcome="Create an accurate summary")
    asyncio.run(
        council.run_prompt(
            council.creator_prompt_specs[0],
            {"brief": brief.model_dump(), "council_context": {}},
            model="gpt-5.6-terra",
        )
    )

    assert runtimes[0].model == "gpt-5.6-terra"


def test_council_compiles_a_validated_definition_after_self_test_evidence():
    calls = []

    class FakePromptClient:
        async def execute(self, _, prepared, runtime=None, output_model=None):
            calls.append(prepared.name)
            if prepared.name == "requirements":
                return output_model.model_validate(
                    {
                        "goal": "Create an accurate summary",
                        "inputs": [],
                        "output_contract": "text",
                        "constraints": [],
                        "assumptions": [],
                        "risks": [],
                    }
                )
            if prepared.name.startswith("creator_"):
                return {"draft": "A proposal", "rationale": "A rationale"}
            if prepared.name == "judge":
                return {
                    "final_prompt": "Summarize the input.",
                    "decision_summary": "Combined proposals.",
                }
            if prepared.name == "test_case_generator":
                return {
                    "input": "A project update.",
                    "expected_output": "A concise summary.",
                    "output_format": "text",
                }
            if prepared.name == "generated_prompt_under_test":
                return "A concise summary."
            if prepared.name == "project-summary":
                return "A concise summary."
            if prepared.name == "test_judge":
                return {"score": 0.98, "rationale": "Matches the expectation."}
            if prepared.name == "prompt_compiler":
                return {
                    "definition": {
                        "metadata": {
                            "spec_version": "1.2",
                            "name": "project-summary",
                            "description": "Summarizes project updates.",
                            "used_by": ["backend/tests/test_council_prompts.py"],
                            "version": "1.0.0",
                            "output": "String",
                        },
                        "llm_model": {
                            "provider": "openrouter",
                            "name": "google/gemini-2.5-flash",
                        },
                        "prompt": {
                            "system": "Create a concise summary from the input.",
                            "user": "{{input}}",
                        },
                        "variables": [
                            {
                                "name": "input",
                                "type": "string",
                                "description": "The project update to summarize.",
                                "required": True,
                            }
                        ],
                    }
                }
            raise AssertionError("Unexpected prompt: %s" % prepared.name)

    async def collect_items(council):
        return [
            item
            async for item in council.stream(
                Brief(outcome="Create an accurate summary")
            )
        ]

    council = PromptCouncil(
        creator_models=["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-sol"]
    )
    council.prompt_client = FakePromptClient()
    items = asyncio.run(collect_items(council))
    result = items[-1]
    stage_events = [
        (item.stage, item.status) for item in items if hasattr(item, "stage")
    ]

    assert [creator["model"] for creator in result.creators] == [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
    ]
    assert result.prompt_definition["prompt"]["user"] == "{{input}}"
    assert result.prompt_test["passed"]
    assert result.prompt_definition["tests"] == [
        {
            "name": "Generated self-test fixture",
            "variable": {"input": "A project update."},
            "expected_output": "A concise summary.",
        }
    ]
    assert stage_events.index(("synthesis", "complete")) < stage_events.index(
        ("validation", "started")
    )
    assert (
        calls.index("test_case_generator")
        < calls.index("generated_prompt_under_test")
        < calls.index("test_judge")
        < calls.index("prompt_compiler")
    )
    assert calls[-3:] == ["test_case_generator", "project-summary", "test_judge"]


def test_toml_judge_requires_provider_access(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    council = PromptCouncil()
    brief = Brief(outcome="Create an accurate summary")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        asyncio.run(
            council.run_prompt(
                council.judge_prompt_spec,
                {"brief": brief.model_dump(), "council_context": {}},
            )
        )


def test_prompt_defaults_endpoint_exposes_toml_defined_instructions():
    response = TestClient(app).get("/api/prompts")
    assert response.status_code == 200
    assert "Creator 2" in response.json()["creators"][1]
