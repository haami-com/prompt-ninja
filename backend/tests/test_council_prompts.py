import asyncio

import pytest

from app.agents import CREATOR_PROMPT_FILES, JUDGE_PROMPT_FILE, PromptCouncil, default_agent_instructions
from app.main import app
from app.models import Brief
from app.prompt_ninja import PromptNinja
from fastapi.testclient import TestClient


def test_council_prompt_files_render_with_the_runtime_context():
    brief = Brief(outcome="Create an accurate summary")
    for path in (*CREATOR_PROMPT_FILES, JUDGE_PROMPT_FILE):
        prompt = PromptNinja.from_file(path)
        prepared = prompt.prepare({"brief": brief.model_dump(), "council_context": {"requirements": {}}})
        assert prepared.system
        assert "Create an accurate summary" in prepared.user


def test_council_uses_toml_defaults_and_allows_an_override():
    class FakePromptClient:
        async def execute(self, _, prepared, runtime=None, output_model=None):
            return {"draft": "A proposal", "rationale": "A rationale"}

    council = PromptCouncil()
    council.prompt_client = FakePromptClient()
    brief = Brief(outcome="Create an accurate summary")
    _, default = asyncio.run(council.run_prompt(
        council.creator_prompt_specs[0],
        {"brief": brief.model_dump(), "council_context": {}},
    ))
    _, override = asyncio.run(council.run_prompt(
        council.creator_prompt_specs[0],
        {"brief": brief.model_dump(), "council_context": {}},
        system_override="Custom creator instruction",
    ))

    assert "Creator 1" in default.system
    assert override.system == "Custom creator instruction"
    assert default_agent_instructions()["judge"] == council.judge_prompt_spec.spec.template.system


def test_toml_judge_requires_provider_access(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    council = PromptCouncil()
    brief = Brief(outcome="Create an accurate summary")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        asyncio.run(council.run_prompt(
            council.judge_prompt_spec,
            {"brief": brief.model_dump(), "council_context": {}},
        ))


def test_prompt_defaults_endpoint_exposes_toml_defined_instructions():
    response = TestClient(app).get("/api/prompts")
    assert response.status_code == 200
    assert "Creator 2" in response.json()["creators"][1]
