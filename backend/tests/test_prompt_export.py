from pathlib import Path

from prompt_ninja.main import app
from prompt_ninja.prompt_catalog import PROMPTS_DIRECTORY
from prompt_ninja.prompt_export import export_prompt_toml
from prompt_ninja.models import PromptExportRequest
from prompt_ninja import PromptNinja
from prompt_ninja import PromptTestReport, PromptTestResult
from fastapi.testclient import TestClient


def test_models_endpoint_exposes_openrouter_pricing(monkeypatch):
    async def fake_available_models():
        return (
            {
                "id": "google/gemini-2.5-flash",
                "name": "Gemini 2.5 Flash",
                "context_length": 1_000_000,
                "pricing": {"prompt": "0.0000003", "completion": "0.0000025"},
                "supported_parameters": ["structured_outputs"],
            },
        )

    monkeypatch.setattr("prompt_ninja.main.available_models", fake_available_models)
    response = TestClient(app).get("/api/models")

    assert response.status_code == 200
    model = response.json()["models"][0]
    assert model["id"] == "google/gemini-2.5-flash"
    assert model["pricing"]["prompt"] == "0.0000003"
    assert response.json()["creator_models"] == [
        "openai/gpt-5.6-luna",
        "deepseek/deepseek-v4-flash",
        "google/gemini-3.5-flash",
    ]
    assert response.json()["judge_model"] == "openai/gpt-5.6-terra"


def test_evaluations_endpoint_exposes_creator_1_sampling_state():
    response = TestClient(app).get("/api/evaluations")

    assert response.status_code == 200
    assert response.json()["hook"] == "creator_1"
    assert response.json()["sample_every"] == 1
    assert response.json()["next_sample_in"] == 1
    assert isinstance(response.json()["evaluations"], list)


def test_hooks_endpoint_exposes_quality_and_usage_activity():
    response = TestClient(app).get("/api/hooks")

    assert response.status_code == 200
    assert response.json()["quality"]["hook"] == "creator_1"
    assert response.json()["quality"]["cadence"] == "every_successful_response"
    assert isinstance(response.json()["quality"]["evaluations"], list)
    assert response.json()["usage"]["hook"] == "creator_2"
    assert response.json()["usage"]["cadence"] == "every_successful_response"
    assert isinstance(response.json()["usage"]["records"], list)
    assert response.json()["usage"]["summary"]["total_tokens"] >= 0


def test_export_creates_a_valid_prompt_toml_file(tmp_path):
    request = PromptExportRequest(
        final_prompt="Summarize the user's text in three bullets.",
        goal="Summarize legal documents into plain English",
        model="gpt-5.6-terra",
    )
    content = export_prompt_toml(request)
    path = tmp_path / "export-test.prompt.toml"
    path.write_text(content)
    prompt = PromptNinja.from_file(path)
    assert prompt.name == "summarize-legal-documents-into-plain-english"
    assert prompt.spec.metadata.used_by == []
    assert (
        prompt.prepare({"input": "Example legal clause"}).user == "Example legal clause"
    )


def test_export_declares_placeholders_from_the_generated_prompt(tmp_path):
    content = export_prompt_toml(
        PromptExportRequest(
            final_prompt="Summarize {{meeting_notes}} for {{audience}}.",
            goal="Summarize meeting notes",
            model="gpt-5.6-sol",
        )
    )
    path = tmp_path / "placeholder-export.prompt.toml"
    path.write_text(content)
    prompt = PromptNinja.from_file(path)

    prepared = prompt.prepare(
        {
            "input": "Ignored in the system template",
            "meeting_notes": "Launch planning notes",
            "audience": "executives",
        }
    )

    assert prepared.system.startswith(
        "Summarize Launch planning notes for executives.\n\n"
    )
    assert "metadata.output = 'String'" in prepared.system


def test_export_serializes_a_supplied_prompt_ninja_definition_without_losing_its_schema_or_tests(
    tmp_path,
):
    source = PROMPTS_DIRECTORY / "greeting.prompt.toml"
    original = PromptNinja.from_file(source)
    content = export_prompt_toml(
        PromptExportRequest(
            goal="Export an existing prompt specification",
            definition=original.spec.model_dump(by_alias=True, exclude_none=True),
        )
    )
    path = tmp_path / "canonical-export.prompt.toml"
    path.write_text(content)
    restored = PromptNinja.from_file(path)

    assert restored.spec.model_dump(by_alias=True) == original.spec.model_dump(
        by_alias=True
    )


def test_export_endpoint_downloads_toml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    response = TestClient(app).post(
        "/api/export-prompt",
        json={
            "final_prompt": "Summarize the user's text in three bullets.",
            "goal": "Summarize legal documents into plain English",
            "model": "google/gemini-2.5-flash",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/toml")
    assert "attachment" in response.headers["content-disposition"]
    assert "spec_version" in response.text


def test_export_endpoint_downloads_the_canonical_definition_losslessly(tmp_path):
    source = PROMPTS_DIRECTORY / "greeting.prompt.toml"
    original = PromptNinja.from_file(source)
    response = TestClient(app).post(
        "/api/export-prompt",
        json={
            "goal": "Export the canonical greeting prompt",
            "definition": original.spec.model_dump(by_alias=True, exclude_none=True),
        },
    )

    assert response.status_code == 200
    path = tmp_path / "endpoint-export.prompt.toml"
    path.write_text(response.text)
    restored = PromptNinja.from_file(path)
    assert restored.spec.model_dump(by_alias=True) == original.spec.model_dump(
        by_alias=True
    )


def test_artifact_test_endpoint_returns_complete_diagnostics(monkeypatch):
    source = PROMPTS_DIRECTORY / "greeting.prompt.toml"
    prompt = PromptNinja.from_file(source)

    async def fake_model(_model):
        return True

    async def fake_run(candidate, judge_model):
        assert candidate.name == prompt.name
        assert judge_model == "test/judge"
        return PromptTestReport(
            prompt_name=candidate.name,
            results=(PromptTestResult(
                name="greets by name",
                passed=False,
                input={"name": "Ada"},
                expected="A warm greeting",
                actual="Hello.",
                score=0.4,
                rationale="The name is missing.",
                prompt_suggestion="Require the supplied name.",
                test_suggestion="Clarify what warm means.",
            ),),
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("prompt_ninja.main.is_available_model", fake_model)
    monkeypatch.setattr("prompt_ninja.main.run_prompt_artifact_tests", fake_run)
    response = TestClient(app).post(
        "/api/test-artifact",
        json={
            "definition": prompt.spec.model_dump(by_alias=True, exclude_none=True),
            "judge_model": "test/judge",
        },
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["actual"] == "Hello."
    assert result["rationale"] == "The name is missing."
    assert result["prompt_suggestion"] == "Require the supplied name."
    assert result["test_suggestion"] == "Clarify what warm means."


def test_artifact_update_endpoint_returns_definition_and_rerun(monkeypatch):
    source = PROMPTS_DIRECTORY / "greeting.prompt.toml"
    prompt = PromptNinja.from_file(source)
    updated = PromptNinja(
        {
            **prompt.spec.model_dump(by_alias=True, exclude_none=True),
            "prompt": {
                **prompt.spec.prompt.model_dump(),
                "system": prompt.spec.prompt.system + " Always use the supplied name.",
            },
        }
    )

    async def fake_model(_model):
        return True

    async def fake_update(original, feedback, model):
        assert original.tests == updated.tests
        assert feedback == "Use the supplied name."
        assert model == "test/model"
        return updated

    async def fake_run(candidate, judge_model):
        return PromptTestReport(
            prompt_name=candidate.name,
            results=(PromptTestResult(
                name="greets by name", passed=True, expected="Greeting", score=1.0
            ),),
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("prompt_ninja.main.is_available_model", fake_model)
    monkeypatch.setattr("prompt_ninja.main.update_prompt_artifact", fake_update)
    monkeypatch.setattr("prompt_ninja.main.run_prompt_artifact_tests", fake_run)
    response = TestClient(app).post(
        "/api/update-artifact",
        json={
            "definition": prompt.spec.model_dump(by_alias=True, exclude_none=True),
            "feedback": "Use the supplied name.",
            "model": "test/model",
            "judge_model": "test/judge",
        },
    )

    assert response.status_code == 200
    assert response.json()["definition"]["prompt"]["system"].endswith(
        "Always use the supplied name."
    )
    assert response.json()["report"]["passed"] is True
