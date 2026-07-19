from pathlib import Path

from app.main import app
from app.prompt_export import export_prompt_toml
from app.models import PromptExportRequest
from app.prompt_ninja import PromptNinja
from fastapi.testclient import TestClient


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

    assert prepared.system == "Summarize Launch planning notes for executives."


def test_export_serializes_a_supplied_prompt_ninja_definition_without_losing_its_schema_or_tests(
    tmp_path,
):
    source = Path(__file__).resolve().parents[1] / "prompts" / "greeting.prompt.toml"
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
            "model": "gpt-5.6-sol",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/toml")
    assert "attachment" in response.headers["content-disposition"]
    assert "spec_version" in response.text


def test_export_endpoint_downloads_the_canonical_definition_losslessly(tmp_path):
    source = Path(__file__).resolve().parents[1] / "prompts" / "greeting.prompt.toml"
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
