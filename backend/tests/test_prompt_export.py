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
    assert prompt.prepare({"input": "Example legal clause"}).user == "Example legal clause"


def test_export_endpoint_downloads_toml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    response = TestClient(app).post("/api/export-prompt", json={
        "final_prompt": "Summarize the user's text in three bullets.",
        "goal": "Summarize legal documents into plain English",
        "model": "gpt-5.6-terra",
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/toml")
    assert "attachment" in response.headers["content-disposition"]
    assert "spec_version" in response.text
