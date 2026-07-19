import asyncio
from types import SimpleNamespace

from app.models import GeneratedPromptTestRequest
from app.main import app
from app.prompt_testing import PromptTestHarness
from fastapi.testclient import TestClient


def test_harness_generates_fixture_runs_prompt_and_judges_result():
    class FakeCompletions:
        def __init__(self):
            self.responses = [
                '{"input":"Bonjour, monde!","expected_output":"A French translation that says Hello, world!","output_format":"text","expected_schema":{"type":"string"}}',
                "Hello, world!",
                '{"score":0.97,"rationale":"The response satisfies the expected translation."}',
            ]
            self.requests = []

        async def create(self, **request):
            self.requests.append(request)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.responses.pop(0)))])

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    harness = PromptTestHarness(client=client)
    result = asyncio.run(harness.run(GeneratedPromptTestRequest(
        final_prompt="Translate the user's text to English.",
        goal="Translate French text to English",
        expected_output="Return the English translation.",
        model="gpt-5.6-sol",
    )))

    assert result.passed
    assert result.score == 0.97
    assert result.input == "Bonjour, monde!"
    assert result.expected_schema == {"type": "string"}
    assert result.schema_valid
    assert result.actual_output == "Hello, world!"
    assert len(completions.requests) == 3


def test_generated_prompt_test_endpoint_requires_a_configured_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = TestClient(app).post("/api/test-generated", json={
        "final_prompt": "Translate the user's text to English.",
        "goal": "Translate French text to English",
        "model": "gpt-5.6-sol",
    })
    assert response.status_code == 503
