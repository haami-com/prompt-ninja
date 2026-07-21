import asyncio
from types import SimpleNamespace

from prompt_ninja.models import GeneratedPromptTestRequest
from prompt_ninja.main import app
from prompt_ninja.prompt_catalog import PROMPTS
from prompt_ninja.prompt_testing import PromptTestHarness
from fastapi.testclient import TestClient


def test_harness_generates_fixture_runs_prompt_and_judges_result():
    class FakeResponses:
        def __init__(self):
            self.responses = [
                '{"input":"Bonjour, monde!","expected_output":"A French translation that says Hello, world!","output_format":"text"}',
                "Hello, world!",
                '{"score":0.97,"rationale":"The response satisfies the expected translation."}',
            ]
            self.requests = []

        async def create(self, **request):
            self.requests.append(request)
            return SimpleNamespace(output_text=self.responses.pop(0))

    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    harness = PromptTestHarness(client=client)
    result = asyncio.run(
        harness.run(
            GeneratedPromptTestRequest(
                final_prompt="Translate {{meeting_notes}} to English.",
                goal="Translate French text to English",
                expected_output="Return the English translation.",
                model="gpt-5.6-sol",
            )
        )
    )

    assert result.passed
    assert result.score == 0.97
    assert result.input == "Bonjour, monde!"
    assert result.actual_output == "Hello, world!"
    assert len(responses.requests) == 3
    assert (
        "Translate Bonjour, monde! to English." in responses.requests[1]["instructions"]
    )


def test_harness_reuses_the_startup_prompt_collection():
    harness = PromptTestHarness()

    assert harness.fixture_generator is PROMPTS.test_case_generator
    assert harness.judge is PROMPTS.test_judge


def test_harness_executes_the_canonical_definition_with_type_correct_fixture_values():
    class FakeResponses:
        def __init__(self):
            self.responses = [
                '{"input":"Project launch notes","expected_output":"A concise summary","output_format":"text"}',
                '{"summary":"Launch is on track."}',
                '{"score":0.99,"rationale":"Correct and concise."}',
            ]
            self.requests = []

        async def create(self, **request):
            self.requests.append(request)
            return SimpleNamespace(output_text=self.responses.pop(0))

    responses = FakeResponses()
    definition = {
        "metadata": {
            "spec_version": "1.2",
            "name": "project-summary",
            "description": "Summarizes notes.",
            "used_by": ["backend/tests/test_prompt_testing.py"],
            "version": "1.0.0",
            "output": "prompt_ninja.JsonObjectOutput",
        },
        "llm_model": {"provider": "openrouter", "name": "google/gemini-2.5-flash"},
        "prompt": {
            "system": "Summarize {{meeting_notes}}.",
            "user": "Limit {{max_items}}. Metadata: {{metadata}}",
        },
        "variables": [
            {
                "name": "meeting_notes",
                "type": "string",
                "description": "Notes to summarize.",
                "required": True,
            },
            {
                "name": "max_items",
                "type": "integer",
                "description": "Maximum number of items.",
                "required": True,
            },
            {
                "name": "metadata",
                "type": "object",
                "description": "Supporting metadata.",
                "required": True,
            },
        ],
    }
    result = asyncio.run(
        PromptTestHarness(client=SimpleNamespace(responses=responses)).run(
            GeneratedPromptTestRequest(
                final_prompt="Fallback prompt text.",
                goal="Summarize project launch notes",
                model="gpt-5.6-sol",
                definition=definition,
            )
        )
    )

    execution_request = responses.requests[1]
    assert "Summarize Project launch notes." in execution_request["instructions"]
    assert (
        'Limit 1. Metadata: {"input": "Project launch notes"}'
        in execution_request["input"]
    )
    assert result.passed


def test_generated_prompt_test_endpoint_requires_a_configured_provider(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    response = TestClient(app).post(
        "/api/test-generated",
        json={
            "final_prompt": "Translate the user's text to English.",
            "goal": "Translate French text to English",
            "model": "google/gemini-2.5-flash",
        },
    )
    assert response.status_code == 503
