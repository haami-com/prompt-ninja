import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.brief_enhancement import BriefEnhancer
from app.main import app
from app.models import BriefEnhancementResult
from app.prompt_catalog import PROMPTS


def enhancement_result() -> BriefEnhancementResult:
    return BriefEnhancementResult(
        enhanced_request="Use File #1 as the source and File #2 as the tone reference.",
        outcome="Create a concise weekly project update.",
        context="Used by team leads in an operations workspace.",
        expected_output="Return decisions, owners, and due dates as bullets.",
        constraints="Do not invent missing details.",
        file_references=["File #1", "File #2"],
    )


def test_brief_enhancer_preserves_numbered_file_context_in_openai_request():
    class FakeResponses:
        def __init__(self):
            self.request = None

        async def parse(self, **request):
            self.request = request
            return SimpleNamespace(output_parsed=enhancement_result())

    responses = FakeResponses()
    result = asyncio.run(
        BriefEnhancer(client=SimpleNamespace(responses=responses)).enhance(
            "Use File #1 for facts and File #2 for tone.",
            [
                {
                    "number": 1,
                    "label": "File #1",
                    "name": "notes.txt",
                    "content": "Launch is Friday.",
                },
                {
                    "number": 2,
                    "label": "File #2",
                    "name": "voice.md",
                    "content": "Write directly.",
                },
            ],
        )
    )

    assert result.file_references == ["File #1", "File #2"]
    assert "File #1" in responses.request["input"]
    assert "notes.txt" in responses.request["input"]
    assert "Launch is Friday." in responses.request["input"]
    assert responses.request["text_format"] is BriefEnhancementResult
    assert "text" not in responses.request
    assert "JSON" in responses.request["instructions"]


def test_brief_enhancer_rejects_a_reference_to_a_file_that_was_not_uploaded():
    class FakeResponses:
        async def create(self, **_request):
            invalid = enhancement_result().model_copy(
                update={"file_references": ["File #3"]}
            )
            return SimpleNamespace(output_text=invalid.model_dump_json())

    with pytest.raises(ValueError, match="File #3"):
        asyncio.run(
            BriefEnhancer(client=SimpleNamespace(responses=FakeResponses())).enhance(
                "Use File #1 to create a project update.",
                [
                    {
                        "number": 1,
                        "label": "File #1",
                        "name": "notes.txt",
                        "content": "Launch is Friday.",
                    }
                ],
            )
        )


def test_brief_enhancer_reuses_the_startup_prompt_collection():
    assert BriefEnhancer().prompt is PROMPTS.brief_enhancer


def test_enhance_endpoint_assigns_stable_file_numbers(monkeypatch):
    captured = {}

    class FakeEnhancer:
        async def enhance(self, request_text, file_sources):
            captured["request_text"] = request_text
            captured["file_sources"] = file_sources
            return enhancement_result()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(main_module, "BriefEnhancer", FakeEnhancer)
    response = TestClient(app).post(
        "/api/enhance-brief",
        data={"request_text": "Use File #1 for facts and File #2 for tone."},
        files=[
            ("files", ("notes.txt", b"Launch is Friday.", "text/plain")),
            ("files", ("voice.md", b"Write directly.", "text/markdown")),
        ],
    )

    assert response.status_code == 200
    assert [source["label"] for source in captured["file_sources"]] == [
        "File #1",
        "File #2",
    ]
    assert [source["name"] for source in captured["file_sources"]] == [
        "notes.txt",
        "voice.md",
    ]
    assert response.json()["file_references"] == ["File #1", "File #2"]


def test_enhance_endpoint_rejects_more_than_five_files(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = TestClient(app).post(
        "/api/enhance-brief",
        data={"request_text": "Create a useful prompt from these source files."},
        files=[
            ("files", (f"source-{index}.txt", b"content", "text/plain"))
            for index in range(6)
        ],
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Upload up to five reference files."


def test_generate_labels_file_content_for_board_agents(monkeypatch):
    captured = {}

    class FakeBoard:
        def __init__(self, **_kwargs):
            pass

        async def stream(self, brief):
            captured["brief"] = brief
            if False:
                yield None

    monkeypatch.setattr(main_module, "PromptCouncil", FakeBoard)
    response = TestClient(app).post(
        "/api/generate",
        data={
            "outcome": "Create a concise project status update.",
            "source_text": "Use the uploaded references.",
        },
        files=[
            ("files", ("notes.txt", b"Launch is Friday.", "text/plain")),
            ("files", ("owners.md", b"Owner: Priya", "text/markdown")),
        ],
    )

    assert response.status_code == 200
    source_text = captured["brief"].source_text
    assert "[File #1: notes.txt]\nLaunch is Friday." in source_text
    assert "[File #2: owners.md]\nOwner: Priya" in source_text
