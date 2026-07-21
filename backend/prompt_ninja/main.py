import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agents import PromptCouncil, default_agent_instructions
from .brief_enhancement import BriefEnhancer
from .extractors import extract_upload
from .hooks.runtime import (
    creator_1_eval_hook,
    creator_1_evaluation_store,
    creator_2_usage_hook,
    creator_2_usage_store,
)
from .models import (
    Brief,
    BriefEnhancementResult,
    CouncilResult,
    GeneratedPromptTestRequest,
    GeneratedPromptTestResult,
    PromptArtifactReport,
    PromptArtifactTestRequest,
    PromptArtifactUpdateRequest,
    PromptArtifactUpdateResult,
    PromptExportRequest,
)
from .model_config import (
    DEFAULT_CREATOR_MODELS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MODEL,
    available_models,
    is_available_model,
)
from .prompt_export import export_prompt_toml, prompt_filename
from .prompt_artifacts import run_prompt_artifact_tests, update_prompt_artifact
from .core import PromptNinja, PromptNinjaError
from .prompt_testing import PromptTestHarness

app = FastAPI(title="Board of Prompts API", version="0.1.0")
origins_env = (
    os.getenv("FRONTEND_ORIGINS")
    or os.getenv("FRONTEND_ORIGIN")
    or (
        "http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173,http://127.0.0.1:5174"
    )
)
frontend_origins = [
    origin.strip() for origin in origins_env.split(",") if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"ok": True, "provider_configured": bool(os.getenv("OPENROUTER_API_KEY"))}


@app.get("/api/prompts")
async def prompt_defaults():
    return default_agent_instructions()


@app.get("/api/models")
async def model_defaults():
    return {
        "models": await available_models(),
        "default_model": DEFAULT_MODEL,
        "creator_models": DEFAULT_CREATOR_MODELS,
        "judge_model": DEFAULT_JUDGE_MODEL,
    }


@app.get("/api/evaluations")
async def sampled_evaluations():
    completed = creator_1_eval_hook.completed_calls
    remainder = completed % creator_1_eval_hook.every
    return {
        "hook": "creator_1",
        "sample_every": creator_1_eval_hook.every,
        "completed_calls": completed,
        "next_sample_in": creator_1_eval_hook.every - remainder,
        "evaluations": [
            evaluation.model_dump(mode="json")
            for evaluation in creator_1_evaluation_store.snapshot()
        ],
    }


@app.get("/api/hooks")
async def hook_activity():
    return {
        "quality": {
            "hook": "creator_1",
            "cadence": "every_successful_response",
            "completed_calls": creator_1_eval_hook.completed_calls,
            "pending": creator_1_eval_hook.pending_evaluations,
            "evaluations": [
                evaluation.model_dump(mode="json")
                for evaluation in creator_1_evaluation_store.snapshot()
            ],
        },
        "usage": {
            "hook": "creator_2",
            "cadence": "every_successful_response",
            "pending": creator_2_usage_hook.pending_records,
            "summary": creator_2_usage_store.summary(),
            "records": [
                record.model_dump(mode="json")
                for record in creator_2_usage_store.snapshot()
            ],
        },
    }


async def _extract_numbered_files(
    files: list[UploadFile],
) -> list[dict[str, str | int]]:
    numbered_sources = []
    extracted_character_count = 0
    for index, upload in enumerate(files, start=1):
        try:
            content = await extract_upload(upload)
        except ValueError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        extracted_character_count += len(content)
        if extracted_character_count > 30000:
            raise HTTPException(
                status_code=413,
                detail="Combined extracted file content must be 30,000 characters or less.",
            )
        numbered_sources.append(
            {
                "number": index,
                "label": f"File #{index}",
                "name": upload.filename or f"file-{index}",
                "content": content,
            }
        )
    return numbered_sources


@app.post("/api/enhance-brief", response_model=BriefEnhancementResult)
async def enhance_brief(
    request_text: str = Form(...),
    files: list[UploadFile] = File(default=[]),
):
    if len(request_text.strip()) < 8:
        raise HTTPException(
            status_code=422, detail="Describe what you need in at least 8 characters."
        )
    if len(request_text) > 12000:
        raise HTTPException(
            status_code=422, detail="Keep the request under 12,000 characters."
        )
    if len(files) > 5:
        raise HTTPException(
            status_code=413, detail="Upload up to five reference files."
        )
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(
            status_code=503, detail="Set OPENROUTER_API_KEY to enhance the brief."
        )
    numbered_sources = await _extract_numbered_files(files)
    try:
        return await BriefEnhancer().enhance(request_text, numbered_sources)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="Brief enhancement failed: %s" % exc
        ) from exc


@app.post("/api/test-generated", response_model=GeneratedPromptTestResult)
async def test_generated_prompt(request: GeneratedPromptTestRequest):
    if not await is_available_model(request.model) or not await is_available_model(
        request.judge_model
    ):
        raise HTTPException(
            status_code=422, detail="Choose supported runner and judge models."
        )
    harness = PromptTestHarness()
    if not harness.enabled:
        raise HTTPException(
            status_code=503,
            detail="Set OPENROUTER_API_KEY to run generated-prompt tests.",
        )
    try:
        return await harness.run(request)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="Generated-prompt test failed: %s" % exc
        ) from exc


def _artifact_report(report) -> PromptArtifactReport:
    return PromptArtifactReport(
        prompt_name=report.prompt_name,
        passed=report.passed,
        results=[result.model_dump(mode="json") for result in report.results],
    )


@app.post("/api/test-artifact", response_model=PromptArtifactReport)
async def test_prompt_artifact(request: PromptArtifactTestRequest):
    if not await is_available_model(request.judge_model):
        raise HTTPException(status_code=422, detail="Choose a supported judge model.")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(
            status_code=503, detail="Set OPENROUTER_API_KEY to run prompt tests."
        )
    try:
        prompt = PromptNinja(request.definition, source="<browser artifact>")
        return _artifact_report(
            await run_prompt_artifact_tests(prompt, request.judge_model)
        )
    except PromptNinjaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Prompt test failed: %s" % exc) from exc


@app.post("/api/update-artifact", response_model=PromptArtifactUpdateResult)
async def update_prompt_artifact_endpoint(request: PromptArtifactUpdateRequest):
    if not await is_available_model(request.model) or not await is_available_model(
        request.judge_model
    ):
        raise HTTPException(
            status_code=422, detail="Choose supported updater and judge models."
        )
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(
            status_code=503, detail="Set OPENROUTER_API_KEY to update prompts."
        )
    try:
        original = PromptNinja(request.definition, source="<browser artifact>")
        updated = await update_prompt_artifact(original, request.feedback, request.model)
        report = await run_prompt_artifact_tests(updated, request.judge_model)
        return PromptArtifactUpdateResult(
            definition=updated.spec.model_dump(by_alias=True, exclude_none=True),
            report=_artifact_report(report),
        )
    except PromptNinjaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="Prompt update failed: %s" % exc
        ) from exc


@app.post("/api/export-prompt")
async def export_prompt(request: PromptExportRequest):
    if not await is_available_model(request.model):
        raise HTTPException(
            status_code=422, detail="Choose a supported model for the exported prompt."
        )
    try:
        content = export_prompt_toml(request)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail="Could not export prompt: %s" % exc
        ) from exc
    return Response(
        content=content,
        media_type="application/toml",
        headers={
            "Content-Disposition": 'attachment; filename="%s"'
            % prompt_filename(request.goal)
        },
    )


@app.post("/api/generate")
async def generate(
    outcome: str = Form(...),
    context: str = Form(""),
    source_text: str = Form(""),
    expected_output: str = Form(""),
    constraints: str = Form(""),
    creator_models: str = Form(""),
    judge_model: str = Form(""),
    creator_prompts: str = Form("[]"),
    judge_prompt: str = Form(""),
    files: list[UploadFile] = File(default=[]),
):
    if len(outcome.strip()) < 8:
        raise HTTPException(
            status_code=422, detail="Describe the outcome in at least 8 characters."
        )
    if len(files) > 5:
        raise HTTPException(
            status_code=413, detail="Upload up to five reference files."
        )
    if creator_models:
        try:
            selected_creators = json.loads(creator_models)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422, detail="Creator models must be a JSON array."
            ) from exc
        valid_creator_models = (
            isinstance(selected_creators, list)
            and len(selected_creators) == 3
            and all(isinstance(model, str) for model in selected_creators)
        )
        if valid_creator_models:
            valid_creator_models = all(
                [await is_available_model(model) for model in selected_creators]
            )
        if not valid_creator_models:
            raise HTTPException(
                status_code=422, detail="Choose exactly three supported creator models."
            )
    else:
        selected_creators = None
    if judge_model and not await is_available_model(judge_model):
        raise HTTPException(status_code=422, detail="Choose a supported judge model.")
    try:
        selected_creator_prompts = json.loads(creator_prompts)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422, detail="Creator prompts must be a JSON array."
        ) from exc
    if selected_creator_prompts and (
        not isinstance(selected_creator_prompts, list)
        or len(selected_creator_prompts) != 3
        or any(
            not isinstance(prompt, str) or len(prompt) > 12000
            for prompt in selected_creator_prompts
        )
    ):
        raise HTTPException(
            status_code=422,
            detail="Provide three creator prompts, each under 12,000 characters.",
        )
    if len(judge_prompt) > 12000:
        raise HTTPException(
            status_code=422, detail="The judge prompt must be under 12,000 characters."
        )
    numbered_sources = await _extract_numbered_files(files)
    extracted = [
        "[{label}: {name}]\n{content}".format(**source) for source in numbered_sources
    ]
    combined_source_text = "\n\n".join(
        part for part in [source_text, *extracted] if part
    )
    if len(combined_source_text) > 30000:
        raise HTTPException(
            status_code=413,
            detail="Combined request and extracted file content must be 30,000 characters or less.",
        )
    brief = Brief(
        outcome=outcome,
        context=context,
        source_text=combined_source_text,
        expected_output=expected_output,
        constraints=constraints,
    )

    async def events():
        try:
            async for item in PromptCouncil(
                creator_models=selected_creators,
                judge_model=judge_model or None,
                creator_prompts=selected_creator_prompts or None,
                judge_prompt=judge_prompt or None,
                creator_1_hooks=(creator_1_eval_hook,),
                creator_2_hooks=(creator_2_usage_hook,),
            ).stream(brief):
                if isinstance(item, CouncilResult):
                    yield json.dumps(
                        {"type": "result", "data": item.model_dump()}
                    ) + "\n"
                else:
                    yield json.dumps(
                        {"type": "agent", "data": item.model_dump()}
                    ) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


frontend_dist = Path(
    os.getenv(
        "FRONTEND_DIST",
        Path(__file__).resolve().parents[2] / "frontend" / "dist",
    )
)
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
