"""Click commands for generating, testing, updating, and serving prompts."""

from __future__ import annotations

import asyncio
import os
import tomllib
from pathlib import Path
from typing import Any

import click

from .agents import PromptCouncil
from .model_config import DEFAULT_MODEL
from .models import Brief, CouncilResult, PromptExportRequest
from .prompt_export import prompt_filename, prompt_from_export_request
from .prompt_ninja import (
    OpenAIPromptClient,
    PromptNinja,
    PromptNinjaError,
    PromptTestReport,
)

TEST_JUDGE_PROMPT_FILE = (
    Path(__file__).resolve().parents[1] / "prompts" / "test-judge.prompt.toml"
)

UPDATE_PROMPT = PromptNinja(
    {
        "spec_version": "1.0",
        "prompt": {
            "name": "prompt_updater",
            "description": "Updates a Prompt Ninja TOML file from concise feedback.",
            "used_in": ["backend/app/cli.py"],
        },
        "model": {"provider": "openai", "name": DEFAULT_MODEL},
        "template": {
            "system": (
                "You update Prompt Ninja prompt files. Apply the feedback to the TOML prompt file. "
                "Return only a complete, valid *.prompt.toml document; do not use Markdown fences. "
                'Preserve spec_version = "1.0" and every required section. '
                "The top-level output value must be String, BigInt, or a dotted path to an "
                "existing importable Pydantic BaseModel class. When feedback says an output "
                "model cannot be imported, use an existing model path only when the prompt "
                "clearly requires that model; otherwise use String for text or BigInt for an integer. "
                "Never invent a replacement model path."
            ),
            "user": "PROMPT FILE:\n{{prompt_toml}}\n\nFEEDBACK:\n{{feedback}}",
        },
        "variables": [
            {"name": "prompt_toml", "type": "string", "required": True},
            {"name": "feedback", "type": "string", "required": True},
        ],
        "output": "String",
    },
    source="<built-in update prompt>",
)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as config_file:
            loaded = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise click.ClickException("Could not read %s: %s" % (path, exc)) from exc
    brief = loaded.get("brief", loaded)
    if not isinstance(brief, dict):
        raise click.ClickException(
            "%s must contain top-level fields or a [brief] table." % path
        )
    return brief


def _write_generated_prompt(path: Path, goal: str, result: CouncilResult) -> None:
    prompt = prompt_from_export_request(
        PromptExportRequest(
            final_prompt=result.final_prompt,
            goal=goal,
            model=result.judge_model or DEFAULT_MODEL,
            definition=result.prompt_definition or None,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt.to_toml(), encoding="utf-8")


async def _generate(brief: Brief) -> CouncilResult:
    result: CouncilResult | None = None
    async for item in PromptCouncil().stream(brief):
        if isinstance(item, CouncilResult):
            result = item
    if result is None:
        raise click.ClickException("The Board of Prompts did not return a result.")
    return result


def _prompt_path(path: Path | None) -> Path:
    if path is not None:
        return path
    matches = sorted(Path("prompts").glob("*.prompt.toml"))
    if not matches:
        raise click.ClickException(
            "No prompt file found. Pass --prompt or run generate first."
        )
    return matches[0]


def _openai_executor(prompt: PromptNinja):
    if not os.getenv("OPENAI_API_KEY"):
        raise click.ClickException(
            "OPENAI_API_KEY is required to run LLM prompt tests."
        )
    client = OpenAIPromptClient()

    async def execute(prepared):
        return await client.execute(prompt, prepared)

    return execute


async def _run_tests(prompt: PromptNinja, judge_model: str) -> PromptTestReport:
    if not prompt.tests:
        return PromptTestReport(prompt_name=prompt.name, results=())
    executor = _openai_executor(prompt)
    judge_prompt = PromptNinja.from_file(TEST_JUDGE_PROMPT_FILE)

    async def judge(test, actual):
        return await judge_prompt.run_openai(
            {
                "expected_output": test.expected_output,
                "actual_output": json.dumps(actual, ensure_ascii=False),
            },
            model=judge_model,
        )

    return await prompt.arun_tests(executor, judge=judge)


def _show_report(report: PromptTestReport, verbose: bool) -> None:
    if not report.results:
        click.echo("%s: no embedded test cases (skipped)" % report.prompt_name)
        return
    for result in report.results:
        state = (
            click.style("PASS", fg="green")
            if result.passed
            else click.style("FAIL", fg="red")
        )
        score = "" if result.score is None else " (score %.2f)" % result.score
        click.echo("%s %s — %s%s" % (state, report.prompt_name, result.name, score))
        if verbose and result.passed:
            click.echo("  actual: %s" % json.dumps(result.actual, ensure_ascii=False))
        if result.rationale:
            click.echo("  %s" % result.rationale)
        if result.error:
            click.echo("  %s" % result.error)


def _strip_toml_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return "\n".join(stripped.splitlines()[1:-1]).strip()
    return stripped


async def _repair_prompt_file(prompt_file: Path, feedback: str, model: str) -> str:
    original = prompt_file.read_text(encoding="utf-8")
    updated = await UPDATE_PROMPT.run_openai(
        {"prompt_toml": original, "feedback": feedback}, model=model
    )
    candidate = _strip_toml_fence(updated)
    try:
        PromptNinja(tomllib.loads(candidate), source=str(prompt_file))
    except (tomllib.TOMLDecodeError, PromptNinjaError) as exc:
        raise click.ClickException(
            "The model returned an invalid prompt file: %s" % exc
        ) from exc
    backup = prompt_file.with_suffix(prompt_file.suffix + ".bak")
    backup.write_text(original, encoding="utf-8")
    prompt_file.write_text(candidate + "\n", encoding="utf-8")
    return str(backup)


def _validation_paths(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.prompt.toml"))
    return [path]


@click.group()
def cli() -> None:
    """Prompt Ninja command-line tools."""


@cli.command()
@click.option("--goal", help="Goal to generate a prompt for.")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("prompt-ninja.toml"),
    show_default=True,
)
@click.option(
    "--output", type=click.Path(path_type=Path), help="Destination *.prompt.toml file."
)
def generate(goal: str | None, config_path: Path, output: Path | None) -> None:
    """Generate a validated prompt TOML file from a goal."""
    config = _load_config(config_path)
    resolved_goal = goal or config.get("goal") or config.get("outcome")
    if not isinstance(resolved_goal, str) or len(resolved_goal.strip()) < 8:
        raise click.UsageError(
            "Provide --goal (at least 8 characters) or set goal in prompt-ninja.toml."
        )
    brief = Brief(
        outcome=resolved_goal,
        context=str(config.get("context", "")),
        source_text=str(config.get("source_text", "")),
        expected_output=str(config.get("expected_output", "")),
        constraints=str(config.get("constraints", "")),
    )
    destination = output or Path("prompts") / prompt_filename(resolved_goal)
    result = asyncio.run(_generate(brief))
    _write_generated_prompt(destination, resolved_goal, result)
    click.echo("Generated %s" % destination)


@cli.command("test")
@click.option(
    "--prompt",
    "prompt_path",
    type=click.Path(path_type=Path),
    help="Prompt file to test.",
)
@click.option("--judge-model", default=DEFAULT_MODEL, show_default=True)
@click.option(
    "--verbose", "-v", is_flag=True, help="Show model output for passing cases."
)
def test_prompt(prompt_path: Path | None, judge_model: str, verbose: bool) -> None:
    """Run a prompt's embedded examples and score them with an LLM judge."""
    prompt = PromptNinja.from_file(_prompt_path(prompt_path))
    report = asyncio.run(_run_tests(prompt, judge_model))
    _show_report(report, verbose)
    if not report.passed:
        raise click.exceptions.Exit(1)


@cli.command("test-prompts")
@click.option(
    "--prompts-dir",
    "-t",
    type=click.Path(path_type=Path),
    default=Path("prompts"),
    show_default=True,
)
@click.option(
    "--prompt-name", "-p", help="Run only the prompt with this [prompt].name."
)
@click.option("--judge-model", default=DEFAULT_MODEL, show_default=True)
@click.option(
    "--verbose", "-v", is_flag=True, help="Show model output for passing cases."
)
def test_prompts(
    prompts_dir: Path, prompt_name: str | None, judge_model: str, verbose: bool
) -> None:
    """Run embedded test cases across pipeline prompt TOML files."""
    paths = sorted(prompts_dir.glob("*.prompt.toml"))
    if not paths:
        raise click.ClickException("No *.prompt.toml files found in %s." % prompts_dir)
    reports: list[PromptTestReport] = []
    for path in paths:
        prompt = PromptNinja.from_file(path)
        if prompt_name and prompt.name != prompt_name:
            continue
        reports.append(asyncio.run(_run_tests(prompt, judge_model)))
    if prompt_name and not reports:
        raise click.ClickException(
            "No prompt named %r found in %s." % (prompt_name, prompts_dir)
        )
    for report in reports:
        _show_report(report, verbose)
    if any(not report.passed for report in reports):
        raise click.exceptions.Exit(1)


@cli.command()
@click.argument(
    "prompt_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("feedback")
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
def update(prompt_file: Path, feedback: str, model: str) -> None:
    """Update a prompt TOML file from natural-language feedback."""
    if not prompt_file.name.endswith(".prompt.toml"):
        raise click.UsageError("prompt_file must use the .prompt.toml extension.")
    if not os.getenv("OPENAI_API_KEY"):
        raise click.ClickException(
            "OPENAI_API_KEY is required to update a prompt with LLM feedback."
        )
    backup = asyncio.run(_repair_prompt_file(prompt_file, feedback, model))
    click.echo("Updated %s (backup: %s)" % (prompt_file, backup))


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--fix", is_flag=True, help="Use the LLM updater to repair invalid prompt files."
)
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
def validate(path: Path, fix: bool, model: str) -> None:
    """Validate one *.prompt.toml file or every such file in a directory."""
    paths = _validation_paths(path)
    if not paths:
        raise click.ClickException("No *.prompt.toml files found in %s." % path)
    if fix and not os.getenv("OPENAI_API_KEY"):
        raise click.ClickException("OPENAI_API_KEY is required to fix prompt files.")
    failures = 0
    for prompt_file in paths:
        try:
            PromptNinja.from_file(prompt_file)
            click.echo(click.style("VALID", fg="green") + " " + str(prompt_file))
        except PromptNinjaError as exc:
            failures += 1
            click.echo(
                click.style("INVALID", fg="red") + " " + str(prompt_file), err=True
            )
            click.echo(str(exc), err=True)
            if fix:
                feedback = (
                    "Repair every TOML and Prompt Ninja validation error below while preserving "
                    "the prompt's intent. Output must be String, BigInt, or an importable dotted "
                    "path to a Pydantic BaseModel. If a broken model path cannot be verified, "
                    "do not invent a path; use String for text output or BigInt for integer output.\n%s"
                    % exc
                )
                try:
                    backup = asyncio.run(
                        _repair_prompt_file(prompt_file, feedback, model)
                    )
                    PromptNinja.from_file(prompt_file)
                    failures -= 1
                    click.echo(
                        click.style("FIXED", fg="green")
                        + " %s (backup: %s)" % (prompt_file, backup)
                    )
                except click.ClickException as repair_error:
                    click.echo("Fix failed: %s" % repair_error, err=True)
    if failures:
        raise click.exceptions.Exit(1)


@cli.command()
@click.option("--port", type=click.IntRange(1, 65535), default=8000, show_default=True)
def ui(port: int) -> None:
    """Launch the Prompt Ninja web API for the bundled UI."""
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=port)


if __name__ == "__main__":
    cli()
