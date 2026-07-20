"""Click commands for generating, testing, updating, and serving prompts."""

from __future__ import annotations

import asyncio
import json
import os
import tomllib
from pathlib import Path
from typing import Any, Callable

import click
from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.text import Text

from .agents import PromptCouncil
from .model_config import DEFAULT_MODEL
from .models import Brief, CouncilResult, PromptExportRequest
from .prompt_catalog import PROMPTS
from .prompt_export import prompt_filename, prompt_from_export_request
from .prompt_ninja import (
    OpenRouterPromptClient,
    PromptCollection,
    PromptNinja,
    PromptNinjaError,
    PromptRuntimeOptions,
    PromptTestReport,
    PromptTestResult,
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


def _select_test(prompt: PromptNinja, test_name: str | None) -> PromptNinja | None:
    """Return a prompt containing only the requested named test, when selected."""
    if test_name is None:
        return prompt
    matches = [test for test in prompt.tests if test.name == test_name]
    if not matches:
        return None
    definition = prompt.spec.model_dump(by_alias=True, exclude_none=True)
    definition["tests"] = [matches[0].model_dump(exclude_none=True)]
    return PromptNinja(definition, source=prompt.source)


def _openrouter_executor(prompt: PromptNinja, client: OpenRouterPromptClient):
    if not os.getenv("OPENROUTER_API_KEY"):
        raise click.ClickException(
            "OPENROUTER_API_KEY is required to run LLM prompt tests."
        )

    async def execute(prepared):
        return await client.execute(prompt, prepared)

    return execute


async def _run_tests(
    prompt: PromptNinja,
    judge_model: str,
    client: OpenRouterPromptClient | None = None,
    on_start: Callable[[Any], Any] | None = None,
    on_result: Callable[[PromptTestResult], Any] | None = None,
) -> PromptTestReport:
    if not prompt.tests:
        return PromptTestReport(prompt_name=prompt.name, results=())
    owned_client = client is None
    prompt_client = client or OpenRouterPromptClient()
    try:
        executor = _openrouter_executor(prompt, prompt_client)
        judge_prompt = PROMPTS.test_judge

        async def judge(test, actual):
            if isinstance(actual, BaseModel):
                actual = actual.model_dump(mode="json")
            prepared = judge_prompt.prepare(
                {
                    "expected_output": test.expected_output,
                    "actual_output": json.dumps(actual, ensure_ascii=False),
                }
            )
            return await prompt_client.execute(
                judge_prompt,
                prepared,
                runtime=PromptRuntimeOptions(model=judge_model),
            )

        return await prompt.arun_tests(
            executor,
            judge=judge,
            on_start=on_start,
            on_result=on_result,
        )
    finally:
        if owned_client:
            await prompt_client.aclose()


async def _run_prompt_test_suite(
    prompts: list[PromptNinja], judge_model: str, show_progress: bool = True
) -> list[PromptTestReport]:
    """Keep one HTTP client and event loop alive for a complete CLI test run."""
    client = OpenRouterPromptClient()
    total = sum(len(prompt.tests) for prompt in prompts)
    progress = Progress() if total and show_progress else None
    task_id = (
        progress.add_task("Preparing prompt tests", total=total) if progress else None
    )
    try:
        if progress:
            progress.start()
        reports: list[PromptTestReport] = []
        for prompt in prompts:
            if progress and task_id is not None:
                progress.update(
                    task_id,
                    description="Running %s (%d tests)"
                    % (prompt.name, len(prompt.tests)),
                )

            def on_result(
                result: PromptTestResult, prompt_name: str = prompt.name
            ) -> None:
                status = "passed" if result.passed else "failed"
                if progress and task_id is not None:
                    progress.update(
                        task_id,
                        advance=1,
                        description="%s: %s %s" % (prompt_name, status, result.name),
                    )

            def on_start(test: Any, prompt_name: str = prompt.name) -> None:
                if progress and task_id is not None:
                    progress.update(
                        task_id,
                        description="%s: running %s"
                        % (prompt_name, test.name or "unnamed test"),
                    )

            reports.append(
                await _run_tests(
                    prompt,
                    judge_model,
                    client=client,
                    on_start=on_start,
                    on_result=on_result,
                )
            )
        if progress and task_id is not None:
            progress.update(task_id, description="Completed prompt tests")
        return reports
    finally:
        if progress:
            progress.stop()
        await client.aclose()


def _short_text(value: str, limit: int = 110) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else "%s…" % compact[: limit - 1]


def _json_display(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _show_verbose_result(
    report: PromptTestReport, result: PromptTestResult, console: Console | None
) -> None:
    sections = [
        "%s / %s" % (report.prompt_name, result.name),
        "input:\n%s" % _json_display(result.input),
        "expected:\n%s" % _json_display(result.expected),
    ]
    if result.actual is not None:
        sections.append("actual:\n%s" % _json_display(result.actual))
    if result.rationale:
        sections.append("rationale:\n%s" % result.rationale)
    if result.error:
        sections.append("error:\n%s" % result.error)
    output = "\n".join(sections)
    if console is None:
        click.echo(output)
    else:
        console.print(Text(output, style="dim"))


def _show_report(
    report: PromptTestReport, verbose: bool, console: Console | None
) -> None:
    if console is None:
        if not report.results:
            click.echo("%s: no embedded test cases (skipped)" % report.prompt_name)
            return
        for result in report.results:
            state = "PASS" if result.passed else "FAIL"
            score = "—" if result.score is None else "%.2f" % result.score
            detail = result.error or result.rationale or "No rationale returned."
            click.echo(
                "%s %s / %s (%s): %s"
                % (state, report.prompt_name, result.name, score, _short_text(detail))
            )
            if verbose:
                _show_verbose_result(report, result, console)
        return
    if not report.results:
        console.print(
            Text(
                "%s: no embedded test cases (skipped)" % report.prompt_name,
                style="yellow",
            )
        )
        return

    table = Table()
    table.add_column("Status", justify="center", no_wrap=True)
    table.add_column("Test", min_width=24)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Result", ratio=1)
    for result in report.results:
        state = Text.styled(
            "PASS" if result.passed else "FAIL",
            "bold green" if result.passed else "bold red",
        )
        detail = result.error or result.rationale or "No rationale returned."
        table.add_row(
            state,
            Text(result.name),
            Text("—" if result.score is None else "%.2f" % result.score),
            Text(_short_text(detail), style="red" if result.error else ""),
        )
    console.print(
        Text("%s (%d tests)" % (report.prompt_name, len(report.results)), style="bold")
    )
    console.print(table)
    if verbose:
        for result in report.results:
            _show_verbose_result(report, result, console)


def _show_test_summary(
    reports: list[PromptTestReport], console: Console | None
) -> None:
    results = [result for report in reports for result in report.results]
    passed = sum(result.passed for result in results)
    failed = len(results) - passed
    style = "bold green" if failed == 0 else "bold red"
    summary = "Summary: %d passed, %d failed across %d prompts." % (
        passed,
        failed,
        len(reports),
    )
    if console is None:
        click.echo(summary)
    else:
        console.print(
            Text.styled(
                summary,
                style,
            )
        )


def _strip_toml_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return "\n".join(stripped.splitlines()[1:-1]).strip()
    return stripped


async def _repair_prompt_file(prompt_file: Path, feedback: str, model: str) -> str:
    original = prompt_file.read_text(encoding="utf-8")
    updater_prompt = PROMPTS.prompt_updater
    updated = await updater_prompt.run_openrouter(
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
    "--test-name", "-n", help="Run only one embedded test by its [[tests]].name."
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Show actual model output for every case."
)
@click.option("--plain", is_flag=True, help="Disable Rich tables and progress output.")
def test_prompt(
    prompt_path: Path | None,
    judge_model: str,
    test_name: str | None,
    verbose: bool,
    plain: bool,
) -> None:
    """Run a prompt's embedded examples and score them with an LLM judge."""
    prompt = PromptNinja.from_file(_prompt_path(prompt_path))
    selected_prompt = _select_test(prompt, test_name)
    if selected_prompt is None:
        raise click.ClickException(
            "No embedded test named %r found in %s." % (test_name, prompt.name)
        )
    report = asyncio.run(
        _run_prompt_test_suite([selected_prompt], judge_model, show_progress=not plain)
    )[0]
    console = None if plain else Console()
    _show_report(report, verbose, console)
    _show_test_summary([report], console)
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
    "--test-name", "-n", help="Run only one embedded test by its [[tests]].name."
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Show actual model output for every case."
)
@click.option("--plain", is_flag=True, help="Disable Rich tables and progress output.")
def test_prompts(
    prompts_dir: Path,
    prompt_name: str | None,
    judge_model: str,
    test_name: str | None,
    verbose: bool,
    plain: bool,
) -> None:
    """Run embedded test cases across pipeline prompt TOML files."""
    paths = sorted(prompts_dir.glob("*.prompt.toml"))
    if not paths:
        raise click.ClickException("No *.prompt.toml files found in %s." % prompts_dir)
    prompts: list[PromptNinja] = []
    for prompt in PromptCollection(dir=prompts_dir).values():
        if prompt_name and prompt.name != prompt_name:
            continue
        selected_prompt = _select_test(prompt, test_name)
        if selected_prompt is not None:
            prompts.append(selected_prompt)
    if prompt_name and not prompts:
        raise click.ClickException(
            "No prompt named %r found in %s." % (prompt_name, prompts_dir)
        )
    if test_name and not prompts:
        raise click.ClickException(
            "No embedded test named %r found in %s." % (test_name, prompts_dir)
        )
    reports = asyncio.run(
        _run_prompt_test_suite(prompts, judge_model, show_progress=not plain)
    )
    console = None if plain else Console()
    for report in reports:
        _show_report(report, verbose, console)
    _show_test_summary(reports, console)
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
    if not os.getenv("OPENROUTER_API_KEY"):
        raise click.ClickException(
            "OPENROUTER_API_KEY is required to update a prompt with LLM feedback."
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
    if fix and not os.getenv("OPENROUTER_API_KEY"):
        raise click.ClickException(
            "OPENROUTER_API_KEY is required to fix prompt files."
        )
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
