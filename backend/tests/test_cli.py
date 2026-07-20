import asyncio
from pathlib import Path

from click.testing import CliRunner

import app.cli as cli_module
from app.cli import cli
from app.models import CouncilResult, PromptSpec
from app.prompt_ninja import PromptNinja, PromptTestReport, PromptTestResult


def test_generate_uses_goal_from_config_and_writes_valid_prompt(tmp_path, monkeypatch):
    async def fake_generate(_):
        return CouncilResult(
            final_prompt="Summarize the user's text in plain English.",
            prompt_spec=PromptSpec(
                goal="summary",
                inputs=[],
                output_contract="text",
                constraints=[],
                assumptions=[],
            ),
            agents=[],
            judge_model="gpt-5.6-terra",
        )

    monkeypatch.setattr(cli_module, "_generate", fake_generate)
    runner = CliRunner()
    config = tmp_path / "prompt-ninja.toml"
    config.write_text('goal = "Summarize legal documents into plain English"\n')
    output = tmp_path / "prompts" / "legal-summary.prompt.toml"

    result = runner.invoke(
        cli, ["generate", "--config", str(config), "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert "Generated" in result.output
    assert (
        PromptNinja.from_file(output).name
        == "summarize-legal-documents-into-plain-english"
    )


def test_generate_preserves_the_compiled_prompt_definition(tmp_path, monkeypatch):
    definition = {
        "metadata": {
            "spec_version": "1.2",
            "name": "compiled-summary",
            "description": "Summarizes supplied project updates.",
            "used_by": ["backend/tests/test_cli.py"],
            "version": "1.0.0",
            "output": "String",
        },
        "llm_model": {"provider": "openrouter", "name": "google/gemini-2.5-flash"},
        "prompt": {"system": "Summarize the update.", "user": "Update: {{update}}"},
        "variables": [
            {
                "name": "update",
                "type": "string",
                "description": "The project update.",
                "required": True,
            }
        ],
        "tests": [
            {
                "name": "summary fixture",
                "variable": {"update": "Launch is on track."},
                "expected_output": "A concise status summary.",
            }
        ],
    }

    async def fake_generate(_):
        return CouncilResult(
            final_prompt="Summarize the update.",
            prompt_spec=PromptSpec(
                goal="summary",
                inputs=[],
                output_contract="text",
                constraints=[],
                assumptions=[],
            ),
            prompt_definition=definition,
            agents=[],
            judge_model="gpt-5.6-terra",
        )

    monkeypatch.setattr(cli_module, "_generate", fake_generate)
    output = tmp_path / "compiled.prompt.toml"
    result = CliRunner().invoke(
        cli,
        ["generate", "--goal", "Summarize project updates", "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert PromptNinja.from_file(output).spec.model_dump(
        by_alias=True, exclude_none=True
    ) == PromptNinja(definition).spec.model_dump(by_alias=True, exclude_none=True)


def test_cli_registers_requested_commands():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("generate", "test", "test-prompts", "update", "validate", "ui"):
        assert command in result.output


def test_test_prompts_reports_prompt_without_embedded_cases(tmp_path):
    prompt = tmp_path / "empty.prompt.toml"
    prompt.write_text("""[metadata]
spec_version = \"1.2\"
name = \"empty\"
description = \"No tests\"
used_by = [\"backend/tests/test_cli.py\"]
version = \"1.0.0\"
output = \"String\"
[llm_model]
provider = \"openrouter\"
name = \"google/gemini-2.5-flash\"
[prompt]
system = \"Do work.\"
""")

    result = CliRunner().invoke(
        cli, ["test-prompts", "--prompts-dir", str(tmp_path), "--verbose"]
    )
    assert result.exit_code == 0
    assert "no embedded test cases" in result.output
    assert "Summary: 0 passed, 0 failed" in result.output


def test_test_prompts_renders_rich_results_table(tmp_path, monkeypatch):
    prompt = tmp_path / "rich-output.prompt.toml"
    prompt.write_text("""[metadata]
spec_version = "1.2"
name = "rich-output"
description = "Rich CLI fixture"
used_by = ["backend/tests/test_cli.py"]
version = "1.0.0"
output = "String"
[llm_model]
provider = "openrouter"
name = "google/gemini-2.5-flash"
[prompt]
system = "Do work."
""")

    async def fake_suite(prompts, _judge_model, show_progress=True):
        return [
            PromptTestReport(
                prompt_name=prompts[0].name,
                results=(
                    PromptTestResult(
                        name="semantic fixture",
                        passed=True,
                        expected="A correct result.",
                        actual="Correct result.",
                        score=1.0,
                        rationale="Matches the contract.",
                    ),
                ),
            )
        ]

    monkeypatch.setattr(cli_module, "_run_prompt_test_suite", fake_suite)
    result = CliRunner().invoke(
        cli, ["test-prompts", "--prompts-dir", str(tmp_path), "--verbose"]
    )

    assert result.exit_code == 0, result.output
    assert "Status" in result.output
    assert "semantic fixture" in result.output
    assert "expected:" in result.output
    assert "Correct result." in result.output
    assert "Summary: 1 passed, 0 failed" in result.output


def test_test_commands_select_one_named_case_and_support_plain_output(
    tmp_path, monkeypatch
):
    prompt = tmp_path / "filterable.prompt.toml"
    prompt.write_text("""[metadata]
spec_version = "1.2"
name = "filterable"
description = "CLI selection fixture"
used_by = ["backend/tests/test_cli.py"]
version = "1.0.0"
output = "String"
[llm_model]
provider = "openrouter"
name = "google/gemini-2.5-flash"
[prompt]
user = "{{source}}"
[[variables]]
name = "source"
type = "string"
description = "Source text."
required = true
[[tests]]
name = "first case"
variable.source = "first"
expected_output = "A first result."
[[tests]]
name = "second case"
variable.source = "second"
expected_output = "A second result."
""")
    captured = {}

    async def fake_suite(prompts, _judge_model, show_progress=True):
        captured["tests"] = [test.name for test in prompts[0].tests]
        captured["show_progress"] = show_progress
        return [
            PromptTestReport(
                prompt_name=prompts[0].name,
                results=(
                    PromptTestResult(
                        name=prompts[0].tests[0].name or "",
                        passed=True,
                        expected="A second result.",
                        score=1.0,
                        rationale="Matches.",
                    ),
                ),
            )
        ]

    monkeypatch.setattr(cli_module, "_run_prompt_test_suite", fake_suite)
    result = CliRunner().invoke(
        cli,
        [
            "test",
            "--prompt",
            str(prompt),
            "--test-name",
            "second case",
            "--plain",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {"tests": ["second case"], "show_progress": False}
    assert "PASS filterable / second case" in result.output
    assert "┏" not in result.output


def test_prompt_test_suite_reuses_and_closes_one_openrouter_client(monkeypatch):
    class FakePromptClient:
        instances = []

        def __init__(self):
            self.closed = False
            self.instances.append(self)

        async def execute(self, prompt, _prepared, runtime=None):
            if prompt.name == "test_judge":
                return {"score": 1.0, "rationale": "Matches the contract."}
            return "A correct summary."

        async def aclose(self):
            self.closed = True

    definition = {
        "metadata": {
            "spec_version": "1.2",
            "name": "suite-fixture",
            "description": "A prompt for exercising the CLI test suite.",
            "used_by": ["backend/tests/test_cli.py"],
            "version": "1.0.0",
            "output": "String",
        },
        "llm_model": {"provider": "openrouter", "name": "google/gemini-2.5-flash"},
        "prompt": {"user": "Summarize {{source}}."},
        "variables": [
            {
                "name": "source",
                "type": "string",
                "description": "Text to summarize.",
                "required": True,
            }
        ],
        "tests": [
            {
                "name": "summary fixture",
                "variable": {"source": "Launch notes"},
                "expected_output": "A correct summary.",
            }
        ],
    }
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(cli_module, "OpenRouterPromptClient", FakePromptClient)

    reports = asyncio.run(
        cli_module._run_prompt_test_suite(
            [PromptNinja(definition), PromptNinja(definition)],
            "google/gemini-2.5-flash",
        )
    )

    assert all(report.passed for report in reports)
    assert len(FakePromptClient.instances) == 1
    assert FakePromptClient.instances[0].closed


def test_validate_reports_valid_and_invalid_prompt_files(tmp_path):
    valid = tmp_path / "valid.prompt.toml"
    valid.write_text("""[metadata]
spec_version = \"1.2\"
name = \"valid\"
description = \"A valid prompt\"
used_by = [\"backend/tests/test_cli.py\"]
version = \"1.0.0\"
output = \"String\"
[llm_model]
provider = \"openrouter\"
name = \"google/gemini-2.5-flash\"
[prompt]
system = \"Do work.\"
""")
    invalid = tmp_path / "invalid.prompt.toml"
    invalid.write_text(
        valid.read_text().replace(
            'name = "google/gemini-2.5-flash"',
            'name = "google/gemini-2.5-flash"\nunknown = true',
        )
    )

    result = CliRunner().invoke(cli, ["validate", str(tmp_path)])
    assert result.exit_code == 1
    assert "VALID" in result.output
    assert "INVALID" in result.output
    assert "Extra inputs" in result.output


def test_validate_reports_a_missing_output_model(tmp_path):
    prompt_file = tmp_path / "missing-model.prompt.toml"
    prompt_file.write_text("""[metadata]
spec_version = "1.2"
name = "missing_model"
description = "Has a missing output model"
used_by = ["backend/tests/test_cli.py"]
version = "1.0.0"
output = "app.models.OutputModelThatDoesNotExist"
[llm_model]
provider = "openrouter"
name = "google/gemini-2.5-flash"
[prompt]
system = "Do work."
""")

    result = CliRunner().invoke(cli, ["validate", str(prompt_file)])

    assert result.exit_code == 1
    assert "INVALID" in result.output
    assert "app.models.OutputModelThatDoesNotExist" in result.output
    assert "could not be imported" in result.output


def test_validate_fix_repairs_a_missing_output_model(tmp_path, monkeypatch):
    prompt_file = tmp_path / "missing-model.prompt.toml"
    prompt_file.write_text("""[metadata]
spec_version = "1.2"
name = "missing_model"
description = "Has a missing output model"
used_by = ["backend/tests/test_cli.py"]
version = "1.0.0"
output = "app.models.OutputModelThatDoesNotExist"
[llm_model]
provider = "openrouter"
name = "google/gemini-2.5-flash"
[prompt]
system = "Return a short text response."
""")
    repair_call = {}

    async def fake_repair(path, feedback, model):
        repair_call.update(path=path, feedback=feedback, model=model)
        original = path.read_text()
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(original)
        path.write_text(
            original.replace(
                'output = "app.models.OutputModelThatDoesNotExist"',
                'output = "String"',
            )
        )
        return str(backup)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(cli_module, "_repair_prompt_file", fake_repair)

    result = CliRunner().invoke(
        cli,
        ["validate", str(prompt_file), "--fix", "--model", "gpt-5.6-sol"],
    )

    assert result.exit_code == 0, result.output
    assert "FIXED" in result.output
    assert repair_call["path"] == prompt_file
    assert repair_call["model"] == "gpt-5.6-sol"
    assert "could not be imported" in repair_call["feedback"]
    assert "do not invent a path" in repair_call["feedback"]
    assert PromptNinja.from_file(prompt_file).output_format == "text"
