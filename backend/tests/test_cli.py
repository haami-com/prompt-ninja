from pathlib import Path

from click.testing import CliRunner

import app.cli as cli_module
from app.cli import cli
from app.models import CouncilResult, PromptSpec
from app.prompt_ninja import PromptNinja


def test_generate_uses_goal_from_config_and_writes_valid_prompt(tmp_path, monkeypatch):
    async def fake_generate(_):
        return CouncilResult(
            final_prompt="Summarize the user's text in plain English.",
            prompt_spec=PromptSpec(goal="summary", inputs=[], output_contract="text", constraints=[], assumptions=[]),
            agents=[],
            judge_model="gpt-5.6-terra",
        )

    monkeypatch.setattr(cli_module, "_generate", fake_generate)
    runner = CliRunner()
    config = tmp_path / "prompt-ninja.toml"
    config.write_text('goal = "Summarize legal documents into plain English"\n')
    output = tmp_path / "prompts" / "legal-summary.prompt.toml"

    result = runner.invoke(cli, ["generate", "--config", str(config), "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert "Generated" in result.output
    assert PromptNinja.from_file(output).name == "summarize-legal-documents-into-plain-english"


def test_cli_registers_requested_commands():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("generate", "test", "test-prompts", "update", "validate", "ui"):
        assert command in result.output


def test_test_prompts_reports_prompt_without_embedded_cases(tmp_path):
    prompt = tmp_path / "empty.prompt.toml"
    prompt.write_text(
        """spec_version = \"1.0\"
[prompt]
name = \"empty\"
description = \"No tests\"
used_in = [\"tests\"]
[model]
provider = \"openai\"
name = \"gpt-5.6\"
[template]
system = \"Do work.\"
[output]
format = \"text\"
"""
    )

    result = CliRunner().invoke(cli, ["test-prompts", "--prompts-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no embedded test cases" in result.output


def test_validate_reports_valid_and_invalid_prompt_files(tmp_path):
    valid = tmp_path / "valid.prompt.toml"
    valid.write_text(
        """spec_version = \"1.0\"
[prompt]
name = \"valid\"
description = \"A valid prompt\"
used_in = [\"tests\"]
[model]
provider = \"openai\"
name = \"gpt-5.6\"
[template]
system = \"Do work.\"
[output]
format = \"text\"
"""
    )
    invalid = tmp_path / "invalid.prompt.toml"
    invalid.write_text(valid.read_text().replace('name = "gpt-5.6"', 'name = "gpt-5.6"\nunknown = true'))

    result = CliRunner().invoke(cli, ["validate", str(tmp_path)])
    assert result.exit_code == 1
    assert "VALID" in result.output
    assert "INVALID" in result.output
    assert "Extra inputs" in result.output
