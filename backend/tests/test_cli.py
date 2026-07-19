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
        "spec_version": "1.0",
        "prompt": {
            "name": "compiled-summary",
            "description": "Summarizes supplied project updates.",
            "used_in": ["backend/tests/test_cli.py"],
        },
        "model": {"provider": "openai", "name": "gpt-5.6-terra"},
        "template": {"system": "Summarize the update.", "user": "Update: {{update}}"},
        "variables": [{"name": "update", "type": "string", "required": True}],
        "output": "String",
        "tests": [
            {
                "name": "summary fixture",
                "input": {"update": "Launch is on track."},
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
    prompt.write_text("""spec_version = \"1.0\"
output = \"String\"
[prompt]
name = \"empty\"
description = \"No tests\"
used_in = [\"backend/tests/test_cli.py\"]
[model]
provider = \"openai\"
name = \"gpt-5.6\"
[template]
system = \"Do work.\"
""")

    result = CliRunner().invoke(cli, ["test-prompts", "--prompts-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no embedded test cases" in result.output


def test_validate_reports_valid_and_invalid_prompt_files(tmp_path):
    valid = tmp_path / "valid.prompt.toml"
    valid.write_text("""spec_version = \"1.0\"
output = \"String\"
[prompt]
name = \"valid\"
description = \"A valid prompt\"
used_in = [\"backend/tests/test_cli.py\"]
[model]
provider = \"openai\"
name = \"gpt-5.6\"
[template]
system = \"Do work.\"
""")
    invalid = tmp_path / "invalid.prompt.toml"
    invalid.write_text(
        valid.read_text().replace(
            'name = "gpt-5.6"', 'name = "gpt-5.6"\nunknown = true'
        )
    )

    result = CliRunner().invoke(cli, ["validate", str(tmp_path)])
    assert result.exit_code == 1
    assert "VALID" in result.output
    assert "INVALID" in result.output
    assert "Extra inputs" in result.output


def test_validate_reports_a_missing_output_model(tmp_path):
    prompt_file = tmp_path / "missing-model.prompt.toml"
    prompt_file.write_text("""spec_version = "1.0"
output = "app.models.OutputModelThatDoesNotExist"
[prompt]
name = "missing_model"
description = "Has a missing output model"
used_in = ["backend/tests/test_cli.py"]
[model]
provider = "openai"
name = "gpt-5.6"
[template]
system = "Do work."
""")

    result = CliRunner().invoke(cli, ["validate", str(prompt_file)])

    assert result.exit_code == 1
    assert "INVALID" in result.output
    assert "app.models.OutputModelThatDoesNotExist" in result.output
    assert "could not be imported" in result.output


def test_validate_fix_repairs_a_missing_output_model(tmp_path, monkeypatch):
    prompt_file = tmp_path / "missing-model.prompt.toml"
    prompt_file.write_text("""spec_version = "1.0"
output = "app.models.OutputModelThatDoesNotExist"
[prompt]
name = "missing_model"
description = "Has a missing output model"
used_in = ["backend/tests/test_cli.py"]
[model]
provider = "openai"
name = "gpt-5.6"
[template]
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

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
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
