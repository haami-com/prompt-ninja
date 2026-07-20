"""Create validated Prompt Ninja TOML artifacts for download."""

from __future__ import annotations

import re

from .models import PromptExportRequest
from .prompt_ninja import PromptNinja

_TEMPLATE_VARIABLE_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def prompt_filename(goal: str) -> str:
    name = (
        re.sub(r"[^a-z0-9]+", "-", goal.lower()).strip("-")[:60] or "generated-prompt"
    )
    return name + ".prompt.toml"


def prompt_from_export_request(request: PromptExportRequest) -> PromptNinja:
    """Build the canonical validated prompt used by the download endpoint."""
    if request.definition is not None:
        return PromptNinja(request.definition, source="<prompt export>")
    name = prompt_filename(request.goal).removesuffix(".prompt.toml")
    variable_names = [
        "input",
        *sorted(
            set(_TEMPLATE_VARIABLE_PATTERN.findall(request.final_prompt)) - {"input"}
        ),
    ]
    return PromptNinja(
        {
            "metadata": {
                "spec_version": "1.2",
                "name": name,
                "description": "Generated from: " + request.goal,
                "used_by": ["src/prompt_consumer.py"],
                "version": "1.0.0",
                "output": "String",
            },
            "llm_model": {"provider": "openrouter", "name": request.model},
            "prompt": {"system": request.final_prompt, "user": "{{input}}"},
            "variables": [
                {
                    "name": variable_name,
                    "type": "string",
                    "description": "The %s supplied to this prompt."
                    % variable_name.replace("_", " "),
                    "required": True,
                }
                for variable_name in variable_names
            ],
        },
        source="<prompt export>",
    )


def export_prompt_toml(request: PromptExportRequest) -> str:
    """Serialize the canonical validated prompt-file specification."""
    return prompt_from_export_request(request).to_toml()
