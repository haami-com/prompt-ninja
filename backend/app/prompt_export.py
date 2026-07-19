"""Create validated Prompt Ninja TOML artifacts for download."""

from __future__ import annotations

import json
import re
import tomllib

from .models import PromptExportRequest
from .prompt_ninja import PromptNinja


def prompt_filename(goal: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", goal.lower()).strip("-")[:60] or "generated-prompt"
    return name + ".prompt.toml"


def export_prompt_toml(request: PromptExportRequest) -> str:
    """Build and validate the downloadable prompt-file specification."""
    name = prompt_filename(request.goal).removesuffix(".prompt.toml")
    content = "\n".join(
        [
            'spec_version = "1.0"',
            "",
            "[prompt]",
            "name = %s" % json.dumps(name, ensure_ascii=False),
            "description = %s" % json.dumps("Generated from: " + request.goal, ensure_ascii=False),
            'used_in = ["prompt-ninja"]',
            "",
            "[model]",
            'provider = "openai"',
            "name = %s" % json.dumps(request.model, ensure_ascii=False),
            "",
            "[template]",
            "system = %s" % json.dumps(request.final_prompt, ensure_ascii=False),
            'user = "{{input}}"',
            "",
            "[[variables]]",
            'name = "input"',
            'type = "string"',
            "required = true",
            "",
            "[output]",
            'format = "text"',
            "",
        ]
    )
    PromptNinja(tomllib.loads(content), source="<prompt export>")
    return content
