"""Create validated Prompt Ninja TOML artifacts for download."""

from __future__ import annotations

import importlib
import keyword
import re
import sys
from pathlib import Path
from typing import Any

from .models import PromptExportRequest
from .prompt_compiler import CompiledOutputModel
from .core import PromptNinja

_TEMPLATE_VARIABLE_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def prompt_filename(goal: str) -> str:
    name = (
        re.sub(r"[^a-z0-9]+", "-", goal.lower()).strip("-")[:60] or "generated-prompt"
    )
    return name + ".prompt.toml"


_OUTPUT_TYPE_ANNOTATIONS = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "date": "date",
    "datetime": "datetime",
    "list[string]": "list[str]",
    "list[integer]": "list[int]",
    "list[number]": "list[float]",
    "list[boolean]": "list[bool]",
    "list[date]": "list[date]",
    "list[datetime]": "list[datetime]",
}


def write_generated_output_model(
    prompt_path: Path, output_model: dict[str, Any]
) -> str:
    """Write a compiler-declared Pydantic model and return its dotted path."""
    spec = CompiledOutputModel.model_validate(output_model)
    project_root = Path.cwd().resolve()
    model_directory = prompt_path.resolve().parent
    try:
        relative_directory = model_directory.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            "Structured prompt output must be inside the current project so its "
            "generated Pydantic model has an importable Python path."
        ) from exc
    package_parts = list(relative_directory.parts)
    if any(not part.isidentifier() or keyword.iskeyword(part) for part in package_parts):
        raise ValueError(
            "Structured prompt output directories must be valid Python package names."
        )
    model_directory.mkdir(parents=True, exist_ok=True)
    current_directory = project_root
    for part in package_parts:
        current_directory /= part
        (current_directory / "__init__.py").touch(exist_ok=True)
    prompt_name = prompt_path.name.removesuffix(".prompt.toml")
    module_name = re.sub(r"\W+", "_", prompt_name).strip("_") + "_models"
    if not module_name or not module_name.isidentifier():
        raise ValueError("Prompt filename cannot be converted to a Python module name.")
    module_path = ".".join([*package_parts, module_name])
    model_path = model_directory / (module_name + ".py")
    fields = "\n".join(
        "    %s: %s = Field(description=%r)"
        % (field.name, _OUTPUT_TYPE_ANNOTATIONS[field.type], field.description)
        for field in spec.fields
    )
    model_path.write_text(
        "from datetime import date, datetime\n\n"
        "from pydantic import BaseModel, Field\n\n\n"
        "class %s(BaseModel):\n%s\n" % (spec.class_name, fields),
        encoding="utf-8",
    )
    importlib.invalidate_caches()
    sys.modules.pop(module_path, None)
    return "%s.%s" % (module_path, spec.class_name)


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
                "used_by": [],
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
