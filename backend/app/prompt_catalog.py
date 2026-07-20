"""Application-owned prompt collection loaded once during process startup."""

from pathlib import Path

from .prompt_ninja import PromptCollection

PROMPTS_DIRECTORY = Path(__file__).resolve().parents[1] / "prompts"
PROMPTS = PromptCollection(dir=PROMPTS_DIRECTORY)
