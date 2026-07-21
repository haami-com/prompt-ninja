"""Application-owned prompt collection loaded once during process startup."""

from pathlib import Path

from .core import PromptCollection

PROMPTS_DIRECTORY = Path(__file__).resolve().parent / "prompts"
PROMPTS = PromptCollection(dir=PROMPTS_DIRECTORY)
