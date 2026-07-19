from pathlib import Path
from typing import Any

from .model_config import DEFAULT_MODEL
from .models import BriefEnhancementResult
from .prompt_ninja import PromptNinja

BRIEF_ENHANCER_PROMPT_FILE = (
    Path(__file__).resolve().parents[1] / "prompts" / "brief-enhancer.prompt.toml"
)


class BriefEnhancer:
    """Turn a free-flow request and numbered references into a reviewable brief."""

    def __init__(self, client: Any | None = None):
        self.client = client
        self.prompt = PromptNinja.from_file(BRIEF_ENHANCER_PROMPT_FILE)

    async def enhance(
        self,
        request_text: str,
        file_sources: list[dict[str, Any]],
    ) -> BriefEnhancementResult:
        result = await self.prompt.run_openai(
            {
                "request_text": request_text,
                "file_sources": file_sources,
            },
            client=self.client,
            model=DEFAULT_MODEL,
        )
        available_labels = {str(source["label"]) for source in file_sources}
        unknown_labels = sorted(set(result.file_references) - available_labels)
        if unknown_labels:
            raise ValueError(
                "Enhanced brief referenced files that were not uploaded: %s"
                % ", ".join(unknown_labels)
            )
        return result
