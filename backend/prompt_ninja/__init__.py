"""Public Python API for the Prompt Ninja distribution."""

from importlib.metadata import version

from .hooks import (
    EveryNRunEvalHook,
    OpenRouterRunJudge,
    RunEvaluation,
    RunUsage,
    TokenUsageCostHook,
)
from .core import (
    BigIntOutput,
    JsonArrayOutput,
    JsonObjectOutput,
    OpenRouterPromptClient,
    PreparedPrompt,
    PromptCollection,
    PromptFileSpec,
    PromptNinja,
    PromptNinjaError,
    PromptOutputError,
    PromptRenderError,
    PromptRunEvent,
    PromptRunHook,
    PromptRuntimeOptions,
    PromptTestReport,
    PromptTestResult,
    PromptValidationError,
    SamplingRunHook,
    TestJudgment,
    VariableSpec,
)

__version__ = version("prompt-ninja")

__all__ = [
    "BigIntOutput",
    "EveryNRunEvalHook",
    "JsonArrayOutput",
    "JsonObjectOutput",
    "OpenRouterPromptClient",
    "OpenRouterRunJudge",
    "PreparedPrompt",
    "PromptCollection",
    "PromptFileSpec",
    "PromptNinja",
    "PromptNinjaError",
    "PromptOutputError",
    "PromptRenderError",
    "PromptRunEvent",
    "PromptRunHook",
    "PromptRuntimeOptions",
    "PromptTestReport",
    "PromptTestResult",
    "PromptValidationError",
    "RunEvaluation",
    "RunUsage",
    "SamplingRunHook",
    "TestJudgment",
    "TokenUsageCostHook",
    "VariableSpec",
    "__version__",
]
