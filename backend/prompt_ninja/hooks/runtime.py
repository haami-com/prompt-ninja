"""Application-owned hook instances and their bounded demo storage."""

from collections import deque

from .sampled_eval import EveryNRunEvalHook, OpenRouterRunJudge, RunEvaluation
from .usage import RunUsage, TokenUsageCostHook


class InMemoryEvaluationStore:
    """Keep a bounded, process-local feed for the Hooks page."""

    def __init__(self, max_records: int = 100):
        self._records: deque[RunEvaluation] = deque(maxlen=max_records)

    def __call__(self, evaluation: RunEvaluation) -> None:
        self._records.append(evaluation)

    def snapshot(self) -> list[RunEvaluation]:
        return list(reversed(self._records))


class InMemoryUsageStore:
    """Keep bounded usage records and aggregate the current demo window."""

    def __init__(self, max_records: int = 100):
        self._records: deque[RunUsage] = deque(maxlen=max_records)

    def __call__(self, usage: RunUsage) -> None:
        self._records.append(usage)

    def snapshot(self) -> list[RunUsage]:
        return list(reversed(self._records))

    def summary(self) -> dict[str, int | float | None]:
        records = tuple(self._records)
        priced = [
            record.total_cost for record in records if record.total_cost is not None
        ]
        return {
            "runs": len(records),
            "input_tokens": sum(record.input_tokens for record in records),
            "output_tokens": sum(record.output_tokens for record in records),
            "total_tokens": sum(record.total_tokens for record in records),
            "total_cost": sum(priced) if priced else None,
            "priced_runs": len(priced),
        }


creator_1_evaluation_store = InMemoryEvaluationStore()
creator_1_eval_hook = EveryNRunEvalHook(
    judge=OpenRouterRunJudge(),
    sink=creator_1_evaluation_store,
    every=1,
)
creator_2_usage_store = InMemoryUsageStore()
creator_2_usage_hook = TokenUsageCostHook(sink=creator_2_usage_store)
