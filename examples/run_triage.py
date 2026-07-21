"""Load the example artifact and classify every ticket in sample-tickets.json.

Run from this directory so `prompts.ticket_triage_models` is importable:

    cd examples
    uv run --no-editable --directory ../backend python ../examples/run_triage.py

or, with prompt-ninja installed in the active environment:

    cd examples && python run_triage.py
"""

import asyncio
import json
from pathlib import Path

from prompt_ninja import PromptNinja

HERE = Path(__file__).resolve().parent


async def main() -> None:
    prompt = PromptNinja.from_file(HERE / "prompts" / "ticket-triage.prompt.toml")
    tickets = json.loads((HERE / "sample-tickets.json").read_text())

    for ticket in tickets:
        # `result` is already a validated TicketTriage instance, not raw JSON.
        result = await prompt.run_openrouter(ticket)
        flag = "ESCALATE" if result.escalation_needed else "queue"
        print(
            f"{flag:>8}  {result.severity:<8} {result.category:<16} "
            f"{result.affected_component}"
        )


if __name__ == "__main__":
    asyncio.run(main())
