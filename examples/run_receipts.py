"""Load the example artifact and extract expense fields from every sample receipt.

Run from this directory, so `prompts.receipt_models` is importable:

    cd examples && python run_receipts.py
"""

import asyncio
import json
from pathlib import Path

from prompt_ninja import PromptNinja

HERE = Path(__file__).resolve().parent


def show(value) -> str:
    return "-" if value is None else str(value)


async def main() -> None:
    prompt = PromptNinja.from_file(HERE / "prompts" / "receipt-extract.prompt.toml")
    receipts = json.loads((HERE / "sample-receipts.json").read_text())

    print(f"{'MERCHANT':<26}{'DATE':<12}{'TOTAL':>9}{'TAX':>9}{'CARD':>7}")
    for receipt in receipts:
        # `result` is already a validated Receipt instance, not raw JSON.
        result = await prompt.run_openrouter(receipt)
        print(
            f"{result.merchant[:25]:<26}"
            f"{show(result.purchase_date):<12}"
            f"{result.total:>9.2f}"
            f"{show(result.tax):>9}"
            f"{show(result.payment_last4):>7}"
        )

    print("\nA dash means the receipt did not print that value.")
    print("None of those dashes are guesses — the tests enforce it.")


if __name__ == "__main__":
    asyncio.run(main())
