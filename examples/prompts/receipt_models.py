"""Output contract for the receipt example artifact.

Every field a receipt might not show is optional. "Not on the receipt" is a
real answer, and the artifact's tests enforce that the model gives it.
"""

from datetime import date

from pydantic import BaseModel, Field


class Receipt(BaseModel):
    merchant: str = Field(description="Store or business name printed on the receipt.")
    total: float = Field(description="Final amount charged, including tax and tip.")
    purchase_date: date | None = Field(
        default=None,
        description="Date printed on the receipt, or null when it shows none.",
    )
    tax: float | None = Field(
        default=None,
        description="Tax amount printed as its own line, or null when not itemized.",
    )
    payment_last4: str | None = Field(
        default=None,
        description="Last four card digits printed on the receipt, or null when not shown.",
    )
