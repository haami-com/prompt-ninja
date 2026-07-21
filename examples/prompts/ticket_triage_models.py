"""Output contract for the ticket-triage example artifact."""

from typing import Literal

from pydantic import BaseModel, Field


class TicketTriage(BaseModel):
    category: Literal[
        "access",
        "billing",
        "bug",
        "feature_request",
        "how_to",
        "outage",
        "performance",
        "security",
        "other",
    ] = Field(description="The primary reason the customer opened the ticket.")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Impact demonstrated by the ticket, not the tone of the request."
    )
    affected_component: str = Field(
        description="Named system, feature, or integration; 'unknown' when none is stated."
    )
    escalation_needed: bool = Field(
        description="Whether the ticket needs specialist or urgent handling."
    )
