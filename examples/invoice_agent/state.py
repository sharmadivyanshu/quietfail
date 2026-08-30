"""State and data schemas for the invoice exception agent."""

from operator import add
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    amount: float
    gl_code: str | None = None


class ExtractedInvoice(BaseModel):
    vendor_name: str
    vendor_id: str | None = None
    invoice_number: str
    invoice_date: str
    po_number: str | None = None
    subtotal: float
    tax: float
    total: float
    line_items: list[LineItem] = Field(default_factory=list)


class ValidationFinding(BaseModel):
    rule: str
    severity: Literal["error", "warning"]
    detail: str


class Resolution(BaseModel):
    strategy: str
    proposal: dict
    confidence: float


class AgentState(TypedDict, total=False):
    # inputs
    document_uri: str
    tenant_id: str
    # working
    extracted: ExtractedInvoice | None
    # NOT a reducer: validate() re-runs after each fix, so findings must be
    # REPLACED, not accumulated. Accumulating here was a real bug — fixed
    # errors stayed in the list forever and the graph never converged.
    findings: list[ValidationFinding]
    exception_class: str | None
    # This one IS a reducer: the attempt history is the audit trail.
    resolutions: Annotated[list[Resolution], add]
    confidence: float
    iteration: int
    # control
    requires_human: bool
    human_decision: dict | None
    # terminal
    outcome: Literal["posted", "escalated", "rejected"] | None
