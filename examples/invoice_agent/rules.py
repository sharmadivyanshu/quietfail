"""Deterministic validation. No LLM calls belong in this file — ever.

Every rule here is exact arithmetic or an exact lookup. Using a model for
these is slower, costlier, non-reproducible, and untestable.
"""

from .state import ExtractedInvoice, ValidationFinding

TOLERANCE = 0.01


def check_totals(inv: ExtractedInvoice) -> list[ValidationFinding]:
    """subtotal + tax must equal total."""
    expected = round(inv.subtotal + inv.tax, 2)
    if abs(expected - inv.total) > TOLERANCE:
        return [
            ValidationFinding(
                rule="totals_mismatch",
                severity="error",
                detail=f"subtotal {inv.subtotal} + tax {inv.tax} = {expected}, "
                f"but total is {inv.total}",
            )
        ]
    return []


def check_line_items_sum(inv: ExtractedInvoice) -> list[ValidationFinding]:
    """Line items must sum to the subtotal."""
    if not inv.line_items:
        return [
            ValidationFinding(
                rule="no_line_items",
                severity="warning",
                detail="invoice has no line items to verify against subtotal",
            )
        ]
    line_sum = round(sum(li.amount for li in inv.line_items), 2)
    if abs(line_sum - inv.subtotal) > TOLERANCE:
        return [
            ValidationFinding(
                rule="line_items_mismatch",
                severity="error",
                detail=f"line items sum to {line_sum}, subtotal is {inv.subtotal}",
            )
        ]
    return []


def check_gl_codes(inv: ExtractedInvoice) -> list[ValidationFinding]:
    """Every line item needs a GL code before posting."""
    missing = [li.description for li in inv.line_items if not li.gl_code]
    if missing:
        return [
            ValidationFinding(
                rule="missing_gl_code",
                severity="error",
                detail=f"{len(missing)} line item(s) missing GL code: {missing[:3]}",
            )
        ]
    return []


def check_vendor_known(
    inv: ExtractedInvoice, known_vendor_ids: set[str]
) -> list[ValidationFinding]:
    if not inv.vendor_id:
        return [
            ValidationFinding(
                rule="vendor_unresolved",
                severity="error",
                detail=f"no vendor_id matched for '{inv.vendor_name}'",
            )
        ]
    if inv.vendor_id not in known_vendor_ids:
        return [
            ValidationFinding(
                rule="vendor_unknown",
                severity="error",
                detail=f"vendor_id {inv.vendor_id} not in vendor master",
            )
        ]
    return []


def check_duplicate(inv: ExtractedInvoice, seen: set[tuple[str, str]]) -> list[ValidationFinding]:
    key = (inv.vendor_id or inv.vendor_name, inv.invoice_number)
    if key in seen:
        return [
            ValidationFinding(
                rule="duplicate_invoice",
                severity="error",
                detail=f"invoice {inv.invoice_number} already seen for this vendor",
            )
        ]
    return []


def check_po_present(inv: ExtractedInvoice, known_pos: set[str]) -> list[ValidationFinding]:
    if inv.po_number and inv.po_number not in known_pos:
        return [
            ValidationFinding(
                rule="po_not_found",
                severity="error",
                detail=f"PO {inv.po_number} not found in PO master",
            )
        ]
    return []


def run_all(
    inv: ExtractedInvoice,
    known_vendor_ids: set[str],
    known_pos: set[str],
    seen_invoices: set[tuple[str, str]],
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    findings += check_totals(inv)
    findings += check_line_items_sum(inv)
    findings += check_gl_codes(inv)
    findings += check_vendor_known(inv, known_vendor_ids)
    findings += check_duplicate(inv, seen_invoices)
    findings += check_po_present(inv, known_pos)
    return findings
