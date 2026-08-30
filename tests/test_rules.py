"""Validation rules are pure functions — test them without touching the graph."""

import pytest
from invoice_agent import rules
from invoice_agent.state import ExtractedInvoice, LineItem


def invoice(**overrides) -> ExtractedInvoice:
    base = {
        "vendor_name": "Acme Office Supplies",
        "vendor_id": "V-1001",
        "invoice_number": "ACM-1",
        "invoice_date": "2026-07-01",
        "po_number": "PO-88120",
        "subtotal": 100.0,
        "tax": 18.0,
        "total": 118.0,
        "line_items": [
            LineItem(
                description="paper", quantity=1, unit_price=100.0, amount=100.0, gl_code="6120"
            )
        ],
    }
    base.update(overrides)
    return ExtractedInvoice(**base)


def test_clean_invoice_has_no_findings():
    findings = rules.run_all(
        invoice(), known_vendor_ids={"V-1001"}, known_pos={"PO-88120"}, seen_invoices=set()
    )
    assert findings == []


def test_totals_mismatch_detected():
    findings = rules.check_totals(invoice(total=999.0))
    assert [f.rule for f in findings] == ["totals_mismatch"]


@pytest.mark.parametrize("tax", [0.0, 18.0, 5.55])
def test_totals_pass_when_arithmetic_holds(tax):
    assert rules.check_totals(invoice(tax=tax, total=100.0 + tax)) == []


def test_line_items_must_sum_to_subtotal():
    findings = rules.check_line_items_sum(invoice(subtotal=250.0))
    assert [f.rule for f in findings] == ["line_items_mismatch"]


def test_missing_gl_code_detected():
    inv = invoice(
        line_items=[LineItem(description="x", quantity=1, unit_price=100.0, amount=100.0)]
    )
    findings = rules.check_gl_codes(inv)
    assert [f.rule for f in findings] == ["missing_gl_code"]


def test_unresolved_vendor_detected():
    findings = rules.check_vendor_known(invoice(vendor_id=None), {"V-1001"})
    assert [f.rule for f in findings] == ["vendor_unresolved"]


def test_duplicate_detected():
    findings = rules.check_duplicate(invoice(), seen={("V-1001", "ACM-1")})
    assert [f.rule for f in findings] == ["duplicate_invoice"]


def test_rounding_tolerance_does_not_trip_totals():
    """Float noise must not create phantom exceptions — this is the rule that
    generates false escalations in production if you get it wrong."""
    assert rules.check_totals(invoice(subtotal=100.005, tax=18.0, total=118.005)) == []
