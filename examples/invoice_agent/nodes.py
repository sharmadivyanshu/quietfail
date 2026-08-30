"""Graph nodes. Each returns a partial state update, never the whole state."""

from langgraph.types import interrupt

from . import rules, store, tools
from .extract import get_extractor
from .state import AgentState, Resolution

CONFIDENCE_THRESHOLD = 0.75
MAX_RESOLUTION_ROUNDS = 3


def ingest(state: AgentState) -> dict:
    return {"findings": [], "resolutions": [], "requires_human": False, "iteration": 0}


def extract(state: AgentState) -> dict:
    extractor = get_extractor()
    invoice = extractor.extract(state["document_uri"])
    return {"extracted": invoice}


def validate(state: AgentState) -> dict:
    findings = rules.run_all(
        state["extracted"],
        known_vendor_ids=store.known_vendor_ids(),
        known_pos=store.known_pos(),
        seen_invoices=store.seen_invoices(),
    )
    return {"findings": findings}


def classify_exception(state: AgentState) -> dict:
    """Map findings to a single resolution strategy.

    Rule-based on purpose. The exception taxonomy is small and stable, so a
    model here would add latency and non-determinism for no accuracy gain.
    Revisit only when the taxonomy stops being enumerable.
    """
    errors = {f.rule for f in state["findings"] if f.severity == "error"}

    if "duplicate_invoice" in errors:
        return {"exception_class": "duplicate"}
    if errors & {"vendor_unresolved", "vendor_unknown"}:
        return {"exception_class": "vendor_unresolved"}
    if "totals_mismatch" in errors or "line_items_mismatch" in errors:
        return {"exception_class": "arithmetic_mismatch"}
    if "missing_gl_code" in errors:
        return {"exception_class": "coding_incomplete"}
    if "po_not_found" in errors:
        return {"exception_class": "po_unmatched"}
    return {"exception_class": "unclassified"}


def resolve(state: AgentState) -> dict:
    """Narrow resolvers, one per exception class."""
    inv = state["extracted"]
    cls = state["exception_class"]

    if cls == "vendor_unresolved":
        hit = tools.vendor_lookup(inv.vendor_name)
        return {
            "resolutions": [
                Resolution(
                    strategy="vendor_lookup",
                    proposal={"vendor_id": hit["vendor_id"], "matched_on": hit["matched_on"]},
                    confidence=hit["score"],
                )
            ],
            "confidence": hit["score"],
        }

    if cls == "coding_incomplete":
        proposals, scores = {}, []
        for li in inv.line_items:
            if li.gl_code:
                continue
            hit = tools.gl_code_history(inv.vendor_id, li.description)
            proposals[li.description] = hit["gl_code"]
            scores.append(hit["score"])
        conf = min(scores) if scores else 0.0
        return {
            "resolutions": [
                Resolution(strategy="gl_code_history", proposal=proposals, confidence=conf)
            ],
            "confidence": conf,
        }

    if cls == "po_unmatched":
        hit = tools.po_fuzzy_match(inv.po_number, inv.vendor_id, inv.total)
        return {
            "resolutions": [
                Resolution(strategy="po_fuzzy_match", proposal=hit, confidence=hit["score"])
            ],
            "confidence": hit["score"],
        }

    # arithmetic_mismatch, duplicate and unclassified are never auto-resolved.
    # A human decides. Encoding that as confidence 0.0 keeps one routing rule
    # instead of two.
    return {
        "resolutions": [Resolution(strategy="no_auto_resolution", proposal={}, confidence=0.0)],
        "confidence": 0.0,
    }


def apply_resolution(state: AgentState) -> dict:
    """Write the accepted proposal back onto the invoice, then let validate()
    run again. This is what closes the loop — and it is why findings must be
    replaced rather than accumulated."""
    inv = state["extracted"].model_copy(deep=True)
    last = state["resolutions"][-1]

    if last.strategy == "vendor_lookup":
        inv.vendor_id = last.proposal.get("vendor_id")
    elif last.strategy == "gl_code_history":
        for li in inv.line_items:
            if not li.gl_code and li.description in last.proposal:
                li.gl_code = last.proposal[li.description]
    elif last.strategy == "po_fuzzy_match":
        inv.po_number = last.proposal.get("po_number")

    return {"extracted": inv, "iteration": state.get("iteration", 0) + 1}


def human_review(state: AgentState) -> dict:
    """Pause for a human. The payload is designed against approval fatigue:
    it leads with why this one is unusual, not with a raw dump."""
    last = state["resolutions"][-1] if state["resolutions"] else None
    decision = interrupt(
        {
            "invoice_number": state["extracted"].invoice_number,
            "vendor": state["extracted"].vendor_name,
            "total": state["extracted"].total,
            "why_escalated": state["exception_class"],
            "findings": [f.model_dump() for f in state["findings"] if f.severity == "error"],
            "agent_proposal": last.proposal if last else None,
            "confidence": state.get("confidence", 0.0),
        }
    )
    return {"human_decision": decision, "requires_human": False}


def post(state: AgentState) -> dict:
    decision = state.get("human_decision")
    if decision is not None and not decision.get("approved", False):
        return {"outcome": "rejected"}
    return {"outcome": "posted"}


def escalate(state: AgentState) -> dict:
    return {"outcome": "escalated"}


# ---- routing functions (conditional edges) ----------------------------------


def route_after_validation(state: AgentState) -> str:
    """Clean -> post. Dirty -> try to fix, unless we've looped too long.

    The iteration cap is the loop guard. Without it a resolver that keeps
    proposing the same rejected fix spins forever — the exact shape of the
    retry loop that burned $4,200 in a documented postmortem.
    """
    if not any(f.severity == "error" for f in state["findings"]):
        return "post"
    if state.get("iteration", 0) >= MAX_RESOLUTION_ROUNDS:
        return "human_review"
    return "classify_exception"


def route_after_resolution(state: AgentState) -> str:
    """High confidence -> apply the fix and re-validate. Low -> ask a human."""
    if state.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD:
        return "apply_resolution"
    return "human_review"


def route_after_human(state: AgentState) -> str:
    decision = state.get("human_decision") or {}
    return "post" if decision.get("approved") else "escalate"
