"""Resolution tools. These are what the agent calls when validation fails.

Kept narrow on purpose: the classifier routes to a resolver that sees 2-3
tools, not one resolver that sees all of them. Tool-count reduction measurably
improves function-calling accuracy (arXiv 2411.15399).
"""

import difflib

from quietfail import instrument_tool

from .store import GL_HISTORY, PURCHASE_ORDERS, VENDOR_ALIASES, VENDORS


@instrument_tool(is_empty=lambda r: not r.get("vendor_id"))
def vendor_lookup(vendor_name: str) -> dict:
    """Resolve a free-text vendor name to a vendor_id, with a match score."""
    key = vendor_name.lower().strip().rstrip(".")
    if key in VENDOR_ALIASES:
        vid = VENDOR_ALIASES[key]
        return {"vendor_id": vid, "score": 1.0, "matched_on": "alias", **VENDORS[vid]}

    candidates = list(VENDOR_ALIASES)
    best = difflib.get_close_matches(key, candidates, n=1, cutoff=0.6)
    if best:
        vid = VENDOR_ALIASES[best[0]]
        score = difflib.SequenceMatcher(None, key, best[0]).ratio()
        return {"vendor_id": vid, "score": round(score, 3), "matched_on": "fuzzy", **VENDORS[vid]}

    return {"vendor_id": None, "score": 0.0, "matched_on": "none"}


@instrument_tool(is_empty=lambda r: not r.get("po_number"))
def po_fuzzy_match(po_number: str | None, vendor_id: str | None, amount: float) -> dict:
    """Find the most plausible PO for this invoice."""
    if po_number and po_number in PURCHASE_ORDERS:
        po = PURCHASE_ORDERS[po_number]
        return {
            "po_number": po_number,
            "score": 1.0,
            "amount_delta": round(amount - po["amount"], 2),
            **po,
        }

    best, best_delta = None, None
    for pid, po in PURCHASE_ORDERS.items():
        if vendor_id and po["vendor_id"] != vendor_id:
            continue
        if po["status"] != "open":
            continue
        delta = abs(amount - po["amount"])
        if best_delta is None or delta < best_delta:
            best, best_delta = pid, delta

    if best is None:
        return {"po_number": None, "score": 0.0}

    po = PURCHASE_ORDERS[best]
    # score decays as the amount gap widens
    score = max(0.0, 1.0 - (best_delta / max(po["amount"], 1.0)))
    return {
        "po_number": best,
        "score": round(score, 3),
        "amount_delta": round(amount - po["amount"], 2),
        **po,
    }


@instrument_tool(is_empty=lambda r: not r.get("gl_code"))
def gl_code_history(vendor_id: str | None, description: str) -> dict:
    """Infer a GL code from how similar line items were coded before."""
    desc = description.lower()
    for entry in GL_HISTORY:
        if vendor_id and entry["vendor_id"] != vendor_id:
            continue
        hits = [kw for kw in entry["keywords"] if kw in desc]
        if hits:
            return {
                "gl_code": entry["gl_code"],
                "score": round(len(hits) / len(entry["keywords"]), 3),
                "matched_keywords": hits,
            }

    if vendor_id and vendor_id in VENDORS:
        return {
            "gl_code": VENDORS[vendor_id]["default_gl"],
            "score": 0.4,
            "matched_keywords": ["vendor_default"],
        }

    return {"gl_code": None, "score": 0.0, "matched_keywords": []}
