"""Fixture-backed master data. Stands in for the ERP in a real deployment."""

VENDORS = {
    "V-1001": {"name": "Acme Office Supplies", "default_gl": "6120", "terms": "NET30"},
    "V-1002": {"name": "Northwind Logistics", "default_gl": "6310", "terms": "NET45"},
    "V-1003": {"name": "Blue Ridge Software", "default_gl": "6450", "terms": "NET15"},
}

# messy aliases — the real reason vendor matching is hard
VENDOR_ALIASES = {
    "acme office supplies": "V-1001",
    "acme office supplies inc": "V-1001",
    "acme office": "V-1001",
    "northwind logistics": "V-1002",
    "northwind logistics llc": "V-1002",
    "blue ridge software": "V-1003",
    "blueridge software pvt ltd": "V-1003",
}

PURCHASE_ORDERS = {
    "PO-88120": {"vendor_id": "V-1001", "amount": 1250.00, "status": "open"},
    "PO-88121": {"vendor_id": "V-1002", "amount": 4300.00, "status": "open"},
    "PO-88122": {"vendor_id": "V-1003", "amount": 9600.00, "status": "closed"},
}

# historical coding decisions — what the RAG-ish GL lookup learns from
GL_HISTORY = [
    {"vendor_id": "V-1001", "keywords": ["paper", "toner", "stationery"], "gl_code": "6120"},
    {"vendor_id": "V-1001", "keywords": ["chair", "desk", "furniture"], "gl_code": "6130"},
    {"vendor_id": "V-1002", "keywords": ["freight", "shipping", "courier"], "gl_code": "6310"},
    {"vendor_id": "V-1003", "keywords": ["license", "subscription", "saas"], "gl_code": "6450"},
    {"vendor_id": "V-1003", "keywords": ["implementation", "consulting"], "gl_code": "6460"},
]

SEEN_INVOICES: set[tuple[str, str]] = {("V-1002", "NW-2291")}


def known_vendor_ids() -> set[str]:
    return set(VENDORS)


def known_pos() -> set[str]:
    return set(PURCHASE_ORDERS)


def seen_invoices() -> set[tuple[str, str]]:
    return set(SEEN_INVOICES)
