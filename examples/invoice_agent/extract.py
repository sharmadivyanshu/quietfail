"""Extraction layer.

Two implementations behind one interface so the graph runs with no API key.
Swap via IEA_EXTRACTOR=llm once you want to exercise the real path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from .state import ExtractedInvoice

FIXTURES = Path(__file__).parent / "fixtures" / "invoices"


class ExtractionError(RuntimeError):
    """Raised when extraction fails after every retry.

    Deliberately loud. An extractor that quietly returns an empty invoice is
    exactly the silent failure quietfail exists to catch.
    """


class Extractor(Protocol):
    def extract(self, document_uri: str) -> ExtractedInvoice: ...


EXTRACTION_PROMPT = """You extract structured data from vendor invoices.
Return every field you can read from the document. Do not infer or invent
values that are not present — leave them null. Do not compute totals; report
only what is printed on the document.

Document:
{document}
"""


class StubExtractor:
    """Reads pre-parsed fixtures. Deterministic, free, offline."""

    def extract(self, document_uri: str) -> ExtractedInvoice:
        path = FIXTURES / f"{document_uri}.json"
        if not path.exists():
            raise ExtractionError(f"no fixture for {document_uri!r} at {path}")
        return ExtractedInvoice(**json.loads(path.read_text()))


class LLMExtractor:
    """Structured output with a validation retry.

    The retry matters: the common failure is a near-miss schema, not a total
    refusal. `llm` is injectable so the retry logic is testable without an
    API key — see tests/test_extract.py.
    """

    def __init__(self, model: str = "gpt-4o-mini", max_retries: int = 2, llm: Any = None):
        if llm is None:
            from langchain.chat_models import init_chat_model

            llm = init_chat_model(model).with_structured_output(ExtractedInvoice)
        self.llm = llm
        self.max_retries = max_retries

    def extract(self, document_uri: str) -> ExtractedInvoice:
        path = FIXTURES / f"{document_uri}.txt"
        if not path.exists():
            raise ExtractionError(f"no document text for {document_uri!r} at {path}")
        raw = path.read_text()

        attempts: list[str] = []
        for attempt in range(1, self.max_retries + 2):
            try:
                result = self.llm.invoke(EXTRACTION_PROMPT.format(document=raw))
                if not isinstance(result, ExtractedInvoice):
                    result = ExtractedInvoice(**result)
                return result
            except Exception as exc:
                attempts.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        raise ExtractionError(
            f"extraction failed for {document_uri!r} after {len(attempts)} attempts:\n"
            + "\n".join(attempts)
        )


def get_extractor() -> Extractor:
    if os.getenv("IEA_EXTRACTOR", "stub").lower() == "llm":
        return LLMExtractor(os.getenv("IEA_MODEL", "gpt-4o-mini"))
    return StubExtractor()
