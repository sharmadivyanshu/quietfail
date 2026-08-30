"""Extraction: fixture path, LLM retry logic, and failure loudness.

The LLM path is exercised with an injected fake model, so the retry and
error-handling logic is covered without an API key. What is NOT covered here
is real model output quality — that needs a live key and a labelled set.
"""

import pytest
from invoice_agent.extract import (
    EXTRACTION_PROMPT,
    ExtractionError,
    LLMExtractor,
    StubExtractor,
)
from invoice_agent.state import ExtractedInvoice

VALID = {
    "vendor_name": "Acme Office Supplies Inc",
    "vendor_id": "V-1001",
    "invoice_number": "ACM-4471",
    "invoice_date": "2026-07-14",
    "po_number": "PO-88120",
    "subtotal": 1250.0,
    "tax": 225.0,
    "total": 1475.0,
    "line_items": [],
}


class FakeLLM:
    """Fails `fail_times` times, then returns `result`."""

    def __init__(self, result, fail_times: int = 0):
        self.result = result
        self.fail_times = fail_times
        self.calls = 0
        self.last_prompt = None

    def invoke(self, prompt):
        self.calls += 1
        self.last_prompt = prompt
        if self.calls <= self.fail_times:
            raise ValueError("schema validation failed")
        return self.result


def test_stub_reads_the_fixture():
    invoice = StubExtractor().extract("clean-001")
    assert invoice.invoice_number == "ACM-4471"
    assert invoice.total == 1475.0


def test_stub_fails_loudly_on_unknown_document():
    with pytest.raises(ExtractionError):
        StubExtractor().extract("does-not-exist")


def test_llm_extractor_succeeds_first_try():
    llm = FakeLLM(ExtractedInvoice(**VALID))
    invoice = LLMExtractor(llm=llm).extract("clean-001")
    assert invoice.invoice_number == "ACM-4471"
    assert llm.calls == 1


def test_llm_extractor_retries_then_succeeds():
    llm = FakeLLM(ExtractedInvoice(**VALID), fail_times=2)
    invoice = LLMExtractor(llm=llm, max_retries=2).extract("clean-001")
    assert invoice.total == 1475.0
    assert llm.calls == 3


def test_llm_extractor_raises_after_exhausting_retries():
    llm = FakeLLM(ExtractedInvoice(**VALID), fail_times=99)
    with pytest.raises(ExtractionError) as exc:
        LLMExtractor(llm=llm, max_retries=2).extract("clean-001")
    assert "after 3 attempts" in str(exc.value)
    assert llm.calls == 3


def test_llm_extractor_coerces_a_plain_dict():
    """Some providers return a dict rather than the model instance."""
    invoice = LLMExtractor(llm=FakeLLM(VALID)).extract("clean-001")
    assert isinstance(invoice, ExtractedInvoice)


def test_llm_extractor_needs_document_text():
    with pytest.raises(ExtractionError):
        LLMExtractor(llm=FakeLLM(VALID)).extract("no-such-doc")


def test_prompt_forbids_inventing_values():
    """The prompt is part of the contract — an extractor that invents totals
    is the silent failure this whole project is about."""
    llm = FakeLLM(ExtractedInvoice(**VALID))
    LLMExtractor(llm=llm).extract("clean-001")
    assert "do not infer or invent" in llm.last_prompt.lower()
    assert "do not compute totals" in llm.last_prompt.lower()
    assert "ACM-4471" in llm.last_prompt


def test_every_json_fixture_has_document_text():
    """Guards the exact gap that shipped in v0.1.0: the LLM path was
    documented but had no .txt inputs, so it failed instantly."""
    from invoice_agent.extract import FIXTURES

    for json_file in FIXTURES.glob("*.json"):
        assert json_file.with_suffix(".txt").exists(), f"missing text for {json_file.stem}"


def test_prompt_template_has_a_document_slot():
    assert "{document}" in EXTRACTION_PROMPT
