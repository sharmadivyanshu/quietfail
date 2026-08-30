"""Content drift (signal 5) and alert sinks."""

import pytest

from quietfail.drift import (
    MIN_BASELINE_SAMPLES,
    LexicalEmbedder,
    build_content_profile,
    cosine_distance,
    evaluate_content,
)
from quietfail.sinks import fan_out, severity_filter, stdout_sink
from quietfail.store import Alert

NORMAL = [
    "Acme Office Supplies | A4 paper, 20 reams | Toner cartridges HP 26A",
    "Acme Office Supplies | A4 paper, 40 reams | Toner cartridges HP 12A",
    "Northwind Logistics | Freight charges, July | courier services",
    "Northwind Logistics | Freight charges, June | courier services",
    "Blue Ridge Software | Annual platform license renewal | consulting",
] * 4


def alert(severity: str = "warn") -> Alert:
    return Alert(signal="s", severity=severity, summary="sum", detail="det", scope="run")


# --- embedder ---------------------------------------------------------------


def test_identical_text_has_zero_distance():
    embedder = LexicalEmbedder()
    vector = embedder.embed("freight charges july")
    assert cosine_distance(vector, vector) == pytest.approx(0.0, abs=1e-9)


def test_word_order_does_not_matter():
    """Bag of tokens — this is a stated limitation, so pin it."""
    embedder = LexicalEmbedder()
    a = embedder.embed("freight charges july")
    b = embedder.embed("july charges freight")
    assert cosine_distance(a, b) == pytest.approx(0.0, abs=1e-9)


def test_different_text_is_distant():
    embedder = LexicalEmbedder()
    a = embedder.embed("freight charges july courier")
    b = embedder.embed("annual platform license renewal")
    assert cosine_distance(a, b) > 0.9


def test_embedding_is_normalised():
    vector = LexicalEmbedder().embed("some invoice text here")
    assert sum(v * v for v in vector) == pytest.approx(1.0, abs=1e-9)


def test_empty_text_embeds_without_dividing_by_zero():
    assert LexicalEmbedder().embed("") == [0.0] * 256


# --- profile ----------------------------------------------------------------


def test_profile_needs_enough_samples():
    assert build_content_profile(["text"] * (MIN_BASELINE_SAMPLES - 1)) is None


def test_profile_ignores_blank_texts():
    assert build_content_profile([""] * 50) is None


def test_profile_records_spread():
    profile = build_content_profile(NORMAL)
    assert profile is not None
    assert profile["samples"] == len(NORMAL)
    assert profile["std_distance"] >= 0.0
    assert profile["embedder"] == "LexicalEmbedder"


# --- evaluation -------------------------------------------------------------


def test_familiar_text_is_quiet():
    profile = build_content_profile(NORMAL)
    assert evaluate_content(NORMAL[0], profile) == []


def test_garbled_text_is_flagged():
    """The OCR-degradation case: same structure, mangled words."""
    profile = build_content_profile(NORMAL)
    garbled = "@cm3  0ff!c3  5upp1!35 | @4  p@p3r,  20  r3@m5"
    signals = {a.signal for a in evaluate_content(garbled, profile)}
    assert "content.drift" in signals


def test_no_profile_means_no_alerts():
    assert evaluate_content("anything at all", None) == []


def test_missing_text_means_no_alerts():
    profile = build_content_profile(NORMAL)
    assert evaluate_content(None, profile) == []
    assert evaluate_content("   ", profile) == []


def test_distance_floor_prevents_zero_variance_false_positives():
    """With identical baseline outputs the std is 0, so any wording change is
    infinitely many sigmas out. The floor is what stops that firing."""
    profile = build_content_profile(["exactly the same text"] * 20)
    assert profile["std_distance"] == pytest.approx(0.0, abs=1e-9)
    assert evaluate_content("exactly the same text", profile) == []


# --- sinks ------------------------------------------------------------------


def test_fan_out_reaches_every_sink():
    seen = []
    fan_out(seen.append, seen.append)(alert())
    assert len(seen) == 2


def test_one_broken_sink_does_not_block_the_others():
    def broken(_):
        raise RuntimeError("slack is down")

    seen = []
    fan_out(broken, seen.append)(alert())
    assert len(seen) == 1


def test_severity_filter_drops_below_threshold():
    seen = []
    sink = severity_filter(seen.append, minimum="critical")
    sink(alert("warn"))
    sink(alert("critical"))
    assert [a.severity for a in seen] == ["critical"]


def test_stdout_sink_writes_the_alert(capsys):
    stdout_sink()(alert())
    assert "s: sum" in capsys.readouterr().err
