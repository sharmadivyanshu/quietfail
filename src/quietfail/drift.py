"""Content drift — signal 5.

The other five signals watch *structure*: which nodes ran, which keys came
back, how the outcome mix moved. None of them notice an agent whose shape is
perfect and whose words changed. This one does.

The default embedder is lexical, not semantic: a hashed bag of tokens. That is
a deliberate trade — it is dependency-free, offline, deterministic and free,
and it reliably catches the common case (a prompt edit that changes what the
agent says). It will NOT catch a paraphrase that means something different in
the same vocabulary. Pass a real embedder for that:

    QuietfailDrift(embedder=OpenAIEmbedder(client))

Anything with `.embed(text) -> list[float]` works.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .store import Alert

TOKEN = re.compile(r"[a-z0-9]+")
# Distance must clear both the statistical bar AND this floor. Without the
# floor, an agent whose outputs are near-identical has a near-zero standard
# deviation, and trivial wording noise looks like a 12-sigma event.
MIN_DISTANCE = 0.15
DEFAULT_SIGMA = 4.0
MIN_BASELINE_SAMPLES = 10


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class LexicalEmbedder:
    """Hashed bag-of-tokens, L2-normalised. No dependencies, no network."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector


def cosine_distance(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(0.0, min(2.0, 1.0 - sum(x * y for x, y in zip(a, b, strict=True))))


def _centroid(vectors: list[list[float]]) -> list[float]:
    dim = len(vectors[0])
    summed = [0.0] * dim
    for vector in vectors:
        for i, value in enumerate(vector):
            summed[i] += value
    norm = math.sqrt(sum(v * v for v in summed))
    return [v / norm for v in summed] if norm else summed


def build_content_profile(texts: list[str], embedder: Embedder | None = None) -> dict | None:
    """Centroid of baseline outputs plus the spread of distances around it."""
    embedder = embedder or LexicalEmbedder()
    usable = [t for t in texts if t and t.strip()]
    if len(usable) < MIN_BASELINE_SAMPLES:
        return None

    vectors = [embedder.embed(text) for text in usable]
    centroid = _centroid(vectors)
    distances = [cosine_distance(v, centroid) for v in vectors]
    mean = sum(distances) / len(distances)
    variance = sum((d - mean) ** 2 for d in distances) / len(distances)

    return {
        "centroid": centroid,
        "mean_distance": mean,
        "std_distance": math.sqrt(variance),
        "samples": len(usable),
        "embedder": type(embedder).__name__,
    }


def evaluate_content(
    text: str | None,
    profile: dict | None,
    *,
    embedder: Embedder | None = None,
    sigma: float = DEFAULT_SIGMA,
    run_id: str | None = None,
) -> list[Alert]:
    if not profile or not text or not text.strip():
        return []

    embedder = embedder or LexicalEmbedder()
    distance = cosine_distance(embedder.embed(text), profile["centroid"])
    threshold = max(
        profile["mean_distance"] + sigma * profile["std_distance"],
        MIN_DISTANCE,
    )
    if distance <= threshold:
        return []

    return [
        Alert(
            signal="content.drift",
            severity="warn",
            summary=f"output wording diverged from baseline (distance {distance:.2f})",
            detail=(
                f"baseline mean {profile['mean_distance']:.2f} "
                f"±{profile['std_distance']:.2f} over {profile['samples']} runs, "
                f"threshold {threshold:.2f}, embedder {profile['embedder']}"
            ),
            scope="run",
            run_id=run_id,
        )
    ]
