from __future__ import annotations

import re
from typing import Any

from app.retrieval import RetrievedChunk

CONTACT_RE = re.compile(
    r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|\+?\d[\d\s().-]{7,}\d)",
    re.IGNORECASE,
)


def contains_contact_info(text: str) -> bool:
    return bool(CONTACT_RE.search(text))


def has_enough_evidence(
    chunks: list[RetrievedChunk],
    min_similarity: float = 0.75,
) -> bool:
    """Evidence is considered strong only if at least one retrieved chunk
    meets the similarity threshold.

    This makes refusal behavior robust for weak queries.
    """
    if not chunks:
        return False
    best = max(chunks, key=lambda c: c.similarity)
    return best.similarity >= min_similarity


def grounding_contract(
    chunks: list[RetrievedChunk],
    min_similarity: float = 0.75,
) -> dict[str, Any]:
    if not has_enough_evidence(chunks, min_similarity=min_similarity):
        return {"answer": "Insufficient evidence", "citations": []}
    return {"answer": "Evidence available", "citations": [c.citation for c in chunks]}


def citation_labels(chunks: list[RetrievedChunk]) -> list[str]:
    return [chunk.citation for chunk in chunks if chunk.citation]
