from __future__ import annotations

import re
from typing import Any

from app.config import get_settings
from app.evaluation import grounding_contract
from app.retrieval import CandidateRetriever, RetrievedChunk, context_for_prompt

SYSTEM_PROMPT = """You answer recruiter questions using only the provided candidate evidence.
Cite the evidence labels that support your answer. If the evidence is not enough,
say that there is insufficient evidence rather than guessing.
"""

# Privacy: ensure contact details aren't exposed in outputs.
CONTACT_RE = re.compile(
    r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|\+?\d[\d\s().-]{7,}\d)",
    re.IGNORECASE,
)


def mask_sensitive_text(text: str) -> str:
    return CONTACT_RE.sub("[REDACTED_CONTACT]", text)


def build_public_payload(answer: str, chunks: list[RetrievedChunk]) -> dict[str, Any]:
    return {
        "answer": mask_sensitive_text(answer),
        "citations": [chunk.citation for chunk in chunks],
    }


def answer_question(
    question: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("A provider key is required for end-to-end generation")

    retriever = CandidateRetriever()
    chunks = retriever.retrieve(question, filters=filters, top_k=top_k)

    # Refusal-style response when evidence is weak.
    contract = grounding_contract(chunks, min_similarity=0.8)
    if contract["citations"] == []:
        return {"answer": mask_sensitive_text(contract["answer"]), "citations": []}

    context = context_for_prompt(chunks)

    # Lazy import to keep evaluation/unit tests runnable without the provider lib.
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Evidence:\n{context}\n\nQuestion: {question}",
            },
        ],
    )

    answer = response.choices[0].message.content or ""
    return build_public_payload(answer, chunks)
