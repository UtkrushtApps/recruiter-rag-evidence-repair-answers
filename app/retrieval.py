from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.db import fetch_all, vector_literal
from app.embeddings import EmbeddingService


@dataclass(frozen=True)
class RetrievedChunk:
    id: int
    text: str
    similarity: float
    source_name: str
    citation: str
    candidate_id: str | None
    candidate_name: str | None
    document_type: str
    section_title: str | None


class CandidateRetriever:
    def __init__(self, embedder: EmbeddingService | None = None) -> None:
        self.embedder = embedder or EmbeddingService()

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        filters = filters or {}
        query_vector = vector_literal(self.embedder.embed_query(query))

        where_clauses: list[str] = []
        params: dict[str, Any] = {
            "query_vec": query_vector,
            "top_k": top_k,
        }

        if filters.get("candidate_id"):
            where_clauses.append("candidate_id = %(candidate_id)s")
            params["candidate_id"] = filters["candidate_id"]

        if filters.get("document_type"):
            where_clauses.append("document_type = %(document_type)s")
            params["document_type"] = filters["document_type"]

        where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"

        rows = fetch_all(
            f"""
            SELECT
                id,
                chunk_text,
                source_name,
                citation_label,
                candidate_id,
                candidate_name,
                document_type,
                section_title,
                1 - (embedding <=> %(query_vec)s::vector) AS similarity
            FROM document_chunks
            WHERE {where_sql}
            ORDER BY embedding <=> %(query_vec)s::vector
            LIMIT %(top_k)s
            """,
            params,
        )

        return [
            RetrievedChunk(
                id=int(row["id"]),
                text=row["chunk_text"],
                similarity=float(row["similarity"]),
                source_name=row["source_name"],
                citation=row["citation_label"],
                candidate_id=row["candidate_id"],
                candidate_name=row["candidate_name"],
                document_type=row["document_type"],
                section_title=row["section_title"],
            )
            for row in rows
        ]


def context_for_prompt(chunks: list[RetrievedChunk], token_budget: int = 900) -> str:
    lines: list[str] = []
    used = 0
    for chunk in chunks:
        estimate = max(1, len(chunk.text.split()))
        if used + estimate > token_budget:
            break
        used += estimate
        lines.append(f"[{chunk.citation}] {chunk.text}")
    return "\n\n".join(lines)
