from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from app.chunker import split_text
from app.config import get_settings
from app.db import connection, vector_literal
from app.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

# Privacy: avoid storing recruiter-irrelevant contact details in chunk_text.
CONTACT_RE = re.compile(
    r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|\+?\d[\d\s().-]{7,}\d)",
    re.IGNORECASE,
)


def _sanitize_sensitive_text(text: str) -> str:
    """Remove emails/phone-like contact details from persisted chunks."""
    return CONTACT_RE.sub("[REDACTED_CONTACT]", text)


def ingest_documents(paths: list[Path] | None = None) -> int:
    settings = get_settings()
    document_paths = paths or sorted(settings.data_dir.glob("*.md"))
    embedder = EmbeddingService()
    inserted = 0

    for path in document_paths:
        try:
            metadata, body = read_markdown_document(path)
            # Sanitize before chunking/embedding so the store is privacy-safe.
            normalized_body = normalize_text(body)
            sanitized_body = _sanitize_sensitive_text(normalized_body)

            chunks = split_text(sanitized_body)
            embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
            document_id = upsert_source_document(path, metadata, body)
            inserted += insert_chunks(document_id, path.name, metadata, chunks, embeddings)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            logger.warning(
                "source ingestion failed",
                extra={"file_name": str(path), "reason": str(exc)},
            )
    return inserted


def read_markdown_document(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("source document is empty")
    if not text.startswith("---"):
        return {"document_type": "unknown"}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {"document_type": "unknown"}, text

    metadata: dict[str, Any] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata, parts[2].strip()


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def upsert_source_document(path: Path, metadata: dict[str, Any], body: str) -> int:
    content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO source_documents (
                    source_path, source_name, document_type, candidate_id,
                    candidate_name, source_version, raw_metadata, content_sha256,
                    ingested_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, now())
                ON CONFLICT (source_path) DO UPDATE SET
                    source_name = EXCLUDED.source_name,
                    document_type = EXCLUDED.document_type,
                    candidate_id = EXCLUDED.candidate_id,
                    candidate_name = EXCLUDED.candidate_name,
                    source_version = EXCLUDED.source_version,
                    raw_metadata = EXCLUDED.raw_metadata,
                    content_sha256 = EXCLUDED.content_sha256,
                    ingested_at = now()
                RETURNING id
                """,
                (
                    str(path),
                    path.name,
                    metadata.get("document_type", "unknown"),
                    metadata.get("candidate_id"),
                    metadata.get("candidate_name"),
                    metadata.get("source_version"),
                    __import__("json").dumps(metadata),
                    content_hash,
                ),
            )
            row = cur.fetchone()
            return int(row["id"])


def insert_chunks(
    document_id: int,
    source_name: str,
    metadata: dict[str, Any],
    chunks,
    embeddings: list[list[float]],
) -> int:
    """Insert chunks for a document.

    To keep repeated ingestion stable, we delete existing chunks for the
    document_id first (so chunk_count doesn't grow on re-ingest).
    """
    count = 0
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
            for chunk, embedding in zip(chunks, embeddings, strict=False):
                citation = f"{source_name}#chunk-{chunk.index}"
                cur.execute(
                    """
                    INSERT INTO document_chunks (
                        document_id, source_name, chunk_index, chunk_text,
                        candidate_id, candidate_name, document_type,
                        section_title, citation_label, token_estimate, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    (
                        document_id,
                        source_name,
                        chunk.index,
                        chunk.text,
                        metadata.get("candidate_id"),
                        metadata.get("candidate_name"),
                        metadata.get("document_type", "unknown"),
                        chunk.section_title,
                        citation,
                        chunk.token_estimate,
                        vector_literal(embedding),
                    ),
                )
                count += 1
    return count
