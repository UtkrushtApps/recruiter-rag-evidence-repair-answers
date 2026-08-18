# Solution Steps

1. Update ingestion to be privacy-safe: sanitize email/phone-like contact details from markdown text *before chunking and embedding*, so `document_chunks.chunk_text` never contains sensitive contact patterns.

2. Ensure repeated ingestion stays stable: before inserting chunks for a given `document_id`, delete any existing rows in `document_chunks` for that document. This prevents chunk duplication and preserves evidence ranking quality across re-ingests.

3. Fix metadata-scoped retrieval: in `CandidateRetriever.retrieve`, apply `filters` (at least `candidate_id` and `document_type`) in the SQL `WHERE` clause so results always come from the correct candidate’s evidence.

4. Improve evidence adequacy checks: change `has_enough_evidence` to consider similarity thresholds (e.g., at least one retrieved chunk must have `similarity >= min_similarity`), so weak queries trigger refusal behavior reliably.

5. Enforce refusal in generation: in `answer_question`, run `grounding_contract` on retrieved chunks; if evidence is weak, return a refusal-style response (`{"answer": "Insufficient evidence", "citations": []}`) without calling the generation model.

6. Prevent sensitive details from being exposed in outputs: implement `mask_sensitive_text` in `app/generation.py` to redact email/phone patterns from the final model answer, while still returning citations for grounded evidence.

7. Keep production-ready and modular: lazily import the provider SDK (`openai`) inside `answer_question` (not at module import time), continue using the retrieval pipeline for context, and avoid any hardcoded fake model calls.

