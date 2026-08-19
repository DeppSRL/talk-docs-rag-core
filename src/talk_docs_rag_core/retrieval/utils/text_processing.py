"""Utility di testo vendorizzate da talk-docs.

NOTA VENDORING (spec §3): ``clean_text`` è riusato. ``chunk_text`` (char-based) **non**
è riusato dal PoC: il chunking lo possiede l'ingest (C1, token-based via tiktoken). Lo
teniamo qui solo per completezza del modulo, ma nessun componente del PoC lo importa.
"""

import uuid

from ..models.document import DocumentChunk


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200, source_id: str = "") -> list[DocumentChunk]:
    """Split text into chunks with overlap.

    NON usato dal PoC (vedi nota di modulo). Superato dal chunking token-based di C1.
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk_content = text[start:end].strip()

        if chunk_content:
            chunk = DocumentChunk(
                chunk_id=f"{source_id}_{uuid.uuid4().hex[:8]}",
                content=chunk_content,
                start_index=start,
                end_index=end,
                metadata={"source_id": source_id},
            )
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + 1

        start = next_start

    return chunks


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""

    text = text.strip()

    import re

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

    return text
