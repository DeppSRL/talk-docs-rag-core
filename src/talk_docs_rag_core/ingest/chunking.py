"""C1 — chunking token-based ricorsivo per struttura (via tiktoken).

Il chunking lo possiede l'ingest (spec §3): il ``chunk_text`` char-based di talk-docs
**non** si riusa. Qui:

1. il testo è spezzato in *blocchi* per struttura (heading markdown → paragrafi);
2. i blocchi sono impacchettati fino a ``chunk_tokens`` con overlap ``overlap_tokens``;
3. un blocco più grande di ``chunk_tokens`` è a sua volta spezzato per finestra di token.

Ogni chunk porta: testo, indici di token (offset nel documento), sezione (ultimo
heading visto) e verrà completato a valle con ``doc_id``/hash nella pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")

# Heading markdown ("# ...", "## ...") o ATX; usato per la "sezione" del chunk.
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass
class Chunk:
    """Chunk pronto per l'indicizzazione (prima di embed/hash)."""

    content: str
    start_token: int
    end_token: int
    section: str | None = None
    metadata: dict = field(default_factory=dict)


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _split_into_blocks(text: str) -> list[tuple[str, str | None]]:
    """Spezza per struttura. Ritorna lista di (blocco, sezione corrente)."""
    lines = text.splitlines()
    blocks: list[tuple[str, str | None]] = []
    current_section: str | None = None
    buf: list[str] = []

    def flush():
        if buf:
            joined = "\n".join(buf).strip()
            if joined:
                blocks.append((joined, current_section))
            buf.clear()

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            # nuovo heading: chiude il blocco corrente e diventa la sezione
            flush()
            current_section = m.group(2).strip()
            blocks.append((line.strip(), current_section))
            continue
        if line.strip() == "":
            flush()  # riga vuota = confine di paragrafo
        else:
            buf.append(line)
    flush()
    return blocks


def _split_long_block(content: str, section: str | None, chunk_tokens: int, overlap_tokens: int) -> list[Chunk]:
    """Spezza un blocco troppo grande per finestra di token con overlap."""
    tokens = _ENC.encode(content)
    chunks: list[Chunk] = []
    step = max(1, chunk_tokens - overlap_tokens)
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_tokens]
        if not window:
            break
        chunks.append(
            Chunk(
                content=_ENC.decode(window).strip(),
                start_token=start,
                end_token=start + len(window),
                section=section,
            )
        )
        if start + chunk_tokens >= len(tokens):
            break
    return chunks


def chunk_document(text: str, chunk_tokens: int, overlap_tokens: int) -> list[Chunk]:
    """Chunking ricorsivo per struttura con packing a ``chunk_tokens`` e overlap."""
    if not text or not text.strip():
        return []

    blocks = _split_into_blocks(text)
    chunks: list[Chunk] = []

    buf_tokens: list[int] = []
    buf_section: str | None = None
    doc_cursor = 0  # posizione (in token) nel documento, per gli offset

    def emit(tokens: list[int], section: str | None, start_at: int):
        content = _ENC.decode(tokens).strip()
        if content:
            chunks.append(
                Chunk(
                    content=content,
                    start_token=start_at,
                    end_token=start_at + len(tokens),
                    section=section,
                )
            )

    buf_start = 0
    for block, section in blocks:
        block_tokens = _ENC.encode(block)

        # blocco singolo troppo grande → spezzalo da solo
        if len(block_tokens) > chunk_tokens:
            if buf_tokens:
                emit(buf_tokens, buf_section, buf_start)
                doc_cursor += len(buf_tokens)
                buf_tokens = []
            sub = _split_long_block(block, section, chunk_tokens, overlap_tokens)
            for c in sub:
                c.start_token += doc_cursor
                c.end_token += doc_cursor
            chunks.extend(sub)
            doc_cursor += len(block_tokens)
            buf_start = doc_cursor
            buf_section = None
            continue

        # non entra nel buffer corrente → emetti e riparti con overlap
        if buf_tokens and len(buf_tokens) + 1 + len(block_tokens) > chunk_tokens:
            emit(buf_tokens, buf_section, buf_start)
            doc_cursor_before = buf_start
            # overlap: coda del buffer appena emesso
            tail = buf_tokens[-overlap_tokens:] if overlap_tokens > 0 else []
            buf_start = doc_cursor_before + len(buf_tokens) - len(tail)
            buf_tokens = list(tail)
            buf_section = section

        if not buf_tokens:
            buf_section = section
        # separatore fra blocchi
        sep = _ENC.encode("\n\n") if buf_tokens else []
        buf_tokens = buf_tokens + sep + block_tokens

    if buf_tokens:
        emit(buf_tokens, buf_section, buf_start)

    return chunks
