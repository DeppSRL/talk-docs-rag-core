"""C1 — pipeline di ingest: parsing → chunking token-based → metadati+hash → embed →
Chroma (collection retrieval) + Whoosh (keyword IT), e scrittura di ``manifest.json``.

Versioning (base dell'invalidazione cache C4):
- ``content_hash`` per file  = sha256 del testo estratto;
- ``corpus_content_hash``     = sha256 su (relpath, content_hash) ordinati;
- ``corpus_version``          = sha256 su corpus_content_hash + parametri di chunking +
  modello embedding. Cambiare corpus **o** parametri di chunking/embedding cambia la
  versione → la cache semantica si invalida (le risposte cambierebbero).

Idempotenza: chunk_id deterministici (``{doc_id}::{i}``) e ricostruzione pulita delle
collection ad ogni run → a corpus e parametri invariati il ``corpus_version`` è identico.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from config import REPO_ROOT, RagConfig
from vendor.talkdocs.models.document import Document, DocumentChunk, DocumentMetadata
from vendor.talkdocs.utils.text_processing import clean_text

from .chunking import chunk_document, count_tokens
from .parsers import SUPPORTED_SUFFIXES, parse_file

EMBED_BATCH = 64


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class IngestReport:
    corpus_version: str
    corpus_content_hash: str
    n_files: int
    n_chunks: int
    manifest_path: str


def _discover_files(corpus_dir: Path, card_dir: Path | None = None) -> list[Path]:
    """File del corpus, **esclusa la scheda**.

    La scheda vive dentro `corpus/` per stare accanto ai documenti che descrive, e i suoi
    file sono `.md`: senza questa esclusione l'ingest la indicizza come se fosse una
    fonte. Misurato il 2026-08-08: tre documenti in più nell'indice
    (`delibere/card/*.md`) in tutte le run dell'8 agosto.

    Non è un problema di conteggi — lo `StructuredStore` filtra su `is_delibera` e la
    scheda non ha un codice delibera — ma di **natura di ciò che si può citare**: la
    scheda è testo nostro, che descrive che cosa il sistema sa fare. Recuperata come
    passaggio diventerebbe una fonte citabile, e una risposta potrebbe sostenere
    un'affermazione sul corpus citando ciò che abbiamo scritto noi sul corpus. Su un
    prodotto di accountability è il genere di circolarità che squalifica una citazione.
    """
    escluse = []
    if card_dir is not None and card_dir.is_dir():
        escluse.append(card_dir.resolve())
    files = [
        p
        for p in sorted(corpus_dir.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and p.name != "README.md"
        and not any(d in p.resolve().parents for d in escluse)
    ]
    return files


async def _embed_all(embedding_service, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        out.extend(await embedding_service.get_embeddings(batch))
    return out


async def run_ingest(cfg: RagConfig, corpus_dir: Path | None = None) -> IngestReport:
    from app.wiring import (
        build_chroma_client,
        build_embedding_service,
        build_retrieval_store,
        build_whoosh,
    )

    corpus_dir = corpus_dir or (REPO_ROOT / "corpus")
    card_dir = Path(cfg.corpus_card_dir)
    if not card_dir.is_absolute():
        card_dir = REPO_ROOT / card_dir
    files = _discover_files(corpus_dir, card_dir)
    if not files:
        raise SystemExit(
            f"Nessun file supportato in {corpus_dir} ({sorted(SUPPORTED_SUFFIXES)}). "
            "Il corpus pubblico è un gate di M3 (vedi piano)."
        )

    embedding_service = build_embedding_service(cfg)
    chroma_client = build_chroma_client(cfg)

    # Ricostruzione pulita delle collection (idempotenza dello stato dello store).
    for name in (cfg.chroma_collection_retrieval,):
        try:
            chroma_client.delete_collection(name)
        except Exception:
            pass
    retrieval_store = build_retrieval_store(cfg, embedding_service, chroma_client)

    whoosh = await build_whoosh(cfg)
    await whoosh.clear_index()

    manifest_files = []
    all_documents: list[Document] = []
    total_chunks = 0

    for path in files:
        rel = path.relative_to(corpus_dir).as_posix()
        raw_text, title = parse_file(path)
        text = clean_text(raw_text)
        content_hash = _sha256(text)
        doc_id = rel  # stabile e leggibile

        raw_chunks = chunk_document(text, cfg.chunk_tokens, cfg.chunk_overlap_tokens)
        stat = path.stat()

        chunks: list[DocumentChunk] = []
        for i, rc in enumerate(raw_chunks):
            chunk_id = f"{doc_id}::{i}"
            chunk_hash = _sha256(rc.content)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=rc.content,
                    start_index=rc.start_token,
                    end_index=rc.end_token,
                    metadata={
                        "doc_id": doc_id,
                        "title": title,
                        "section": rc.section or "",
                        "source": rel,
                        "chunk_index": i,
                        "content_hash": chunk_hash,
                        "n_tokens": count_tokens(rc.content),
                    },
                )
            )

        metadata = DocumentMetadata(
            source_id=doc_id,
            name=title,
            path=rel,
            size=stat.st_size,
            mime_type=path.suffix.lower().lstrip("."),
            author="corpus-pubblico",
            created_date=datetime.fromtimestamp(stat.st_ctime, tz=UTC),
            modified_date=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            hash=content_hash,
        )
        all_documents.append(Document(source_id=doc_id, content=text, metadata=metadata, chunks=chunks))
        total_chunks += len(chunks)
        manifest_files.append(
            {"path": rel, "title": title, "content_hash": content_hash, "n_chunks": len(chunks)}
        )
        print(f"  · {rel}: {len(chunks)} chunk (titolo: {title!r})")

    # Embedding di tutti i chunk (batch) → attach.
    flat_chunks = [c for d in all_documents for c in d.chunks]
    embeddings = await _embed_all(embedding_service, [c.content for c in flat_chunks])
    for c, emb in zip(flat_chunks, embeddings, strict=True):
        c.embedding = emb

    # Indicizza in Chroma (denso) + Whoosh (keyword IT). Whoosh in batch: un solo writer
    # e un solo commit — il per-documento del vendored costa 10-20 min su 13.670 chunk
    # (segmento per commit, merge crescenti) e si ripagherebbe a ogni run di CI.
    await retrieval_store.add_documents(all_documents)
    await whoosh.index_documents(
        [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "source": c.metadata["source"],
                "metadata": c.metadata,
            }
            for c in flat_chunks
        ]
    )
    await whoosh.close()

    # Document frequency per il guardiano di astensione (C3b). Calcolata qui, sui chunk che
    # abbiamo già in memoria: farlo all'avvio della pipeline costerebbe una passata su tutto
    # il corpus a ogni `ask`.
    if cfg.abstention_idf_threshold > 0:
        from rag.guard import TermStats

        TermStats.from_documents([c.content for c in flat_chunks]).save(cfg.term_df_path)
        print(f"[ingest] statistiche IDF → {cfg.term_df_path}")

    # Indice delle frasi ricorrenti: il materiale della nota di provenienza. Si costruisce
    # qui perché qui i testi interi sono già in memoria — e perché il conteggio che serve è
    # per DOCUMENTO, che a valle, sui chunk, non sarebbe più ricostruibile.
    if cfg.provenienza_enabled:
        from ingest.frasi import costruisci_indice

        indice = costruisci_indice(
            [(d.source_id, d.content) for d in all_documents], soglia=cfg.frase_min_documenti
        )
        percorso = indice.salva(cfg.frasi_index_path)
        print(
            f"[ingest] frasi ricorrenti (≥{cfg.frase_min_documenti} documenti): "
            f"{len(indice.frasi)} frasi, {len(indice.norme)} norme → {percorso}"
        )

    # Versioning.
    corpus_content_hash = _sha256(
        "\n".join(f"{f['path']}:{f['content_hash']}" for f in sorted(manifest_files, key=lambda x: x["path"]))
    )
    version_material = json.dumps(
        {
            "corpus_content_hash": corpus_content_hash,
            "chunk_tokens": cfg.chunk_tokens,
            "chunk_overlap_ratio": cfg.chunk_overlap_ratio,
            "embed_model": cfg.mistral_embed_model,
            "embed_dim": cfg.embed_dim,
        },
        sort_keys=True,
    )
    corpus_version = _sha256(version_material)

    manifest = {
        "corpus_version": corpus_version,
        "corpus_content_hash": corpus_content_hash,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "params": {
            "chunk_tokens": cfg.chunk_tokens,
            "chunk_overlap_ratio": cfg.chunk_overlap_ratio,
            "embed_model": cfg.mistral_embed_model,
            "embed_dim": cfg.embed_dim,
        },
        "n_files": len(manifest_files),
        "n_chunks": total_chunks,
        "files": sorted(manifest_files, key=lambda x: x["path"]),
    }
    manifest_path = corpus_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return IngestReport(
        corpus_version=corpus_version,
        corpus_content_hash=corpus_content_hash,
        n_files=len(manifest_files),
        n_chunks=total_chunks,
        manifest_path=str(manifest_path),
    )


def main() -> int:
    cfg = RagConfig.from_env()
    print(f"[ingest] corpus → Chroma '{cfg.chroma_collection_retrieval}' + Whoosh '{cfg.whoosh_index_dir}'")
    print(
        f"[ingest] chunk_tokens={cfg.chunk_tokens} overlap={cfg.chunk_overlap_tokens} "
        f"embed={cfg.mistral_embed_model}"
    )
    report = asyncio.run(run_ingest(cfg))
    print("\n=== manifest ===")
    print(f"  file           : {report.n_files}")
    print(f"  chunk          : {report.n_chunks}")
    print(f"  corpus_content : {report.corpus_content_hash}")
    print(f"  corpus_version : {report.corpus_version}")
    print(f"  manifest       : {report.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
