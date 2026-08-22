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

from talk_docs_rag_core.config import RagConfig
from talk_docs_rag_core.retrieval.models.document import Document, DocumentChunk, DocumentMetadata
from talk_docs_rag_core.retrieval.utils.text_processing import clean_text

from .chunking import chunk_document, count_tokens
from .parsers import SUPPORTED_SUFFIXES, parse_file

EMBED_BATCH = 64


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class StatoIngest:
    """Ciò che il chiamante può sapere di una run **anche quando fallisce**.

    `IngestReport` torna solo se l'ingest arriva alla fine. Ma la domanda che conta dopo un
    fallimento è un'altra: *lo store è stato toccato?* Da quando le scritture su Chroma sono a
    lotti, un guasto a metà lascia una collection parziale, e chi serve le risposte deve
    svuotarla — mentre un guasto **prima** della prima scrittura lascia l'indice precedente
    intatto, e svuotarlo lì significa cancellare un indice funzionante.

    Prima questo si deduceva dal flusso di controllo del chiamante («ho chiamato `run_ingest`,
    quindi assumo che abbia toccato lo store»), e l'inferenza era sbagliata ai due estremi: il
    corpus senza file supportati alza `SystemExit` prima di ogni scrittura, e un guasto dopo la
    fine non ha nulla di parziale da svuotare. Due modi di cancellare 511 documenti buoni.

    Si passa a `run_ingest` e si legge dopo, anche dal ramo `except`:

        stato = StatoIngest()
        try:
            rep = await run_ingest(cfg, corpus_dir=dir, stato=stato)
        except Exception:
            if stato.store_toccato:
                ...  # la collection è parziale: svuotala

    Perché un oggetto mutabile e non un valore di ritorno: `run_ingest` **alza**, e un valore
    di ritorno non arriva a chi gestisce l'eccezione. Perché non un'eccezione dedicata: il
    consumatore distingue già famiglie di guasti (fonte, `SystemExit`, resto) e incapsularle
    tutte gli farebbe perdere quella distinzione.
    """

    store_toccato: bool = False
    """`True` da quando lo store può essere stato alterato — cioè dalla cancellazione della
    collection in avanti, compresa la cancellazione stessa. Non «da quando la scrittura è
    riuscita»: una `delete_collection` interrotta a metà è già un'alterazione."""


@dataclass
class IngestReport:
    corpus_version: str
    corpus_content_hash: str
    n_files: int
    n_chunks: int
    manifest_path: str
    # Costo dell'indicizzazione, misurato e non stimato. `embed_cost` è `None` quando il
    # prezzo non è configurato (`PRICE_EMBED_PER_MTOK`): meglio un campo vuoto che uno zero,
    # che si legge come «gratis» invece che come «non lo sappiamo».
    embed_usage: dict[str, int] | None = None
    embed_cost: float | None = None


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


async def _embedda_e_scrivi_a_lotti(embedding_service, store, chunks: list, metadata_json: dict) -> int:
    """Embedda un lotto, lo scrive, **libera i vettori**. Poi il lotto successivo.

    È il rimedio al difetto della memoria dell'ingest, e la forma è tutta qui: prima i
    vettori di tutto il corpus dovevano sopravvivere fino all'unica chiamata
    `add_documents` finale — 18.396 chunk × 1024 float distinti, centinaia di MiB che
    nessuno rilegge più. Ora nessun vettore vive più a lungo del lotto che l'ha prodotto.

    **I lotti di embedding sono gli stessi di prima** (`EMBED_BATCH`, nello stesso ordine):
    stesse chiamate al provider, stessi token, stessi vettori. Cambia solo *quando* si
    scrive — e quindi l'ordine di inserimento in Chroma, che per una ricerca per similarità
    non è un ordine di cui qualcosa dipenda.

    Ciò che il picco NON perde: i testi. Il calcolo dell'IDF e l'indice delle frasi
    ricorrenti rileggono i contenuti dopo l'indicizzazione, e il conteggio delle frasi è
    **per documento** — a valle, sui chunk, non sarebbe più ricostruibile. Quindi i testi
    restano in memoria per costruzione, e sono il pavimento sotto cui questa funzione non
    può scendere. Dirlo qui perché la prossima misura non venga letta come una regressione.
    """
    n = 0
    for i in range(0, len(chunks), EMBED_BATCH):
        lotto = chunks[i : i + EMBED_BATCH]
        vettori = await embedding_service.get_embeddings([c.content for c in lotto])
        for c, emb in zip(lotto, vettori, strict=True):
            c.embedding = emb
        await store.add_chunks(lotto, metadata_json)
        # Il punto di tutto l'esercizio: da qui il vettore non serve più a nessuno.
        for c in lotto:
            c.embedding = None
        n += len(lotto)
    return n


async def run_ingest(
    cfg: RagConfig, corpus_dir: Path | None = None, stato: StatoIngest | None = None
) -> IngestReport:
    """Costruisce l'indice da un corpus su disco. Vedi `StatoIngest` per il parametro `stato`.

    `stato` è facoltativo e retrocompatibile: chi non lo passa ottiene esattamente il
    comportamento di prima. Chi lo passa può sapere, dal ramo `except`, se questa run ha
    toccato lo store — l'unica cosa che autorizza a svuotare una collection.
    """
    from talk_docs_rag_core.wiring import (
        build_chroma_client,
        build_embedding_service,
        build_retrieval_store,
        build_whoosh,
    )

    corpus_dir = Path(corpus_dir) if corpus_dir is not None else Path(cfg.corpus_dir)
    card_dir = Path(cfg.corpus_card_dir)
    files = _discover_files(corpus_dir, card_dir)
    if not files:
        raise SystemExit(
            f"Nessun file supportato in {corpus_dir} ({sorted(SUPPORTED_SUFFIXES)}). "
            "Il corpus pubblico è un gate di M3 (vedi piano)."
        )

    embedding_service = build_embedding_service(cfg)
    chroma_client = build_chroma_client(cfg)

    # Da QUI in avanti lo store può essere alterato, e il chiamante deve poterlo sapere anche
    # se la riga successiva alza: la cancellazione stessa è un'alterazione, quindi il flag si
    # scrive **prima** e non dopo. Tutto ciò che sta sopra — discovery dei file, costruzione del
    # servizio di embedding, del client Chroma — fallisce lasciando l'indice precedente intatto.
    if stato is not None:
        stato.store_toccato = True

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

    # Embedding e scrittura su Chroma A LOTTI, non in due fasi.
    #
    # Le due fasi (embedda tutto → scrivi tutto) obbligavano i vettori dell'intero corpus a
    # coesistere: è il difetto della memoria dell'ingest. Qui ogni lotto viene scritto e
    # lasciato andare. I metadati del documento si serializzano UNA volta per documento,
    # perché un lotto attraversa più documenti.
    flat_chunks = [c for d in all_documents for c in d.chunks]
    metadata_json = {d.source_id: d.metadata.model_dump_json() for d in all_documents}
    await _embedda_e_scrivi_a_lotti(embedding_service, retrieval_store, flat_chunks, metadata_json)

    # Whoosh in batch: un solo writer e un solo commit — il per-documento del vendored costa
    # 10-20 min su 13.670 chunk (segmento per commit, merge crescenti) e si ripagherebbe a
    # ogni run di CI. Non tocca gli embedding, quindi resta dov'era.
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
        from talk_docs_rag_core.rag.guard import TermStats

        TermStats.from_documents([c.content for c in flat_chunks]).save(cfg.term_df_path)
        print(f"[ingest] statistiche IDF → {cfg.term_df_path}")

    # Indice delle frasi ricorrenti: il materiale della nota di provenienza. Si costruisce
    # qui perché qui i testi interi sono già in memoria — e perché il conteggio che serve è
    # per DOCUMENTO, che a valle, sui chunk, non sarebbe più ricostruibile.
    if cfg.provenienza_enabled:
        from talk_docs_rag_core.ingest.frasi import costruisci_indice

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

    # Niente timestamp qui: il manifest è committato dal consumatore e rigenerato a ogni
    # ingest → deve essere idempotente anche a livello di git (nessun diff a corpus e
    # parametri invariati). La provenienza temporale delle run vive nelle tuple di audit.
    manifest = {
        "corpus_version": corpus_version,
        "corpus_content_hash": corpus_content_hash,
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

    embed_usage = dict(getattr(embedding_service, "usage", {}) or {})
    tok = embed_usage.get("total_tokens") or embed_usage.get("prompt_tokens") or 0
    embed_cost = (tok / 1_000_000) * cfg.price_embed_per_mtok if (tok and cfg.price_embed_per_mtok) else None

    return IngestReport(
        corpus_version=corpus_version,
        corpus_content_hash=corpus_content_hash,
        n_files=len(manifest_files),
        n_chunks=total_chunks,
        manifest_path=str(manifest_path),
        embed_usage=embed_usage or None,
        embed_cost=embed_cost,
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
    if report.embed_usage:
        u = report.embed_usage
        costo = f"{report.embed_cost:.6f} $" if report.embed_cost is not None else "prezzo non configurato"
        print(f"  embedding      : {u.get('calls', 0)} chiamate · {u.get('total_tokens', 0)} token · {costo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
