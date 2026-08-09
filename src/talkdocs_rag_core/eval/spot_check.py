"""Spot-check: scheda di lettura + modulo di giudizio per una run di eval.

Perché serve (spec §1): le metriche automatiche del report A/B misurano se il modello
**cita**, non se ha **ragione**. `citation_validity` verifica solo che i marcatori `[n]`
puntino a passaggi in range; nessuno controlla che il passaggio citato sostenga davvero
l'affermazione. È esattamente lì che un RAG su atti amministrativi sbaglia in modo
pericoloso: risposta sicura, ben citata, numero preso da un passaggio vicino ma diverso
(misurato su ``eval-20260805T072128Z``: ``ic-07`` risponde 295.178.000 €, la sua
riformulazione ``ic-07-bis`` risponde 32,5 milioni € — e il report le conta entrambe valide).

Produce due file, deliberatamente separati:

- ``<run>-spotcheck[-cond].md``   la **scheda**: ogni affermazione accanto al testo
  integrale del passaggio citato. Si legge, non si compila.
- ``<run>-giudizi[-cond].csv``    il **modulo**: una riga per item da giudicare, tre
  colonne da riempire. Si compila (LibreOffice va bene), poi ``app eval-human`` lo rilegge.

Il modulo NON sta nel CSV del report: quello è rigenerato a ogni run, e colonne di
giudizio in un file rigenerabile si perdono alla run successiva.

Uso:
    uv run python -m app spot-check --run eval-20260805T072128Z --condition off
"""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

from config import RagConfig

# Colonne che l'umano compila. Fedeltà è BINARIA: «l'affermazione è nel passaggio citato»
# è sì o no, e il criterio di decisione ha bisogno di un tasso, quindi di un binario.
#
# `causa` si compila solo quando fedele=no, e vale `retrieval` o `generazione`: è la
# distinzione da cui dipende dove investire. Con la scheda che mostra TUTTI i passaggi del
# contesto è una domanda a risposta oggettiva — «la risposta è in uno di questi cinque?».
COLONNE_GIUDIZIO = ["fedele", "causa", "citazione_corretta", "italiano_1_5", "note"]
# `ereditato_da` NON è una colonna di giudizio: non la compila l'umano, la scrive il
# sistema quando riporta avanti un giudizio dato su una run precedente per una risposta
# **identica** (vedi `app.judge`). Serve a rendere visibile la differenza fra un tasso di
# fedeltà appena misurato e uno per la maggior parte ereditato: sono due affermazioni
# diverse, e senza questa colonna la seconda si spaccerebbe per la prima.
COLONNE_FORM = [
    "run_id", "condition", "id", "categoria", "tipo", "domanda",
    *COLONNE_GIUDIZIO, "ereditato_da", "risposta",
]

CAUSE_VALIDE = {"retrieval", "generazione"}


def _load_audit(run_id: str, audit_dir: str) -> list[dict]:
    path = Path(audit_dir) / f"{run_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"tupla di audit non trovata: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _index_eval_set(eval_set_path: str) -> dict[str, tuple[str, str]]:
    """``domanda → (id, categoria)``. L'audit registra la query verbatim, quindi il join
    sul testo è esatto; gli id vivono solo nell'eval set."""
    path = Path(eval_set_path)
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["question"]] = (d["id"], d["category"])
    return out


def _fetch_chunks(cfg: RagConfig, chunk_ids: list[str]) -> dict[str, dict]:
    """Testo + metadati dei chunk citati, per id. Il chunk_id È l'id del record Chroma."""
    if not chunk_ids:
        return {}
    from app.wiring import build_chroma_client

    client = build_chroma_client(cfg)
    col = client.get_or_create_collection(cfg.chroma_collection_retrieval)
    res = col.get(ids=sorted(set(chunk_ids)), include=["documents", "metadatas"])
    return {
        cid: {"text": doc or "", "meta": meta or {}}
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=False)
    }


def chunk_risolto(rec: dict, cid: str, chunks: dict) -> dict | None:
    """Testo e metadati di un passaggio, dall'indice **o** dalla tupla di audit.

    Non tutti i passaggi vengono dal vector store: quelli del ramo meta — sezioni della
    scheda del corpus e blocco delle statistiche calcolate — esistono in memoria al momento
    della risposta e da nessuna altra parte. Per questo la tupla se li porta dentro
    (`passages_inline`), e per questo si guarda lì prima di dichiarare un passaggio
    introvabile: una citazione che non si può aprire non si può giudicare, e verrebbe letta
    come un guasto dell'indice invece che come un passaggio di natura diversa.
    """
    ch = chunks.get(cid)
    if ch is not None:
        return ch
    inline = (rec.get("passages_inline") or {}).get(cid)
    if inline is None:
        return None
    return {"text": inline.get("text") or "", "meta": {"source": inline.get("source") or ""}}


def _risposta(rec: dict) -> str:
    """Il testo della risposta, con fallback per le run anteriori ad ``answer_text``.

    Difetto misurato: senza fallback la scheda e il modulo mostravano «—» su tutte le righe,
    cioè si sarebbe giudicata una risposta invisibile. Su quelle run il testo si recupera da
    ``raw_output`` (il JSON del modello), tranne sugli hit di cache semantica dove
    ``raw_output`` è vuoto e il testo è irrecuperabile: lì lo si dichiara.
    """
    testo = (rec.get("answer_text") or "").strip()
    if testo:
        return testo
    raw = (rec.get("raw_output") or "").strip()
    if not raw:
        return ""
    try:
        return (json.loads(raw).get("answer") or "").strip() or raw
    except (json.JSONDecodeError, AttributeError):
        return raw


def _contesto(rec: dict) -> list[str]:
    """I chunk che il modello aveva davanti, in ordine di rank.

    Le run anteriori a ``retrieved_chunk_ids`` non lo hanno: si degrada ai soli citati, e la
    scheda lo dichiara — meglio una scheda parziale ed esplicita che una completa e falsa.
    """
    return rec.get("retrieved_chunk_ids") or rec.get("cited_chunk_ids") or []


def _tipo(rec: dict) -> str:
    if rec.get("refused"):
        return "rifiuto"
    if rec.get("uncertain"):
        return "astensione"
    if rec.get("from_cache"):
        return "hit-cache"
    return "risposta"


def _selezione(records: list[dict], condition: str | None) -> list[dict]:
    if condition in ("off", "on"):
        want = condition == "on"
        records = [r for r in records if r.get("cache_enabled") is want]
    return records


def _da_giudicare(rec: dict) -> bool:
    """I rifiuti deterministici non hanno nulla da giudicare: la pipeline ha rifiutato prima
    di chiamare il modello, e la correttezza del rifiuto è già misurata automaticamente.

    Le **astensioni** nemmeno: non c'è nessuna affermazione di cui verificare la fedeltà. Se
    l'astensione fosse *sbagliata* lo si vede nella tabella per categoria del report A/B (una
    `in_corpus` che finisce fra le astensioni), non nel giudizio di fedeltà.

    Gli hit di cache semantica SÌ: servono la risposta di una domanda *diversa* ma simile,
    e se quella risposta è fedele alla **nuova** domanda è precisamente il rischio della
    cache semantica — l'unico posto dove si può misurare è qui.
    """
    return _tipo(rec) in ("risposta", "hit-cache")


def build_form(records: list[dict], run_id: str, idx: dict) -> list[dict]:
    rows = []
    for rec in records:
        if not _da_giudicare(rec):
            continue
        ident, categoria = idx.get(rec["query"], ("", ""))
        rows.append(
            {
                "run_id": run_id,
                "condition": "on" if rec.get("cache_enabled") else "off",
                "id": ident,
                "categoria": categoria,
                "tipo": _tipo(rec),
                "domanda": rec["query"],
                "fedele": "",
                "causa": "",
                "citazione_corretta": "",
                "italiano_1_5": "",
                "note": "",
                "ereditato_da": "",
                "risposta": _risposta(rec),
            }
        )
    return rows


def build_sheet(
    cfg: RagConfig, records: list[dict], run_id: str, condition: str | None, idx: dict, form_name: str
) -> str:
    # Serve il testo di TUTTI i chunk finiti nel contesto, non solo dei citati: è l'unico modo
    # per rispondere a «la risposta era lì e il modello l'ha mancata?».
    all_ids = [cid for r in records for cid in _contesto(r)]
    chunks = _fetch_chunks(cfg, all_ids)
    n_giudicabili = sum(1 for r in records if _da_giudicare(r))
    senza_contesto = [r for r in records if _da_giudicare(r) and not r.get("retrieved_chunk_ids")]

    lines = [
        f"# Scheda di spot-check — run `{run_id}`",
        "",
        f"- condizione: **{condition or 'tutte'}**  ·  record: **{len(records)}**  ·  "
        f"da giudicare: **{n_giudicabili}**",
        f"- I giudizi si scrivono in **`{form_name}`**, non qui. Questa scheda serve a leggere.",
        "",
        "**Come si giudica.** Sotto ogni risposta trovi **tutti** i passaggi che il modello aveva",
        "in contesto, marcati `CITATO` o `non citato`. Servono a rispondere a due domande diverse:",
        "",
        "- **`fedele`** (sì/no) — l'affermazione è *contenuta nel passaggio citato*? Non «è vera",
        "  nel mondo», non «è plausibile»: è in quel testo. Se il numero viene da un passaggio",
        "  che parla d'altro, è **no** anche se il numero esiste altrove nel corpus.",
        "- **`causa`** — solo se `fedele=no`, e qui serve leggere *tutti* i passaggi:",
        "  - **`retrieval`** = la risposta **non c'è in nessuno** dei passaggi mostrati. Il modello",
        "    non poteva saperla: il difetto è a monte (reranker, `top_k`, arm keyword). Nota che in",
        "    questo caso il modello avrebbe dovuto **astenersi** invece di rispondere.",
        "  - **`generazione`** = la risposta **c'era** in uno dei passaggi e il modello ha estratto",
        "    o attribuito male. Qui il difetto è il modello.",
        "- **`citazione_corretta`** (sì/no) — il rimando punta al *documento giusto*? Domanda",
        "  diversa dalla fedeltà: un'affermazione può essere fedele al testo mostrato ma",
        "  attribuita alla delibera sbagliata. È ciò che rende la risposta verificabile.",
        "- **`italiano_1_5`** — 5 = pubblicabile così a un cittadino; 1 = inutilizzabile.",
        "",
        "> I rifiuti deterministici non compaiono nel modulo: la pipeline ha rifiutato prima di",
        "> chiamare il modello e la correttezza del rifiuto è già misurata automaticamente. Restano",
        "> però nella scheda, coi loro passaggi: è lì che si vede se un rifiuto era dovuto o se la",
        "> risposta c'era e la soglia l'ha buttata via.",
        "",
    ]

    if senza_contesto:
        lines += [
            f"> ⚠️ **{len(senza_contesto)} record senza `retrieved_chunk_ids`**: questa run è",
            "> anteriore al campo. Vedrai solo i passaggi *citati*, quindi la colonna `causa` non è",
            "> compilabile in modo affidabile. Rigira l'eval per avere il contesto completo.",
            "",
        ]

    for rec in records:
        tipo = _tipo(rec)
        ident, categoria = idx.get(rec["query"], ("?", "?"))
        lines += ["---", "", f"## `{ident}` [{tipo}] [{categoria}] {rec['query']}", ""]

        if tipo == "rifiuto":
            # Non entra nel modulo, ma i passaggi si mostrano: un rifiuto è *sbagliato* se la
            # risposta era nel contesto e la soglia l'ha scartata. Non lo si vede altrimenti.
            lines += [
                f"_Rifiutata dalla pipeline: {rec.get('refusal_reason')} — non entra nel modulo._",
                "",
                "> Controllo utile: se la risposta **c'era** in uno dei passaggi qui sotto, questo",
                "> rifiuto è un falso negativo e la soglia è troppo alta.",
                "",
            ]

        if tipo == "astensione":
            manc = ", ".join(f"«{t}»" for t in (rec.get("missing_terms") or []))
            lines += [
                f"_La pipeline si è **astenuta** (segnale IDF {rec.get('abstention_signal', 0):.2f}): "
                f"nei passaggi non compare {manc or '—'}. Non entra nel modulo: non c'è nessuna "
                "affermazione di cui verificare la fedeltà._",
                "",
                "> Controllo utile: se la risposta **c'era** in uno dei passaggi qui sotto,",
                "> l'astensione è un falso allarme e la soglia IDF è troppo bassa.",
                "",
            ]

        if tipo == "hit-cache":
            extra = rec.get("extra") or {}
            lines += [
                f"_Servita dalla **cache semantica** (similarità {extra.get('cache_similarity')})._",
                f"_Domanda originale in cache: «{extra.get('matched_query')}»_",
                "",
                "> Qui la domanda da porsi è: la risposta memorizzata è fedele a **questa**",
                "> domanda, non a quella originale?",
                "",
            ]

        risposta = _risposta(rec)
        if not risposta:
            risposta = "⚠ testo della risposta non disponibile in questa run (né answer_text né raw_output)"
        lines += ["**Risposta:**", "", textwrap.indent(risposta, "> "), ""]

        claims = rec.get("claims") or []
        if not claims and tipo == "risposta":
            lines += ["**⚠ Nessun claim strutturato**: niente da verificare puntualmente.", ""]
        for j, claim in enumerate(claims, 1):
            passaggi = ", ".join(f"[{n}]" for n in claim.get("passages", [])) or "—"
            lines += [f"**Affermazione {j}** (cita {passaggi}): {claim.get('statement', '')}", ""]

        contesto = _contesto(rec)
        cited = set(rec.get("cited_chunk_ids") or [])
        parziale = not rec.get("retrieved_chunk_ids")
        if contesto:
            etichetta = "Passaggi CITATI (contesto completo non disponibile)" if parziale else "Contesto completo"
            lines += [
                f"**{etichetta}** — {len(contesto)} passaggi, "
                f"{len(cited)} citati. La risposta è in uno di questi?",
                "",
            ]
        for n, cid in enumerate(contesto, start=1):
            marca = "**CITATO**" if cid in cited else "non citato"
            ch = chunk_risolto(rec, cid, chunks)
            if ch is None:
                lines += [f"- `[{n}]` {marca} `{cid}` — **non trovato nel vector store**", ""]
                continue
            titolo = ch["meta"].get("title") or ch["meta"].get("source") or "?"
            lines += [
                f"<details><summary>[{n}] {marca} · {titolo} · <code>{cid}</code></summary>",
                "",
                "```text",
                ch["text"].strip(),
                "```",
                "",
                "</details>",
                "",
            ]

        if _da_giudicare(rec):
            lines += [f"→ compila la riga **`{ident}`** ({tipo}) in `{form_name}`", ""]

    return "\n".join(lines) + "\n"


def main(
    run_id: str,
    condition: str | None = None,
    out_dir: str | None = None,
    eval_set: str = "eval/eval_set.jsonl",
) -> int:
    cfg = RagConfig.from_env()
    records = _selezione(_load_audit(run_id, cfg.audit_log_dir), condition)
    if not records:
        print(f"[spot-check] nessun record per condizione {condition!r} nella run {run_id}")
        return 1
    idx = _index_eval_set(eval_set)

    dest = Path(out_dir or "eval/reports")
    dest.mkdir(parents=True, exist_ok=True)
    suffix = f"-{condition}" if condition else ""
    form_path = dest / f"{run_id}-giudizi{suffix}.csv"
    sheet_path = dest / f"{run_id}-spotcheck{suffix}.md"

    form = build_form(records, run_id, idx)
    if form_path.exists():
        # Non sovrascrivere giudizi già dati: è l'unica copia del lavoro umano.
        print(f"[spot-check] modulo già presente, NON sovrascritto: {form_path}")
    else:
        with form_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLONNE_FORM)
            w.writeheader()
            w.writerows(form)
        print(f"[spot-check] modulo da compilare : {form_path}  ({len(form)} righe)")

    sheet_path.write_text(build_sheet(cfg, records, run_id, condition, idx, form_path.name), encoding="utf-8")
    print(f"[spot-check] scheda da leggere   : {sheet_path}")
    cond_arg = f" --condition {condition}" if condition else ""
    print(f"[spot-check] poi: uv run python -m app eval-human --run {run_id}{cond_arg}")
    return 0
