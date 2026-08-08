"""Giudizio umano assistito: dati per la UI e scrittura del modulo.

**Perché esiste.** Il giro `spot-check` → compila il CSV → `eval-human` funziona, ma
chiede di tenere aperti due file e di fare a mano il join fra loro: si legge
un'affermazione nella scheda, si cerca il passaggio, si torna nel CSV e si compila la
riga giusta. Su 34 risposte è lento e induce errori di allineamento — e la fedeltà è
l'unica metrica che decide la promozione del prototipo, quindi il costo di raccoglierla
è un costo del banco.

**Che cosa NON cambia.** Il modulo resta lo stesso CSV di `scripts/spot_check.py`, con le
stesse colonne: `eval-human` continua a rileggerlo senza sapere nulla di questa UI, e un
modulo compilato a mano in LibreOffice resta valido. La UI è un'interfaccia di scrittura
su quel file, non un formato nuovo.

**Il bundle.** L'audit registra i `chunk_id`, non il testo dei passaggi: ricostruirlo
richiede il vector store, che sulla macchina di chi giudica può non esserci (le run
girano in CI, su runner effimeri). `salva_bundle` congela tutto ciò che serve a giudicare
in un JSON che il workflow allega all'artifact: da lì la UI gira senza corpus, senza
chiavi API e senza rete.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from config import RagConfig
from scripts.spot_check import (
    COLONNE_FORM,
    COLONNE_GIUDIZIO,
    _contesto,
    _da_giudicare,
    _fetch_chunks,
    _index_eval_set,
    _load_audit,
    _risposta,
    _selezione,
    _tipo,
    build_form,
)

REPORTS = Path("eval/reports")


def _suffix(condition: str | None) -> str:
    return f"-{condition}" if condition else ""


def percorso_modulo(run_id: str, condition: str | None, out_dir: str | Path = REPORTS) -> Path:
    return Path(out_dir) / f"{run_id}-giudizi{_suffix(condition)}.csv"


def percorso_bundle(run_id: str, condition: str | None, out_dir: str | Path = REPORTS) -> Path:
    return Path(out_dir) / f"{run_id}-bundle{_suffix(condition)}.json"


def _chiave(riga: dict) -> str:
    """Identità di una riga del modulo. `id` quando c'è; altrimenti la domanda — una run
    può contenere query fuori dall'eval set (p.es. `web-session`), e appiattirle tutte su
    una chiave vuota le farebbe sovrascrivere a vicenda."""
    return riga.get("id") or riga.get("domanda") or ""


def costruisci_item(rec: dict, idx: dict, chunks: dict) -> dict:
    """Tutto ciò che serve a giudicare UNA risposta, in un oggetto solo.

    Gli esiti verbatim vengono dall'audit, non ricalcolati: la UI deve mostrare ciò che la
    pipeline **ha** deciso, non una seconda opinione che potrebbe divergere.
    """
    ident, categoria = idx.get(rec["query"], ("", ""))
    citati = set(rec.get("cited_chunk_ids") or [])
    passaggi = []
    for n, cid in enumerate(_contesto(rec), start=1):
        ch = chunks.get(cid) or {}
        meta = ch.get("meta") or {}
        passaggi.append(
            {
                "n": n,
                "chunk_id": cid,
                "titolo": meta.get("title") or meta.get("source") or "?",
                "testo": (ch.get("text") or "").strip(),
                "citato": cid in citati,
                "mancante": cid not in chunks,
            }
        )

    verb = rec.get("verbatim") or {}
    per_claim = {c.get("statement", ""): c for c in (verb.get("per_claim") or [])}
    claims = []
    for c in rec.get("claims") or []:
        st = c.get("statement", "")
        esito = per_claim.get(st) or {}
        claims.append(
            {
                "statement": st,
                "passages": c.get("passages") or [],
                "verbatim": esito.get("verbatim") or c.get("verbatim") or "",
                "esito": esito.get("esito") or "",
                "matched_passage": esito.get("matched_passage"),
            }
        )

    strutturata = rec.get("structured") or None
    if strutturata:
        # Le righe complete stanno nell'audit; qui ne bastano poche a mostrare che cosa è
        # stato contato — la UI non è il posto per scorrere 511 record.
        strutturata = {
            "intent": strutturata.get("intent"),
            "sql": strutturata.get("sql"),
            "params": strutturata.get("params"),
            "computed_value": strutturata.get("computed_value"),
            "n_rows": strutturata.get("n_rows"),
            "completeness": strutturata.get("completeness"),
            "rows": (strutturata.get("rows") or [])[:10],
        }

    return {
        "id": ident,
        "categoria": categoria,
        "tipo": _tipo(rec),
        "route": rec.get("route") or "pointwise",
        "router_source": rec.get("router_source") or "lexical",
        "domanda": rec["query"],
        "risposta": _risposta(rec),
        "claims": claims,
        "passaggi": passaggi,
        "structured": strutturata,
        "support_score": rec.get("support_score"),
        "verbatim_valid_ratio": verb.get("valid_ratio"),
        "cache": {
            "hit": bool(rec.get("from_cache")),
            "similarity": (rec.get("extra") or {}).get("cache_similarity"),
            "matched_query": (rec.get("extra") or {}).get("matched_query"),
        },
    }


def costruisci_bundle(cfg: RagConfig, run_id: str, condition: str | None, eval_set: str) -> dict:
    records = _selezione(_load_audit(run_id, cfg.audit_log_dir), condition)
    if not records:
        raise ValueError(f"nessun record per condizione {condition!r} nella run {run_id}")
    idx = _index_eval_set(eval_set)
    da_giudicare = [r for r in records if _da_giudicare(r)]
    chunks = _fetch_chunks(cfg, [cid for r in da_giudicare for cid in _contesto(r)])
    return {
        "run_id": run_id,
        "condition": condition,
        "model": cfg.mistral_model,
        "items": [costruisci_item(r, idx, chunks) for r in da_giudicare],
        # Il modulo si crea qui, così la UI non deve saper generare righe: le trova o le
        # scrive tutte insieme alla prima apertura.
        "form": build_form(records, run_id, idx),
    }


def salva_bundle(cfg: RagConfig, run_id: str, condition: str | None, eval_set: str, out_dir: str | Path) -> Path:
    bundle = costruisci_bundle(cfg, run_id, condition, eval_set)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = percorso_bundle(run_id, condition, dest)
    path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    return path


def carica_bundle(run_id: str, condition: str | None, out_dir: str | Path = REPORTS) -> dict | None:
    path = percorso_bundle(run_id, condition, out_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def leggi_modulo(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def scrivi_modulo(path: Path, righe: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLONNE_FORM)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in COLONNE_FORM} for r in righe])


def salva_giudizio(path: Path, form_iniziale: list[dict], chiave: str, valori: dict) -> dict:
    """Aggiorna **una** riga del modulo e riscrive il file.

    Il file si riscrive intero a ogni salvataggio perché è l'unico modo di restare
    compatibili con un CSV che l'utente può aver aperto in LibreOffice e riordinato. Le
    righe esistenti si conservano: un modulo già compilato a metà non si perde
    all'apertura della UI — è l'unica copia di lavoro umano non rigenerabile del banco.
    """
    righe = leggi_modulo(path) or [dict(r) for r in form_iniziale]
    trovata = None
    for r in righe:
        if _chiave(r) == chiave:
            trovata = r
            break
    if trovata is None:
        raise KeyError(f"riga non trovata nel modulo: {chiave!r}")
    for col in COLONNE_GIUDIZIO:
        if col in valori:
            trovata[col] = "" if valori[col] is None else str(valori[col])
    scrivi_modulo(path, righe)
    return trovata


def stato_modulo(path: Path, form_iniziale: list[dict]) -> dict[str, dict]:
    """`chiave → giudizio corrente`. Permette alla UI di riaprirsi dove si era rimasti."""
    righe = leggi_modulo(path) or form_iniziale
    return {_chiave(r): {c: r.get(c, "") for c in COLONNE_GIUDIZIO} for r in righe}


def run_disponibili(audit_dir: str | Path, solo_eval: bool = True) -> list[str]:
    """Run di audit presenti in locale, più recenti prima.

    `solo_eval` esclude `run-*` e `web-session`: sono singole `ask` da CLI o dalla
    console, e giudicarle non produce un tasso di fedeltà — il denominatore è l'eval set.
    In `logs/` se ne accumulano decine, e in un menu a tendina seppelliscono le due o tre
    run che si vogliono davvero giudicare.
    """
    d = Path(audit_dir)
    if not d.is_dir():
        return []
    nomi = (p.stem for p in d.glob("*.jsonl"))
    if solo_eval:
        nomi = (n for n in nomi if n.startswith("eval-"))
    return sorted(nomi, reverse=True)


def bundle_disponibili(out_dir: str | Path = REPORTS) -> list[dict]:
    d = Path(out_dir)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*-bundle*.json"), reverse=True):
        stem = p.stem
        run_id, _, coda = stem.partition("-bundle")
        out.append({"run_id": run_id, "condition": coda.lstrip("-") or None, "path": str(p)})
    return out
