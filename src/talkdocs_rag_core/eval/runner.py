"""C6 — eval harness: gira l'eval set con cache OFF e ON, calcola le metriche
automatiche del §1 spec e produce il **report A/B** (CSV + markdown).

Condizioni A/B (stessa run, stesso eval set):
- **OFF** — baseline: cache semantica disattivata e ``prompt_cache_key`` unico per
  richiesta (sopprime anche il provider cache).
- **ON**  — cache semantica attiva + ``prompt_cache_key`` stabile (provider cache attivo).
  Le domande near-duplicate colpiscono la cache semantica; il prefisso stabile scalda
  il provider cache.

Metriche automatiche: esistenza/validità ``source_id``, disciplina del rifiuto per
categoria, hit-rate (semantico + provider), token, ``cached_tokens``, costo (se il
pricing è configurato), latenza. Le colonne di giudizio umano (fedeltà, italiano,
correttezza citazione) sono lasciate vuote da compilare.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from config import RagConfig

# I giudizi umani NON stanno qui. Il CSV del report è **rigenerato a ogni run**: colonne di
# giudizio vuote in un file rigenerabile sono una trappola (si compilano e alla run dopo
# spariscono, e nessuno le rilegge). Vivono in un modulo separato per run_id, prodotto da
# `app spot-check` e riletto da `app eval-human`.


@dataclass
class EvalItem:
    id: str
    category: str
    expect_refuse: bool | None
    question: str
    # Via attesa: "structured" | "refuse" | "pointwise". Campo, non categoria nuova: le
    # categorie sono già cinque e moltiplicarle le renderebbe illeggibili.
    route_attesa: str | None = None
    # Verità numerica per le risposte calcolate. Confronto esatto: a un COUNT non si
    # concede tolleranza.
    expected_value: int | None = None


def _load_set(path: Path) -> list[EvalItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        items.append(
            EvalItem(
                d["id"], d["category"], d.get("expect_refuse"), d["question"],
                d.get("route_attesa"), d.get("expected_value"),
            )
        )
    return items


def _cost(cfg: RagConfig, usage: dict) -> float:
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    cached = usage.get("cached_tokens", 0)
    non_cached = max(0, pt - cached)
    return (
        non_cached * cfg.price_input_per_mtok
        + cached * cfg.price_cached_per_mtok
        + ct * cfg.price_output_per_mtok
    ) / 1_000_000


def _row(cfg: RagConfig, item: EvalItem, condition: str, res) -> dict:
    usage = res.usage or {}
    # «Declinata» = la pipeline non ha dato una risposta sicura, per rifiuto deterministico
    # (sotto soglia di supporto) **o** per astensione (in tema ma senza la cosa chiesta).
    # Sono due meccanismi distinti e vanno riportati separati, ma per l'accuratezza del
    # rifiuto contano insieme: la domanda è «il sistema si è trattenuto quando doveva?».
    declined = bool(res.refused) or bool(res.uncertain)
    answered = not declined
    route = getattr(res, "route", "pointwise")
    strutturata = route == "structured"
    # La fonte di una risposta calcolata è la query eseguita, non un chunk_id: senza questo
    # ramo la metrica segnerebbe «senza fonte» una risposta perfetta.
    source_ok = answered and (strutturata or len(res.cited_chunk_ids) > 0)

    valore = res.structured.computed_value if (strutturata and res.structured) else None
    if item.expected_value is None or valore is None:
        value_ok = ""
    else:
        value_ok = int(valore == item.expected_value)
    route_ok = "" if item.route_attesa is None else int(route == item.route_attesa)
    vb = getattr(res, "verbatim", None)
    if item.expect_refuse is None:
        refusal_correct = ""  # non punteggiato (es. categoria `vaga`)
    else:
        refusal_correct = int(declined == item.expect_refuse)
    row = {
        "id": item.id,
        "category": item.category,
        "condition": condition,
        "expect_refuse": item.expect_refuse if item.expect_refuse is not None else "",
        "refused": int(res.refused),
        "uncertain": int(res.uncertain),
        "truncated": int(res.truncated),
        "abstention_signal": round(res.abstention_signal, 2),
        "missing_terms": ";".join(res.missing_terms),
        "refusal_correct": refusal_correct,
        "support_score": "" if res.support_score != res.support_score else round(res.support_score, 4),  # nan→""
        "source_id_ok": int(source_ok) if answered else "",
        "n_citations": len(res.cited_chunk_ids),
        "invalid_citations": len(res.invalid_citations),
        "route": route,
        "route_attesa": item.route_attesa or "",
        "route_ok": route_ok,
        "expected_value": item.expected_value if item.expected_value is not None else "",
        "computed_value": valore if valore is not None else "",
        "value_ok": value_ok,
        "verbatim_valid_ratio": "" if not vb or vb.valid_ratio is None else round(vb.valid_ratio, 3),
        "verbatim_misattributed": vb.n_misattributed if vb else "",
        "verbatim_not_found": vb.n_not_found if vb else "",
        "uncertain_reason": getattr(res, "uncertain_reason", None) or "",
        "from_cache_semantic": int(res.from_cache and res.cache_kind == "semantic"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cached_tokens": usage.get("cached_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cost": round(_cost(cfg, usage), 6),
        "latency_s": round(res.latency_s, 3) if res.latency_s is not None else "",
        "latency_wall_s": round(res.latency_wall_s, 3) if res.latency_wall_s is not None else "",
        "question": item.question,
        "answer": res.answer_text,
    }
    return row


def _verdict(res) -> str:
    """Etichetta compatta dell'esito, per la riga di progresso."""
    if getattr(res, "route", "pointwise") == "structured" and res.structured is not None:
        s = res.structured
        return f"CALCOLATA ({s.intent}) → {s.computed_value}  righe={s.n_rows}  [0 chiamate]"
    if getattr(res, "route", "pointwise") == "uncovered":
        return "RIFIUTO DICHIARATO (aggregazione fuori copertura)"
    if res.from_cache and res.cache_kind == "semantic":
        sim = res.extra.get("cache_similarity")
        return f"HIT-semantica{f' (sim {sim:.3f})' if sim is not None else ''}"
    if res.refused:
        return f"RIFIUTO ({res.refusal_reason})"
    if res.uncertain:
        return f"ASTENSIONE (idf {res.abstention_signal:.1f}, manca: {','.join(res.missing_terms[:3])})"
    u = res.usage or {}
    cached = u.get("cached_tokens", 0)
    tronca = "  ⚠TRONCATA" if res.truncated else ""
    return (
        f"risposta  tok={u.get('prompt_tokens', 0)}→{u.get('completion_tokens', 0)}"
        f"  cached={cached}  cit={len(res.cited_chunk_ids)}{tronca}"
    )


async def _run_condition(pipeline, items: list[EvalItem], condition: str, writer) -> list[dict]:
    # stato pulito della cache semantica all'inizio di ogni condizione
    pipeline.semantic_cache.clear()
    rows = []
    n = len(items)
    t_start = time.perf_counter()
    print(f"[eval:{condition.upper():3s}] avvio — {n} domande, cache semantica azzerata", flush=True)
    for i, item in enumerate(items):
        if condition == "off":
            res = await pipeline.ask(item.question, use_cache=False, provider_cache_key=f"eval-off-{item.id}-{i}")
        else:
            res = await pipeline.ask(item.question, use_cache=True, provider_cache_key=None)
        writer.record(res, pipeline.corpus_version, cache_enabled=(condition == "on"))
        rows.append(_row(pipeline.cfg, item, condition, res))

        # --- progresso in chiaro, una riga per domanda, flushata subito ---
        elapsed = time.perf_counter() - t_start
        print(
            f"[eval:{condition.upper():3s}] {i + 1:2d}/{n} {item.id:<12s} "
            f"lat={res.latency_s or 0:6.2f}s  {_verdict(res)}"
            f"   [{elapsed / 60:.1f}m trascorsi]",
            flush=True,
        )
        # Divergenza wall vs monotonico = la macchina ha dormito: la latenza wall di questa
        # domanda è un artefatto, non una misura. Meglio saperlo subito che scoprirlo nel report.
        if res.latency_wall_s is not None and res.latency_s is not None:
            drift = res.latency_wall_s - res.latency_s
            if drift > 5.0:
                print(
                    f"    ⚠  macchina sospesa ~{drift / 60:.1f} min durante questa domanda: "
                    f"wall={res.latency_wall_s:.0f}s vs servizio={res.latency_s:.1f}s",
                    flush=True,
                )
    print(f"[eval:{condition.upper():3s}] completata in {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
    return rows


def _aggregate(cfg: RagConfig, rows: list[dict], condition: str) -> dict:
    sub = [r for r in rows if r["condition"] == condition]
    scored = [r for r in sub if r["refusal_correct"] != ""]
    answered = [r for r in sub if r["refused"] == 0 and r.get("uncertain", 0) == 0]
    n_uncertain = sum(r.get("uncertain", 0) for r in sub)
    # Richieste che hanno DAVVERO chiamato il modello: né hit di cache semantica, né
    # rifiuti deterministici (che escono prima della chiamata, con usage vuoto). È il
    # denominatore di provider_hit_rate: gonfiarlo sottostima il prompt caching.
    model_calls = [r for r in sub if r["from_cache_semantic"] == 0 and r["prompt_tokens"] > 0]
    provider_hits = [r for r in model_calls if r["cached_tokens"] > 0]
    lat = [r["latency_s"] for r in sub if isinstance(r["latency_s"], (int, float))]
    # Scarto wall-clock vs monotonico: >0 significa che la macchina è andata in suspend
    # durante la condizione. Serve a marcare il report come contaminato, non a correggerlo.
    drift = sum(
        r["latency_wall_s"] - r["latency_s"]
        for r in sub
        if isinstance(r.get("latency_wall_s"), (int, float)) and isinstance(r["latency_s"], (int, float))
    )

    attesi_strutturati = [r for r in sub if r["route_attesa"] == "structured"]
    attesi_puntuali = [r for r in sub if r["route_attesa"] == "pointwise"]
    con_route = [r for r in sub if r["route_ok"] != ""]
    con_valore = [r for r in sub if r["value_ok"] != ""]
    ratios = [r["verbatim_valid_ratio"] for r in sub if r["verbatim_valid_ratio"] != ""]

    def rate(num, den):
        return round(num / den, 3) if den else 0.0

    return {
        "condition": condition,
        "n": len(sub),
        "refusal_accuracy": rate(sum(r["refusal_correct"] for r in scored), len(scored)),
        "citation_validity": rate(sum(r["source_id_ok"] for r in answered if r["source_id_ok"] != ""), len(answered)),
        "invalid_citation_total": sum(r["invalid_citations"] for r in sub),
        "semantic_hit_rate": rate(sum(r["from_cache_semantic"] for r in sub), len(sub)),
        "provider_hit_rate": rate(len(provider_hits), len(model_calls)),
        "prompt_tokens": sum(r["prompt_tokens"] for r in sub),
        "completion_tokens": sum(r["completion_tokens"] for r in sub),
        "cached_tokens": sum(r["cached_tokens"] for r in sub),
        "total_tokens": sum(r["total_tokens"] for r in sub),
        "cost": round(sum(r["cost"] for r in sub), 6),
        "latency_avg": round(statistics.mean(lat), 3) if lat else 0.0,
        "latency_median": round(statistics.median(lat), 3) if lat else 0.0,
        "latency_max": round(max(lat), 3) if lat else 0.0,
        "suspend_drift_s": round(drift, 1),
        "model_calls": len(model_calls),
        "n_refused": sum(r["refused"] for r in sub),
        "n_uncertain": n_uncertain,
        "n_answered": len(answered),
        "n_truncated": sum(r.get("truncated", 0) for r in sub),
        "routing_accuracy": rate(sum(int(r["route_ok"]) for r in con_route), len(con_route)),
        # Due errori in direzioni opposte: mai una media unica.
        "router_recall": rate(
            sum(1 for r in attesi_strutturati if r["route"] == "structured"), len(attesi_strutturati)
        ),
        "router_false_positive": rate(
            sum(1 for r in attesi_puntuali if r["route"] != "pointwise"), len(attesi_puntuali)
        ),
        "structured_value_accuracy": rate(sum(int(r["value_ok"]) for r in con_valore), len(con_valore)),
        "verbatim_valid_ratio": round(statistics.mean(ratios), 3) if ratios else 0.0,
        "verbatim_misattributed": sum(int(r["verbatim_misattributed"] or 0) for r in sub),
        "verbatim_not_found": sum(int(r["verbatim_not_found"] or 0) for r in sub),
        "n_structured": sum(1 for r in sub if r["route"] == "structured"),
        "n_uncovered": sum(1 for r in sub if r["route"] == "uncovered"),
    }


def _per_categoria(rows: list[dict], condition: str) -> list[str]:
    """Righe markdown con l'esito per categoria.

    È il punto delle categorie: `aggregazione` e `near_miss` devono comportarsi in modo
    *diverso* da `in_corpus`, e una media su tutto lo nasconde.
    """
    sub = [r for r in rows if r["condition"] == condition]
    cats = sorted({r["category"] for r in sub})
    out = ["| Categoria | n | risposte | astensioni | rifiuti | atteso |", "|---|---|---|---|---|---|"]
    atteso = {
        "in_corpus": "risposta",
        "vaga": "astensione (o risposta cauta)",
        "near_miss": "rifiuto/astensione",
        "aggregazione": "risposta *calcolata* sul manifest o rifiuto dichiarato",
        "out_of_corpus": "rifiuto",
        "borderline": "(categoria legacy, ambigua)",
    }
    for c in cats:
        s = [r for r in sub if r["category"] == c]
        risp = sum(1 for r in s if r["refused"] == 0 and r.get("uncertain", 0) == 0)
        inc = sum(r.get("uncertain", 0) for r in s)
        rif = sum(r["refused"] for r in s)
        out.append(f"| `{c}` | {len(s)} | {risp} | {inc} | {rif} | {atteso.get(c, '—')} |")
    return out


def _markdown(
    cfg: RagConfig, agg_off: dict, agg_on: dict, n_items: int, corpus_version: str, rows: list[dict]
) -> str:
    pricing_set = any((cfg.price_input_per_mtok, cfg.price_output_per_mtok, cfg.price_cached_per_mtok))

    def pct(x):
        return f"{x * 100:.1f}%"

    lines = [
        "# Report eval A/B — caching ON vs OFF",
        "",
        f"- **Modello:** `{cfg.mistral_model}`  ·  **embedding:** `{cfg.mistral_embed_model}` (dim {cfg.embed_dim})",
        f"- **corpus_version:** `{corpus_version}`",
        # Derivata dai dati, non cablata: la lista cablata era ferma a `borderline` da due
        # giorni dopo che la tassonomia si era spaccata in near_miss/aggregazione + vaga.
        f"- **eval set:** {n_items} domande ({' / '.join(sorted({r['category'] for r in rows}))})",
        f"- **parametri:** chunk={cfg.chunk_tokens}/{cfg.chunk_overlap_ratio}, "
        f"top_k={cfg.rag_top_k}, pesi v/k={cfg.hybrid_vector_weight}/{cfg.hybrid_keyword_weight}, "
        f"support_thr={cfg.support_threshold}, cache_sim_thr={cfg.cache_sim_threshold}",
        f"- **pricing:** {'configurato' if pricing_set else 'NON configurato (costo=0 finché mancano le tariffe)'}",
        "",
        "| Metrica | OFF (baseline) | ON (cache) |",
        "|---|---|---|",
        f"| Accuratezza rifiuto | {pct(agg_off['refusal_accuracy'])} | {pct(agg_on['refusal_accuracy'])} |",
        f"| Validità citazioni (source_id) | {pct(agg_off['citation_validity'])} "
        f"| {pct(agg_on['citation_validity'])} |",
        f"| Citazioni invalide (tot) | {agg_off['invalid_citation_total']} | {agg_on['invalid_citation_total']} |",
        f"| **Risposte troncate** (tetto {cfg.max_output_tokens} tok) | {agg_off['n_truncated']} "
        f"| {agg_on['n_truncated']} |",
        f"| Hit-rate cache semantica | {pct(agg_off['semantic_hit_rate'])} | {pct(agg_on['semantic_hit_rate'])} |",
        f"| Hit-rate cache provider | {pct(agg_off['provider_hit_rate'])} | {pct(agg_on['provider_hit_rate'])} |",
        f"| Risposte / astensioni / rifiuti | {agg_off['n_answered']} / {agg_off['n_uncertain']} / "
        f"{agg_off['n_refused']} | {agg_on['n_answered']} / {agg_on['n_uncertain']} / {agg_on['n_refused']} |",
        f"| Chiamate al modello | {agg_off['model_calls']} | {agg_on['model_calls']} |",
        f"| Prompt tokens (tot) | {agg_off['prompt_tokens']} | {agg_on['prompt_tokens']} |",
        f"| Cached tokens (tot) | {agg_off['cached_tokens']} | {agg_on['cached_tokens']} |",
        f"| Completion tokens (tot) | {agg_off['completion_tokens']} | {agg_on['completion_tokens']} |",
        f"| Costo stimato | {agg_off['cost']:.6f} | {agg_on['cost']:.6f} |",
        f"| Latenza mediana (s) | {agg_off['latency_median']} | {agg_on['latency_median']} |",
        f"| Latenza media (s) | {agg_off['latency_avg']} | {agg_on['latency_avg']} |",
        f"| Latenza max (s) | {agg_off['latency_max']} | {agg_on['latency_max']} |",
        "",
        "> **Queste metriche misurano la forma, non la verità.** `Validità citazioni` verifica",
        "> solo che i marcatori `[n]` siano in range: nessuno controlla che il passaggio citato",
        "> sostenga l'affermazione. Per la fedeltà serve il giudizio umano:",
        "> `app spot-check --run <run_id>` genera scheda + modulo, `app eval-human --run <run_id>`",
        "> rilegge i giudizi e calcola il tasso di fedeltà.",
        ">",
        "> `Chiamate al modello` esclude i rifiuti deterministici (escono prima della chiamata) e",
        "> gli hit di cache semantica: è il denominatore dell'hit-rate provider.",
        ">",
        "> **Sulla latenza si legge la mediana, non la media.** Su n=33 un solo timeout di rete",
        "> sposta la media di ~18 s senza dire nulla sul modello: confronta `media` con `max` prima",
        "> di citare un numero.",
        "",
        "## Esito per categoria (condizione OFF)",
        "",
        "Le categorie servono a questo: `aggregazione` e `near_miss` **devono** comportarsi in",
        "modo diverso da `in_corpus`, e una media su tutto lo nasconderebbe.",
        "",
        *_per_categoria(rows, "off"),
        "",
        f"> **Astensione (C3b).** Soglia IDF `{cfg.abstention_idf_threshold}`"
        f"{' — guardiano SPENTO' if cfg.abstention_idf_threshold <= 0 else ''}. La pipeline si astiene",
        "> quando un termine *raro* della domanda non compare in nessun passaggio recuperato:",
        "> segnale ortogonale al `support_score`, che misura vicinanza di argomento e non",
        "> presenza della risposta. Non intercetta le domande di aggregazione (lì i termini ci",
        "> sono tutti, manca la vista d'insieme): quelle richiedono una query sui metadati.",
    ]

    lines += [
        "",
        "## Routing e provenienza (incremento 1)",
        "",
        "> ⚠ Schema di output e `SYSTEM_PREFIX` sono cambiati: il prefisso cache-friendly è",
        "> diverso da quello delle run precedenti. **L'A/B del caching è ri-baselinato**: i",
        "> numeri di questa run non sono confrontabili con quelli antecedenti l'incremento 1.",
        "",
        "| Metrica | OFF | ON |",
        "|---|---|---|",
        f"| Accuratezza di routing | {pct(agg_off['routing_accuracy'])} | {pct(agg_on['routing_accuracy'])} |",
        f"| Richiamo del router | {pct(agg_off['router_recall'])} | {pct(agg_on['router_recall'])} |",
        f"| Falsi positivi del router | {pct(agg_off['router_false_positive'])} "
        f"| {pct(agg_on['router_false_positive'])} |",
        f"| Correttezza dei valori calcolati | {pct(agg_off['structured_value_accuracy'])} "
        f"| {pct(agg_on['structured_value_accuracy'])} |",
        f"| Risposte calcolate / rifiuti dichiarati | {agg_off['n_structured']} / {agg_off['n_uncovered']} "
        f"| {agg_on['n_structured']} / {agg_on['n_uncovered']} |",
        f"| Quota media di verbatim validi | {pct(agg_off['verbatim_valid_ratio'])} "
        f"| {pct(agg_on['verbatim_valid_ratio'])} |",
        f"| Span misattribuiti / non trovati | {agg_off['verbatim_misattributed']} / "
        f"{agg_off['verbatim_not_found']} | {agg_on['verbatim_misattributed']} / {agg_on['verbatim_not_found']} |",
    ]

    drift = max(agg_off["suspend_drift_s"], agg_on["suspend_drift_s"])
    if drift > 60:
        lines += [
            "",
            f"> ⚠️ **Run contaminata.** Scarto wall-clock/monotonico di {drift / 60:.1f} min: la macchina è",
            "> andata in suspend durante la run (OFF "
            f"{agg_off['suspend_drift_s'] / 60:.1f} min, ON {agg_on['suspend_drift_s'] / 60:.1f} min).",
            "> Le metriche di token, costo, citazioni e rifiuto restano valide; **la latenza no** —",
            "> ripeti la run su macchina desta prima di usare questi numeri sulla latenza.",
        ]
    return "\n".join(lines) + "\n"


async def run_eval(cfg: RagConfig, eval_set_path: str, out_dir: str) -> dict:
    from app.pipeline import build_pipeline
    from audit.record import AuditWriter

    items = _load_set(Path(eval_set_path))
    pipeline = await build_pipeline(cfg)
    run_id = "eval-" + datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    writer = AuditWriter(run_id, cfg.audit_log_dir)

    print(f"[eval] {len(items)} domande × 2 condizioni (OFF, ON) — run {run_id}")
    rows_off = await _run_condition(pipeline, items, "off", writer)
    print("[eval] condizione OFF completata")
    rows_on = await _run_condition(pipeline, items, "on", writer)
    print("[eval] condizione ON completata")
    rows = rows_off + rows_on

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{run_id}.csv"
    md_path = out / f"{run_id}.md"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    agg_off = _aggregate(cfg, rows, "off")
    agg_on = _aggregate(cfg, rows, "on")
    md_path.write_text(
        _markdown(cfg, agg_off, agg_on, len(items), pipeline.corpus_version, rows), encoding="utf-8"
    )

    print(f"[eval] CSV : {csv_path}")
    print(f"[eval] MD  : {md_path}")
    print(f"[eval] audit: {writer.path}")
    return {"csv": str(csv_path), "md": str(md_path), "off": agg_off, "on": agg_on}


def main() -> int:
    import asyncio

    cfg = RagConfig.from_env()
    asyncio.run(run_eval(cfg, "eval/eval_set.jsonl", "eval/reports"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
