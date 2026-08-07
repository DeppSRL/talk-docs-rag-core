"""Tara ``SUPPORT_THRESHOLD`` sul corpus corrente, invece di scegliere una soglia a occhio.

Il rifiuto deterministico (C3) confronta ``support_score`` — la **miglior similarità densa
fra i top-k** (``rag/generation.py::_support_score``) — con ``support_threshold``. Il
problema misurato: `mistral-embed` ha un **pavimento alto**, cioè assegna ~0,74 anche a
query palesemente fuori tema, quindi una soglia bassa non fa mai scattare il rifiuto.

Il pavimento è una proprietà dell'embedding, ma la finestra utile (pavimento → in-corpus)
va rimisurata a ogni cambio di corpus o di modello. Questo script esegue il retrieval su
tutto l'eval set, stampa la distribuzione del segnale per categoria e propone la soglia
che separa in-corpus e out-of-corpus con il margine più ampio.

Uso:
    uv run python scripts/tara_soglia.py
    uv run python scripts/tara_soglia.py --eval-set eval/eval_set.jsonl --top-k 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.wiring import build_embedding_service, build_hybrid, build_retrieval_store, build_whoosh  # noqa: E402
from config import RagConfig  # noqa: E402


def _support(results) -> float:
    """Stessa definizione di rag/generation.py::_support_score: max densa fra i top-k."""
    dense = [r.vector_score for r in results if r.vector_score is not None]
    return max(dense) if dense else 0.0


async def _run(eval_set: Path, top_k: int) -> int:
    cfg = RagConfig.from_env()
    embedding_service = build_embedding_service(cfg)
    retrieval_store = build_retrieval_store(cfg, embedding_service)
    whoosh = await build_whoosh(cfg)
    hybrid = build_hybrid(cfg, retrieval_store, whoosh)

    items = [json.loads(riga) for riga in eval_set.read_text(encoding="utf-8").splitlines() if riga.strip()]
    per_categoria: dict[str, list[tuple[str, float, str]]] = {}

    for it in items:
        results = await hybrid.search(it["question"], top_k=top_k)
        top_fonte = results[0].source if results else "(nessuno)"
        per_categoria.setdefault(it["category"], []).append((it["id"], _support(results), top_fonte))
    await whoosh.close()

    print(f"segnale: max similarità densa fra i top-{top_k}  ·  soglia attuale: {cfg.support_threshold}\n")
    for cat in ("in_corpus", "borderline", "out_of_corpus"):
        righe = sorted(per_categoria.get(cat, []), key=lambda x: x[1])
        if not righe:
            continue
        val = sorted(s for _, s, _ in righe)
        mediana = val[len(val) // 2]
        print(f"=== {cat} (n={len(righe)}) — min {val[0]:.3f}  mediana {mediana:.3f}  max {val[-1]:.3f}")
        for eid, s, fonte in righe:
            print(f"    {eid}  {s:.3f}  {fonte}")
        print()

    dentro = [s for _, s, _ in per_categoria.get("in_corpus", [])]
    fuori = [s for _, s, _ in per_categoria.get("out_of_corpus", [])]
    if not (dentro and fuori):
        print("Servono entrambe le categorie in_corpus e out_of_corpus per proporre una soglia.")
        return 1

    min_dentro, max_fuori = min(dentro), max(fuori)
    print(f"finestra utile: out-of-corpus max {max_fuori:.3f}  →  in-corpus min {min_dentro:.3f}")
    if min_dentro <= max_fuori:
        print(
            f"[ATTENZIONE] le distribuzioni si SOVRAPPONGONO (margine {min_dentro - max_fuori:+.3f}): nessuna soglia "
            "sul solo segnale denso separa i due insiemi. Serve un segnale migliore (margine top1-top2, "
            "copertura keyword, verificatore NLI) — hook già predisposto in RagConfig."
        )
        return 1

    proposta = round((min_dentro + max_fuori) / 2, 2)
    print(f"soglia proposta (punto medio): SUPPORT_THRESHOLD={proposta}  ·  margine {min_dentro - max_fuori:.3f}")
    if not (max_fuori < cfg.support_threshold < min_dentro):
        print(
            f"[ATTENZIONE] la soglia attuale ({cfg.support_threshold}) è fuori dalla finestra: "
            + ("rifiuta tutto." if cfg.support_threshold >= min_dentro else "non rifiuterà mai nulla.")
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-set", type=Path, default=Path("eval/eval_set.jsonl"))
    ap.add_argument("--top-k", type=int, default=None)
    a = ap.parse_args()
    cfg = RagConfig.from_env()
    return asyncio.run(_run(a.eval_set, a.top_k or cfg.rag_top_k))


if __name__ == "__main__":
    raise SystemExit(main())
