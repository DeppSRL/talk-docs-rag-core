"""La tupla di audit deve permettere di RIESEGUIRE la query, non solo di leggerne l'esito.

È la forma più forte di difendibilità che il PoC produca: non «il modello ha letto questi
documenti», ma «questo totale è la somma di queste righe, con questa query, riproducibile».
"""

import json

from audit.record import AuditWriter
from rag.generation import RagResult
from rag.outcomes import StructuredOutcome, VerbatimOutcome


def _risultato_strutturato():
    return RagResult(
        query="Quante delibere nel 2024?", answer_text="Risultano 93 delibere…", refused=False,
        refusal_reason=None, support_score=0.0, cited_passages=[], cited_chunk_ids=[],
        invalid_citations=[], claims=[], passages=[], usage={}, raw_output="", model="test",
        params={}, route="structured", router_signals={"forma_conteggio": True},
        structured=StructuredOutcome(
            intent="count_delibere", sql="SELECT COUNT(*) … WHERE anno = ?", params=[2024],
            rows=[{"n": 93}], n_rows=1, computed_value=93,
            completeness={"count": 93, "max_numero": 95, "gap": 2}, cited_doc_ids=[],
        ),
    )


def test_la_tupla_porta_query_parametri_e_completezza(tmp_path):
    w = AuditWriter("run-test", str(tmp_path))
    w.record(_risultato_strutturato(), corpus_version="v1", cache_enabled=False)
    riga = json.loads(w.path.read_text(encoding="utf-8").strip())

    assert riga["route"] == "structured"
    assert riga["structured"]["sql"].startswith("SELECT COUNT(*)")
    assert riga["structured"]["params"] == [2024]
    assert riga["structured"]["completeness"]["gap"] == 2
    assert riga["router_signals"]["forma_conteggio"] is True


def test_la_tupla_porta_l_esito_verbatim_anche_a_guardia_spenta(tmp_path):
    """I segnali si registrano SEMPRE: le soglie si ritarano su run passate."""
    res = _risultato_strutturato()
    res.route = "pointwise"
    res.structured = None
    res.verbatim = VerbatimOutcome(n_claims=2, n_valid=1, n_misattributed=1, n_not_found=0,
                                   n_too_short=0, valid_ratio=0.5)
    w = AuditWriter("run-test2", str(tmp_path))
    w.record(res, corpus_version="v1", cache_enabled=False)
    riga = json.loads(w.path.read_text(encoding="utf-8").strip())

    assert riga["verbatim"]["valid_ratio"] == 0.5
    assert riga["verbatim"]["n_misattributed"] == 1
    assert riga["structured"] is None
