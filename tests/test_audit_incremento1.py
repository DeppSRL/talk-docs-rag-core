"""La tupla di audit deve permettere di RIESEGUIRE la query, non solo di leggerne l'esito.

È la forma più forte di difendibilità che il PoC produca: non «il modello ha letto questi
documenti», ma «questo totale è la somma di queste righe, con questa query, riproducibile».
"""

import json

from talkdocs_rag_core.audit.record import AuditWriter
from talkdocs_rag_core.rag.generation import RagResult
from talkdocs_rag_core.rag.outcomes import StructuredOutcome, VerbatimOutcome


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


def test_i_passaggi_fuori_indice_viaggiano_dentro_la_tupla(tmp_path):
    """Una citazione che nessuno può più risolvere non è una citazione.

    I passaggi del ramo meta — sezioni della scheda e blocco delle statistiche calcolate —
    esistono solo in memoria al momento della risposta: il vector store non li contiene e
    non li conterrà. Se la tupla registrasse i soli `chunk_id`, la risposta risulterebbe
    citata e sarebbe indifendibile. Quelli che vengono dall'indice restano fuori: lì il
    `chunk_id` è già la chiave per rileggerli, e duplicarne il testo gonfierebbe l'audit.
    """
    from talkdocs_rag_core.rag.generation import Passage

    res = RagResult(
        query="Di cosa parla questo corpus?", answer_text="Raccoglie delibere CIPE [1] …",
        refused=False, refusal_reason=None, support_score=0.0, cited_passages=[1],
        cited_chunk_ids=["scheda::00-contesto"], invalid_citations=[], claims=[],
        passages=[
            Passage(1, "scheda::00-contesto", "scheda del corpus — 00-contesto", "Le delibere CIPE…", in_index=False),
            Passage(2, "scheda::perimetro", "perimetro calcolato", "Perimetro: 511 delibere", in_index=False),
            Passage(3, "c-42", "Delibera n. 1/2024", "testo dall'indice", in_index=True),
        ],
        usage={}, raw_output="", model="test", params={}, route="meta",
    )
    w = AuditWriter("run-meta", str(tmp_path))
    w.record(res, corpus_version="v1", cache_enabled=False)
    riga = json.loads(w.path.read_text(encoding="utf-8").strip())

    assert set(riga["passages_inline"]) == {"scheda::00-contesto", "scheda::perimetro"}
    assert riga["passages_inline"]["scheda::perimetro"]["text"].endswith("511 delibere")
    assert riga["passages_inline"]["scheda::00-contesto"]["source"].startswith("scheda del corpus")
    assert riga["retrieved_chunk_ids"] == ["scheda::00-contesto", "scheda::perimetro", "c-42"]
