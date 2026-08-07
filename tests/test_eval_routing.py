"""Metriche di routing, e la trappola del `source_id_ok`.

Le metriche automatiche misurano la forma. Una risposta strutturata è perfetta e non ha
chunk citati: senza un ramo dedicato la metrica la segnerebbe come priva di fonte. È
l'errore già visto tre volte, col segno invertito.
"""

from config import RagConfig
from eval.runner import EvalItem, _aggregate, _row, _verdict
from rag.generation import RagResult
from rag.outcomes import StructuredOutcome


def _res_strutturato(valore=93):
    return RagResult(
        query="q", answer_text="Risultano 93 delibere…", refused=False, refusal_reason=None,
        support_score=0.0, cited_passages=[], cited_chunk_ids=[], invalid_citations=[],
        claims=[], passages=[], usage={}, raw_output="", model="m", params={},
        route="structured",
        structured=StructuredOutcome(intent="count_delibere", sql="SELECT …", params=[2024],
                                     rows=[{"n": valore}], n_rows=1, computed_value=valore),
    )


def _item(**kw):
    base = dict(id="bd-03", category="aggregazione", expect_refuse=False,
                question="Quante delibere nel 2024?", route_attesa="structured", expected_value=93)
    base.update(kw)
    return EvalItem(**base)


def test_la_risposta_strutturata_non_viene_segnata_senza_fonte():
    riga = _row(RagConfig(), _item(), "off", _res_strutturato())
    assert riga["source_id_ok"] == 1     # la fonte è la query, non un chunk_id
    assert riga["route"] == "structured"


def test_il_valore_calcolato_viene_confrontato_con_la_verita():
    assert _row(RagConfig(), _item(), "off", _res_strutturato(93))["value_ok"] == 1
    assert _row(RagConfig(), _item(), "off", _res_strutturato(2))["value_ok"] == 0


def test_route_ok_confronta_la_route_attesa():
    riga = _row(RagConfig(), _item(route_attesa="pointwise"), "off", _res_strutturato())
    assert riga["route_ok"] == 0


def test_le_metriche_di_routing_separano_richiamo_e_falsi_positivi():
    """Mai una media unica: sono due errori in direzioni opposte."""
    righe = [
        _row(RagConfig(), _item(id="a"), "off", _res_strutturato()),                       # atteso structured, ok
        _row(RagConfig(), _item(id="b", route_attesa="structured"), "off", _pointwise()),  # fall-through
        _row(RagConfig(), _item(id="c", route_attesa="pointwise", expected_value=None), "off", _pointwise()),
    ]
    agg = _aggregate(RagConfig(), righe, "off")
    assert agg["router_recall"] == 0.5          # 1 su 2
    assert agg["router_false_positive"] == 0.0  # 0 su 1


def _pointwise():
    return RagResult(
        query="q", answer_text="risposta", refused=False, refusal_reason=None, support_score=0.9,
        cited_passages=[1], cited_chunk_ids=["c1"], invalid_citations=[], claims=[], passages=[],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cached_tokens": 0},
        raw_output="{}", model="m", params={}, route="pointwise",
    )


def test_verdict_mostra_intento_e_numero():
    assert "count_delibere" in _verdict(_res_strutturato())
    assert "93" in _verdict(_res_strutturato())
