"""Aggregazione delle metriche di eval.

Difetto misurato sulla run ``eval-20260805T061415Z``: il report dichiarava «Chiamate al
modello: 27» su 27 domande, mentre le chiamate reali erano **20** — i 7 rifiuti
deterministici (C3) non chiamano il modello, escono prima. Il conteggio escludeva solo gli
hit di cache semantica.

Non è cosmetico: ``model_calls`` è il denominatore di ``provider_hit_rate``, quindi
gonfiarlo **sottostima** l'hit-rate del prompt caching, che è la metrica centrale del PoC.
"""

from config import RagConfig
from eval.runner import _aggregate


def _riga(condition="off", **kw):
    # Le colonne di routing/verbatim (incremento 1) sono lette **strette** da `_aggregate`:
    # una metrica del deliverable che diventa silenziosamente 0.0 perché una colonna è
    # sparita è peggio di un KeyError. Qui valgono "non punteggiato", come su una riga
    # pointwise senza `route_attesa`.
    base = {
        "condition": condition,
        "refusal_correct": 1,
        "refused": 0,
        "source_id_ok": 1,
        "invalid_citations": 0,
        "from_cache_semantic": 0,
        "prompt_tokens": 2000,
        "completion_tokens": 100,
        "cached_tokens": 0,
        "total_tokens": 2100,
        "cost": 0.0004,
        "latency_s": 2.0,
        "route": "pointwise",
        "route_attesa": "",
        "route_ok": "",
        "value_ok": "",
        "verbatim_valid_ratio": "",
        "verbatim_misattributed": "",
        "verbatim_not_found": "",
        "router_source": "lexical",
        "router_llm_error": "",
        "router_llm_tokens": 0,
        "router_llm_cost": 0.0,
    }
    base.update(kw)
    return base


def _rifiuto(condition="off"):
    """Rifiuto deterministico: nessuna chiamata, usage vuoto."""
    return _riga(
        condition,
        refused=1,
        source_id_ok="",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        cost=0.0,
        latency_s=1.0,
    )


def test_rifiuti_non_contano_come_chiamate_al_modello():
    rows = [_riga() for _ in range(3)] + [_rifiuto() for _ in range(2)]
    agg = _aggregate(RagConfig(), rows, "off")
    assert agg["n"] == 5
    assert agg["model_calls"] == 3


def test_hit_cache_semantica_non_conta_come_chiamata():
    rows = [_riga(), _riga(from_cache_semantic=1, prompt_tokens=0)]
    agg = _aggregate(RagConfig(), rows, "off")
    assert agg["model_calls"] == 1


def test_provider_hit_rate_usa_le_chiamate_reali_come_denominatore():
    """2 chiamate su 2 con cached_tokens>0 e 3 rifiuti → 100%, non 40%."""
    rows = [_riga(cached_tokens=1500) for _ in range(2)] + [_rifiuto() for _ in range(3)]
    agg = _aggregate(RagConfig(), rows, "off")
    assert agg["model_calls"] == 2
    assert agg["provider_hit_rate"] == 1.0


def test_provider_hit_rate_parziale():
    rows = [_riga(cached_tokens=1500), _riga(cached_tokens=0), _rifiuto()]
    agg = _aggregate(RagConfig(), rows, "off")
    assert agg["model_calls"] == 2
    assert agg["provider_hit_rate"] == 0.5


def test_solo_rifiuti_non_divide_per_zero():
    agg = _aggregate(RagConfig(), [_rifiuto(), _rifiuto()], "off")
    assert agg["model_calls"] == 0
    assert agg["provider_hit_rate"] == 0.0


def test_accuratezza_rifiuto_ignora_le_borderline():
    """Le borderline hanno refusal_correct vuoto: non entrano nel punteggio."""
    rows = [_riga(refusal_correct=1), _riga(refusal_correct=0), _riga(refusal_correct="")]
    agg = _aggregate(RagConfig(), rows, "off")
    assert agg["refusal_accuracy"] == 0.5


def test_condizione_filtra_le_righe():
    rows = [_riga("off"), _riga("on"), _riga("on")]
    assert _aggregate(RagConfig(), rows, "on")["n"] == 2


def test_le_classificazioni_fallite_del_router_sono_contate():
    """Una chiamata di routing morta ricade sul lessicale e serve comunque una route: non
    è un instradamento sbagliato, è un dato mancante travestito da decisione. Sulla run
    `eval-20260808T122852Z` quattro fallimenti su 55 hanno depresso il richiamo del router
    sotto quello lessicale, e la cosa si vedeva solo scavando nell'audit."""
    rows = [
        _riga(router_source="llm", router_llm_tokens=1500, router_llm_cost=0.0003),
        _riga(router_source="lexical", router_llm_tokens=0, router_llm_error="chiamata fallita: RateLimitError"),
    ]
    agg = _aggregate(RagConfig(), rows, "off")
    assert agg["router_llm_failed"] == 1
    assert agg["router_llm_decisions"] == 1
