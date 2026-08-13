"""Il report non deve contraddire il sistema che descrive.

Difetto trovato sulla run ``eval-20260807T181416Z``: due stringhe cablate nel generatore
del report erano rimaste indietro rispetto all'incremento 1.

- l'atteso della categoria ``aggregazione`` diceva ancora «non calcolabile su k chunk»,
  mentre da quell'incremento le aggregative riconosciute **si calcolano** sul manifest
  (6 valori su 6 esatti) e le altre escono in rifiuto dichiarato;
- l'intestazione dichiarava una tassonomia a tre categorie (``in_corpus / borderline /
  out_of_corpus``) abbandonata due giorni prima: ``borderline`` è spaccata in
  ``near_miss``/``aggregazione``, più ``vaga``.

Nessuna metrica ne è affetta — sono stringhe. Ma il report è l'artefatto che legge chi non
ha letto ``STATUS.md``, e in quella riga dichiarava ancora aperto il difetto n.1. La
tassonomia ora si **deriva dai dati** invece di essere cablata: una lista che si scrive da
sé non può ri-desincronizzarsi al prossimo cambio di categorie.
"""

from talkdocs_rag_core.config import RagConfig
from talkdocs_rag_core.eval.runner import _aggregate, _markdown, _per_categoria


def _riga(category, condition="off", **kw):
    base = {
        "condition": condition,
        "category": category,
        "refusal_correct": 1,
        "refused": 0,
        "uncertain": 0,
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


def _intestazione_eval_set(md: str) -> str:
    return next(riga for riga in md.splitlines() if riga.startswith("- **eval set:**"))


def test_atteso_dell_aggregazione_non_dichiara_piu_il_difetto_aperto():
    """L'incremento 1 ha reso calcolabile ciò che il report dice non calcolabile."""
    righe = "\n".join(_per_categoria([_riga("aggregazione")], "off"))
    assert "non calcolabile" not in righe
    assert "calcolata" in righe and "dichiarato" in righe


def test_le_altre_categorie_conservano_il_proprio_atteso():
    """Il fix non deve appiattire le categorie su un unico atteso."""
    rows = [_riga("in_corpus"), _riga("out_of_corpus"), _riga("vaga"), _riga("near_miss")]
    righe = "\n".join(_per_categoria(rows, "off"))
    assert "| `in_corpus` | 1 | 1 | 0 | 0 | risposta |" in righe
    assert "| `out_of_corpus` | 1 | 1 | 0 | 0 | rifiuto |" in righe
    assert "astensione" in righe


def _md(rows):
    cfg = RagConfig()
    return _markdown(cfg, _aggregate(cfg, rows, "off"), _aggregate(cfg, rows, "on"), len(rows), "sha", rows)


def test_intestazione_elenca_le_categorie_realmente_presenti():
    rows = [_riga("in_corpus"), _riga("aggregazione"), _riga("vaga"), _riga("out_of_corpus")]
    riga = _intestazione_eval_set(_md(rows))
    for cat in ("in_corpus", "aggregazione", "vaga", "out_of_corpus"):
        assert cat in riga
    assert "borderline" not in riga


def test_intestazione_segue_i_dati_se_le_categorie_cambiano():
    """La lista si deriva: una categoria nuova ci entra senza toccare il codice."""
    riga = _intestazione_eval_set(_md([_riga("in_corpus"), _riga("categoria_nuova")]))
    assert "categoria_nuova" in riga
