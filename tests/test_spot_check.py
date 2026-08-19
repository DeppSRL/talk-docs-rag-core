"""Selezione degli item da giudicare e ricostruzione del contesto.

Due rischi specifici:
- includere nel modulo i rifiuti deterministici (nulla da giudicare: la pipeline ha
  rifiutato *prima* di chiamare il modello) gonfierebbe il denominatore della fedeltà;
- degradare in silenzio sulle run anteriori a ``retrieved_chunk_ids``, mostrando i soli
  passaggi citati come se fossero il contesto completo, renderebbe la colonna ``causa``
  falsa invece che vuota.
"""

import json

from talk_docs_rag_core.eval.spot_check import COLONNE_FORM, _contesto, _da_giudicare, _risposta, _tipo, build_form


def _rec(query="d", *, refused=False, from_cache=False, cache_enabled=False, retrieved=None, cited=None):
    r = {
        "query": query,
        "refused": refused,
        "from_cache": from_cache,
        "cache_enabled": cache_enabled,
        "answer_text": "risposta",
        "cited_chunk_ids": cited if cited is not None else [],
    }
    if retrieved is not None:
        r["retrieved_chunk_ids"] = retrieved
    return r


def test_risposta_preferisce_answer_text():
    assert _risposta({"answer_text": "la risposta", "raw_output": '{"answer": "altro"}'}) == "la risposta"


def test_risposta_recupera_da_raw_output_sulle_run_vecchie():
    """Difetto misurato: senza questo fallback la scheda mostrava «—» su tutte le righe e si
    sarebbe giudicata una risposta invisibile."""
    raw = json.dumps({"answer": "Il CIPESS è presieduto dal Presidente del Consiglio [1].", "claims": []})
    assert _risposta({"raw_output": raw}).startswith("Il CIPESS è presieduto")


def test_risposta_su_raw_output_non_json_restituisce_il_grezzo():
    assert _risposta({"raw_output": "testo non JSON"}) == "testo non JSON"


def test_risposta_irrecuperabile_resta_vuota():
    """Hit di cache semantica su run vecchia: raw_output è vuoto, il testo non esiste."""
    assert _risposta({"raw_output": "", "from_cache": True}) == ""
    assert _risposta({}) == ""


def test_form_riporta_la_risposta_anche_senza_answer_text():
    raw = json.dumps({"answer": "295.178.000 euro [1].", "claims": []})
    rec = _rec("d", cited=["x"])
    del rec["answer_text"]  # run anteriore al campo: il testo esiste solo in raw_output
    rec["raw_output"] = raw
    rows = build_form([rec], "run-test", {"d": ("ic-07", "in_corpus")})
    assert rows[0]["risposta"] == "295.178.000 euro [1]."


def test_tipo_distingue_i_tre_esiti():
    assert _tipo(_rec(refused=True)) == "rifiuto"
    assert _tipo(_rec(from_cache=True)) == "hit-cache"
    assert _tipo(_rec()) == "risposta"


def test_rifiuti_esclusi_dal_modulo_hit_cache_inclusi():
    """Gli hit di cache semantica servono la risposta di una domanda *diversa*: se sia fedele
    alla domanda nuova è il rischio proprio della cache, e va giudicato."""
    assert _da_giudicare(_rec()) is True
    assert _da_giudicare(_rec(from_cache=True)) is True
    assert _da_giudicare(_rec(refused=True)) is False


def test_contesto_usa_i_recuperati_non_i_citati():
    rec = _rec(retrieved=["a", "b", "c"], cited=["b"])
    assert _contesto(rec) == ["a", "b", "c"]


def test_contesto_degrada_ai_citati_sulle_run_vecchie():
    """Run anteriore al campo: nessun retrieved_chunk_ids → si mostrano i citati."""
    rec = _rec(cited=["b"])
    assert "retrieved_chunk_ids" not in rec
    assert _contesto(rec) == ["b"]


def test_contesto_vuoto_non_esplode():
    assert _contesto(_rec()) == []


def test_build_form_salta_i_rifiuti_e_riempie_gli_id():
    records = [
        _rec("prima", retrieved=["x"], cited=["x"]),
        _rec("rifiutata", refused=True),
        _rec("terza", from_cache=True, cache_enabled=True),
    ]
    idx = {"prima": ("ic-01", "in_corpus"), "terza": ("ic-02", "in_corpus")}
    rows = build_form(records, "run-test", idx)

    assert [r["id"] for r in rows] == ["ic-01", "ic-02"]
    assert [r["tipo"] for r in rows] == ["risposta", "hit-cache"]
    assert [r["condition"] for r in rows] == ["off", "on"]
    # Le colonne di giudizio nascono vuote: nessun default silenzioso.
    for r in rows:
        assert r["fedele"] == ""
        assert r["causa"] == ""
        assert r["citazione_corretta"] == ""
        assert set(r) == set(COLONNE_FORM)


def test_build_form_domanda_fuori_eval_set_resta_senza_id():
    """Una query non presente nell'eval set (es. registrata da `app ask`) non deve far
    fallire il modulo: id vuoto, riga comunque giudicabile."""
    rows = build_form([_rec("ignota")], "run-test", {})
    assert len(rows) == 1
    assert rows[0]["id"] == ""
    assert rows[0]["domanda"] == "ignota"
