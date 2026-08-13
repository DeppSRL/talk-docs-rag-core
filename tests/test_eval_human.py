"""Rilettura del giudizio umano: parsing tollerante e conteggi che non mentono.

Il rischio specifico di questo modulo: una riga **non compilata** contata come corretta
gonficherebbe la fedeltà e farebbe promuovere il prototipo su un campione vuoto. Le righe
vuote devono essere *escluse*, e la copertura dichiarata.
"""

import pytest

from talkdocs_rag_core.eval.eval_human import aggregate, banda, parse_bool, parse_causa, parse_voto


@pytest.mark.parametrize("raw", ["sì", "si", "SI", " Sì ", "s", "1", "true", "y", "yes", "ok"])
def test_parse_bool_vero(raw):
    assert parse_bool(raw) is True


@pytest.mark.parametrize("raw", ["no", "NO", " n ", "0", "false", "falso"])
def test_parse_bool_falso(raw):
    assert parse_bool(raw) is False


@pytest.mark.parametrize("raw", ["", "   ", None, "forse", "boh"])
def test_parse_bool_non_giudicato(raw):
    """Ambiguo o vuoto → None. Mai un default silenzioso."""
    assert parse_bool(raw) is None


@pytest.mark.parametrize(("raw", "atteso"), [("1", 1), ("5", 5), ("3", 3), ("4,0", 4), ("3.6", 4)])
def test_parse_voto_valido(raw, atteso):
    assert parse_voto(raw) == atteso


@pytest.mark.parametrize("raw", ["", None, "0", "6", "-1", "ottimo"])
def test_parse_voto_fuori_scala_o_ignoto(raw):
    assert parse_voto(raw) is None


@pytest.mark.parametrize(
    ("raw", "atteso"),
    [
        ("retrieval", "retrieval"),
        ("Retrieval", "retrieval"),
        (" r ", "retrieval"),
        ("recupero", "retrieval"),
        ("generazione", "generazione"),
        ("gen", "generazione"),
        ("modello", "generazione"),
        ("", None),
        (None, None),
        ("boh", None),
    ],
)
def test_parse_causa(raw, atteso):
    assert parse_causa(raw) == atteso


def _riga(ident, fedele="", cit="", ita="", causa="", **kw):
    base = {
        "id": ident,
        "fedele": fedele,
        "causa": causa,
        "citazione_corretta": cit,
        "italiano_1_5": ita,
        "domanda": f"d-{ident}",
    }
    base.update(kw)
    return base


def test_cause_contate_solo_sulle_infedeltà():
    rows = [
        _riga("a", "sì", causa="retrieval"),  # fedele: la causa non va contata
        _riga("b", "no", causa="retrieval"),
        _riga("c", "no", causa="generazione"),
        _riga("d", "no", causa="gen"),
    ]
    agg = aggregate(rows)
    assert agg["n_infedeli"] == 3
    assert agg["cause"] == {"retrieval": 1, "generazione": 2, "strutturale": 0}
    assert agg["infedeli_senza_causa"] == []


def test_infedelta_senza_causa_viene_segnalata():
    """Un «no» senza causa non dice dove investire: va dichiarato, non ignorato."""
    rows = [_riga("ic-07-bis", "no"), _riga("ic-04-bis", "no", causa="retrieval")]
    agg = aggregate(rows)
    assert agg["infedeli_senza_causa"] == ["ic-07-bis"]
    assert agg["cause"]["retrieval"] == 1


def test_righe_vuote_escluse_non_contate_come_corrette():
    """Il difetto da evitare: 2 giudizi su 10 righe non fanno 20% di fedeltà, fanno 100%
    su copertura 20% — e la copertura va dichiarata."""
    rows = [_riga("a", "sì"), _riga("b", "sì")] + [_riga(f"v{i}") for i in range(8)]
    agg = aggregate(rows)
    assert agg["n_totali"] == 10
    assert agg["n_giudicati"] == 2
    assert agg["fedelta"] == 1.0
    assert agg["copertura"] == 0.2


def test_fedelta_e_lista_infedeli():
    rows = [_riga("ic-07", "sì"), _riga("ic-07-bis", "no"), _riga("ic-08", "sì")]
    agg = aggregate(rows)
    assert agg["n_fedeli"] == 2
    assert agg["n_infedeli"] == 1
    assert agg["fedelta"] == pytest.approx(2 / 3, abs=1e-4)
    assert agg["infedeli_ids"] == ["ic-07-bis"]


def test_citazione_e_italiano_indipendenti_dalla_fedelta():
    """Fedele ma attribuita al documento sbagliato: due metriche distinte."""
    rows = [_riga("a", "sì", cit="no", ita="4"), _riga("b", "sì", cit="sì", ita="2")]
    agg = aggregate(rows)
    assert agg["fedelta"] == 1.0
    assert agg["citazione_corretta"] == 0.5
    assert agg["n_citazioni_sbagliate"] == 1
    assert agg["italiano_medio"] == 3.0


def test_riga_con_solo_italiano_conta_come_giudicata_ma_non_per_fedelta():
    agg = aggregate([_riga("a", ita="5")])
    assert agg["n_giudicati"] == 1
    assert agg["fedelta"] is None
    assert agg["italiano_medio"] == 5.0


def test_modulo_vuoto_non_divide_per_zero():
    agg = aggregate([_riga("a"), _riga("b")])
    assert agg["n_giudicati"] == 0
    assert agg["fedelta"] is None
    assert agg["citazione_corretta"] is None
    assert agg["italiano_medio"] is None


def test_aggregate_su_lista_vuota():
    agg = aggregate([])
    assert agg["n_totali"] == 0
    assert agg["copertura"] == 0.0
    assert agg["fedelta"] is None


@pytest.mark.parametrize(
    ("fedelta", "atteso"),
    [
        (1.0, "ALTA"),
        (0.95, "ALTA"),
        (0.949, "MEDIA"),
        (0.89, "MEDIA"),
        (0.85, "MEDIA"),
        (0.84, "BASSA"),
        (0.0, "BASSA"),
    ],
)
def test_bande_di_decisione(fedelta, atteso):
    assert banda(fedelta)[0] == atteso


def test_banda_senza_giudizi_non_raccomanda_nulla():
    etichetta, testo = banda(None)
    assert etichetta == "INSUFFICIENTE"
    assert "non si può decidere" in testo
