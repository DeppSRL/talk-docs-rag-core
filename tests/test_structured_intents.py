"""Semantic layer: tre intenti tipizzati, niente SQL generato.

I test fissano due proprietà: che nessun valore dell'utente finisca nella *stringa* SQL
(solo come parametro), e che un intento con parametri non legabili restituisca None — la
route degrada a rifiuto dichiarato invece di eseguire una query su un filtro indovinato.
"""

from structured import intents
from structured.store import StructuredStore

MANIFEST = {
    "files": [
        {"path": "delibere/2019/E190001.txt", "title": "Delibera CIPE n. 1/2019", "content_hash": "a", "n_chunks": 1},
        {"path": "delibere/2021/E210001.txt", "title": "Delibera CIPESS n. 1/2021", "content_hash": "b", "n_chunks": 1},
        {"path": "delibere/2021/E210005.txt", "title": "Delibera CIPESS n. 5/2021", "content_hash": "c", "n_chunks": 1},
    ]
}


def test_count_con_anno_lega_il_parametro():
    sql, params = intents.build(intents.COUNT_DELIBERE, {"anno": 2021})
    assert "2021" not in sql  # il valore NON entra nella stringa
    assert params == [2021]
    assert "COUNT(*)" in sql


def test_count_esegue_e_conta_solo_le_delibere():
    store = StructuredStore.from_manifest(MANIFEST)
    sql, params = intents.build(intents.COUNT_DELIBERE, {"anno": 2021})
    righe = store.query(sql, params)
    assert righe[0]["n"] == 2
    assert righe[0]["max_numero"] == 5


def test_count_filtra_per_comitato():
    store = StructuredStore.from_manifest(MANIFEST)
    sql, params = intents.build(intents.COUNT_DELIBERE, {"comitato": "CIPE"})
    assert store.query(sql, params)[0]["n"] == 1


def test_count_su_intervallo_di_anni():
    store = StructuredStore.from_manifest(MANIFEST)
    sql, params = intents.build(intents.COUNT_DELIBERE, {"anno_da": 2020, "anno_a": 2022})
    assert store.query(sql, params)[0]["n"] == 2


def test_list_ordina_e_limita():
    store = StructuredStore.from_manifest(MANIFEST)
    sql, params = intents.build(intents.LIST_DELIBERE, {"anno": 2021, "limit": 1})
    righe = store.query(sql, params)
    assert len(righe) == 1
    assert righe[0]["numero"] == 1  # ORDER BY anno, numero


def test_count_by_year_raggruppa():
    store = StructuredStore.from_manifest(MANIFEST)
    sql, params = intents.build(intents.COUNT_BY_YEAR, {})
    assert store.query(sql, params) == [{"anno": 2019, "n": 1}, {"anno": 2021, "n": 2}]


def test_intento_sconosciuto_restituisce_none():
    assert intents.build("somma_importi", {"anno": 2024}) is None
