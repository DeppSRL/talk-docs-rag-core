"""Tabella strutturata dei documenti del corpus.

Il rischio da coprire non è «DuckDB funziona»: è che la tabella dica cose diverse dal
titolo mostrato nelle citazioni, e che un file non-delibera finisca dentro un conteggio
di delibere. I test usano manifest sintetici: i conteggi reali del corpus vivono
nell'eval set, non qui, così una nuova `app ingest` non rompe i test unitari.
"""

import json

import pytest

from talk_docs_rag_core.structured.store import StructuredStore

MANIFEST = {
    "corpus_version": "test",
    "files": [
        {"path": "delibere/2019/E190001.txt", "title": "Delibera CIPE n. 1/2019", "content_hash": "a", "n_chunks": 3},
        {"path": "delibere/2019/E190002.txt", "title": "Delibera CIPE n. 2/2019", "content_hash": "b", "n_chunks": 4},
        # spezzata solo perché su una riga sola sforerebbe i 120 caratteri di ruff
        {
            "path": "delibere/2024/E240093.txt",
            "title": "Delibera CIPESS n. 93/2024",
            "content_hash": "c",
            "n_chunks": 9,
        },
        {"path": "altro/relazione-annuale.md", "title": "Relazione annuale", "content_hash": "d", "n_chunks": 2},
    ],
}


@pytest.fixture
def store():
    return StructuredStore.from_manifest(MANIFEST)


def test_deriva_anno_numero_e_comitato(store):
    righe = store.query("SELECT anno, numero, comitato FROM documenti WHERE codice = ?", ["E240093"])
    assert righe == [{"anno": 2024, "numero": 93, "comitato": "CIPESS"}]


def test_il_non_delibera_non_entra_nei_conteggi(store):
    """Il PoC è nato con documenti campione: un file senza codice non è una delibera."""
    tutte = store.query("SELECT COUNT(*) AS n FROM documenti", [])
    delibere = store.query("SELECT COUNT(*) AS n FROM documenti WHERE is_delibera", [])
    assert tutte[0]["n"] == 4
    assert delibere[0]["n"] == 3


def test_conteggio_per_anno(store):
    righe = store.query("SELECT COUNT(*) AS n FROM documenti WHERE is_delibera AND anno = ?", [2019])
    assert righe[0]["n"] == 2


def test_manifest_vuoto_non_esplode():
    vuoto = StructuredStore.from_manifest({"files": []})
    assert vuoto.query("SELECT COUNT(*) AS n FROM documenti", [])[0]["n"] == 0


def test_from_path_legge_il_manifest(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(MANIFEST), encoding="utf-8")
    assert StructuredStore.from_path(p).query("SELECT COUNT(*) AS n FROM documenti", [])[0]["n"] == 4


def test_from_path_inesistente_restituisce_none(tmp_path):
    """Senza manifest il router va spento, non deve far fallire la costruzione della pipeline."""
    assert StructuredStore.from_path(tmp_path / "non-esiste.json") is None
