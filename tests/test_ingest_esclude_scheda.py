"""La scheda del corpus non è un documento del corpus.

Difetto misurato il 2026-08-08 guardando l'output di un ingest: `corpus/delibere/card/*.md`
finiva nell'indice come tre documenti (manifest a 514 file invece di 511, in tutte le run
dell'8 agosto). Nessuna risposta ha mai recuperato quei chunk — verificato sulle tuple di
audit delle tre run — quindi i numeri misurati restano validi; ma il difetto è latente e
la sua conseguenza è di natura, non di conteggio: la scheda è testo **nostro**, che
descrive che cosa il sistema sa fare. Se venisse recuperata, una risposta potrebbe
sostenere un'affermazione sul corpus citando ciò che abbiamo scritto noi sul corpus.
"""

from pathlib import Path

from ingest.pipeline import _discover_files


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    (corpus / "delibere" / "2024").mkdir(parents=True)
    (corpus / "delibere" / "2024" / "E240001.txt").write_text("testo della delibera", encoding="utf-8")
    card = corpus / "delibere" / "card"
    card.mkdir()
    (card / "00-contesto.md").write_text("che cos'è questo corpus", encoding="utf-8")
    (card / "README.md").write_text("regole di compilazione", encoding="utf-8")
    return corpus


def test_la_scheda_non_entra_nell_indice(tmp_path):
    corpus = _corpus(tmp_path)
    trovati = _discover_files(corpus, corpus / "delibere" / "card")
    assert [p.name for p in trovati] == ["E240001.txt"]


def test_senza_card_dir_il_comportamento_e_quello_di_prima(tmp_path):
    """L'esclusione è esplicita: chi chiama senza dichiarare la scheda ottiene tutto.
    Serve a non far dipendere il conteggio da una convenzione implicita sul nome."""
    corpus = _corpus(tmp_path)
    nomi = sorted(p.name for p in _discover_files(corpus))
    assert nomi == ["00-contesto.md", "E240001.txt"]   # il README resta escluso comunque


def test_una_directory_card_inesistente_non_rompe(tmp_path):
    corpus = _corpus(tmp_path)
    trovati = _discover_files(corpus, tmp_path / "non-esiste")
    assert any(p.name == "E240001.txt" for p in trovati)
