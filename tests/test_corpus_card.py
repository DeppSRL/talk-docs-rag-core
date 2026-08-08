"""La scheda del corpus: si carica in ordine, degrada in assenza, esclude il README.

Il contratto che conta è il degrado: una directory assente o vuota deve dare ``None``
(la pipeline serve senza scheda), non un'eccezione a costruzione della pipeline.
"""

from rag.corpus_card import CorpusCard


def _scrivi(tmp_path, nome, testo):
    (tmp_path / nome).write_text(testo, encoding="utf-8")


def test_sezioni_in_ordine_di_nome(tmp_path):
    _scrivi(tmp_path, "10-struttura.md", "struttura")
    _scrivi(tmp_path, "00-contesto.md", "contesto")
    card = CorpusCard.load(tmp_path)
    assert [nome for nome, _ in card.sections] == ["00-contesto", "10-struttura"]
    assert card.text == "contesto\n\nstruttura"


def test_il_readme_non_entra_nella_scheda(tmp_path):
    """Il README documenta le regole di compilazione, non il corpus: nel prompt del
    router e nella risposta meta sarebbe rumore che parla di noi invece che dei dati."""
    _scrivi(tmp_path, "README.md", "regole di compilazione")
    _scrivi(tmp_path, "00-contesto.md", "contesto")
    card = CorpusCard.load(tmp_path)
    assert "regole di compilazione" not in card.text


def test_directory_assente_da_none(tmp_path):
    assert CorpusCard.load(tmp_path / "non-esiste") is None


def test_directory_senza_sezioni_da_none(tmp_path):
    _scrivi(tmp_path, "README.md", "solo regole")
    assert CorpusCard.load(tmp_path) is None


def test_sezione_vuota_scartata(tmp_path):
    _scrivi(tmp_path, "00-vuota.md", "   \n")
    _scrivi(tmp_path, "10-piena.md", "testo")
    card = CorpusCard.load(tmp_path)
    assert [nome for nome, _ in card.sections] == ["10-piena"]
