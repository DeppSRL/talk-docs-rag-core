"""Guardiano di astensione (C3b).

Il rischio da coprire non è «funziona», è **non deve diventare la causa dei guasti**: un
guardiano troppo zelante trasforma risposte corrette in astensioni. Perciò i test fissano i
casi misurati che lo hanno tarato, compresi i due artefatti che producevano falsi allarmi:
`qual` trattato come termine di contenuto, e la sillabazione `intel- ligenza` da estrazione PDF.
"""

import math

import pytest

from talkdocs_rag_core.rag.guard import TermStats, abstention_signal, content_terms, missing_terms


def test_content_terms_scarta_stopword_e_interrogativi():
    t = content_terms("Qual è la differenza tra CIPE e CIPESS?")
    assert "qual" not in t  # senza questo produceva 3 falsi allarmi da solo
    assert "differenza" not in t
    assert "cipess" in t


def test_content_terms_spezza_le_elisioni():
    """«dell'intelligenza» deve dare «intelligenza», non «dell'intelligenza»."""
    t = content_terms("Che cosa ha raccomandato sull'impiego dell'intelligenza artificiale?")
    assert "intelligenza" in t
    assert "impiego" in t
    assert not any("'" in x for x in t)


def test_content_terms_ignora_parole_corte():
    """«di», «e», «il» cadono per lunghezza; «capo» resta, è un termine di contenuto."""
    assert content_terms("chi è il capo di ANAS e CIPE") == ["anas", "capo", "cipe"]


def test_missing_terms_assorbe_la_morfologia():
    """«premiali» non deve risultare mancante se il passaggio dice «quote premiali»; e
    «ferroviaria» non deve mancare se il testo dice «ferroviario» (radice condivisa)."""
    p = ["euro 295.178.000 accantonati per le quote premiali per l'anno 2020"]
    assert missing_terms("quote premiali 2020", p) == []
    assert missing_terms("linea ferroviaria", ["la linea di collegamento ferroviario"]) == []


def test_missing_terms_riunisce_la_sillabazione_da_pdf():
    """Caso reale: il corpus contiene «intel- ligenza artificiale» spezzato dall'estrazione
    PDF. Era la causa degli unici 2 falsi allarmi del guardiano."""
    p = ["si invita a garantire la centralità della supervisione umana nei sistemi di intel- ligenza artificiale"]
    assert missing_terms("supervisione umana intelligenza artificiale", p) == []


def test_missing_terms_trova_cio_che_manca_davvero():
    p = ["accantonamento di 32,5 milioni per obiettivi di ricerca sul Fondo sanitario nazionale 2020"]
    assert "premiali" in missing_terms("somme per le quote premiali 2020 del Fondo sanitario", p)


def _stats(df, n=13670):
    return TermStats(n_chunks=n, df=df)


def test_idf_alto_per_termine_raro_basso_per_comune():
    s = _stats({"premia": 50, "fondo": 5000})
    assert s.idf("premiali") > 5.0
    assert s.idf("fondo") < 2.0


def test_termine_assente_dal_corpus_ha_idf_massimo():
    s = _stats({})
    assert s.idf("carbonara") == pytest.approx(math.log(13670), abs=0.01)


def test_segnale_zero_se_nulla_manca():
    s = _stats({"premia": 50})
    seg, manc = abstention_signal("quote premiali", ["le quote premiali del 2020"], s)
    assert seg == 0.0
    assert manc == []


def test_segnale_e_idf_del_mancante_piu_raro():
    """Il segnale NON è la frazione coperta: la copertura piatta diluiva «premiali» in 6
    termini banali e non discriminava (misurato: infedeltà a 0,86 e 1,00, sopra 9 corrette).

    Qui manca solo «premiali»; «quote», «fondo» e «sanitario» sono nel passaggio.
    """
    s = _stats({"premia": 50, "quote": 3000, "fondo": 5000, "sanita": 4000})
    seg, manc = abstention_signal(
        "quote premiali fondo sanitario", ["le quote del fondo sanitario nazionale"], s
    )
    assert manc == ["premiali"]
    assert seg == pytest.approx(s.idf("premiali"), abs=1e-9)


def test_un_termine_comune_mancante_non_fa_scattare_nulla_di_grave():
    """Il segnale deve restare BASSO se manca solo un termine banale: è ciò che evita di
    trasformare risposte corrette in astensioni."""
    s = _stats({"proven": 4000, "sport": 300})
    seg, manc = abstention_signal("provenienti dallo sport", ["fondo per lo sport"], s)
    assert manc == ["provenienti"]
    assert seg < 2.0


def test_senza_statistiche_il_guardiano_non_scatta():
    """Meglio non astenersi che astenersi su un peso inventato: è una rete di sicurezza."""
    seg, manc = abstention_signal("quote premiali", ["testo che non le contiene"], None)
    assert seg == 0.0
    assert manc == ["premiali", "quote"]


def test_termstats_round_trip(tmp_path):
    s = _stats({"premia": 50, "fondo": 5000})
    p = tmp_path / "df.json"
    s.save(p)
    letto = TermStats.load(p)
    assert letto is not None
    assert letto.n_chunks == s.n_chunks
    assert letto.idf("premiali") == pytest.approx(s.idf("premiali"))


def test_termstats_load_assente_o_corrotto(tmp_path):
    assert TermStats.load(tmp_path / "manca.json") is None
    rotto = tmp_path / "rotto.json"
    rotto.write_text("{non json", encoding="utf-8")
    assert TermStats.load(rotto) is None  # meglio ricalcolare che fidarsi


def test_from_documents_conta_i_documenti_non_le_occorrenze():
    s = TermStats.from_documents(["fondo fondo fondo sanitario", "fondo nazionale"])
    assert s.n_chunks == 2
    assert s.df["fondo"] == 2  # document frequency, non term frequency
    assert s.df["sanita"] == 1
