"""Il ramo aggregativo come RagResult: zero chiamate al modello, zero token.

La proprietà da fissare è che questo ramo non abbia `usage`: se comparissero token, il
confronto A/B del caching starebbe misurando una chiamata che non esiste.
"""

import logging
import re

from talk_docs_rag_core.config import RagConfig
from talk_docs_rag_core.rag import router
from talk_docs_rag_core.structured.service import _TESTI_RIFIUTO, serve_structured, serve_uncovered
from talk_docs_rag_core.structured.store import StructuredStore

MANIFEST = {
    "files": [
        {"path": "delibere/2024/E240001.txt", "title": "Delibera CIPESS n. 1/2024", "content_hash": "a", "n_chunks": 1},
        {"path": "delibere/2024/E240005.txt", "title": "Delibera CIPESS n. 5/2024", "content_hash": "b", "n_chunks": 1},
    ]
}


def test_conteggio_calcolato_senza_chiamare_il_modello():
    cfg = RagConfig()
    store = StructuredStore.from_manifest(MANIFEST)
    rotta = router.classify("Quante delibere ha adottato il CIPESS nel 2024?")
    res = serve_structured(cfg, store, "Quante delibere ha adottato il CIPESS nel 2024?", rotta)

    assert res.route == "structured"
    assert res.structured.computed_value == 2
    assert res.usage == {}                    # nessuna chiamata: nessun token
    assert res.raw_output == ""
    assert res.refused is False and res.uncertain is False
    assert res.structured.completeness == {"count": 2, "max_numero": 5, "gap": 3}
    assert "2" in res.answer_text


def test_intento_non_costruibile_degrada_a_rifiuto():
    """`intents.build` che torna None non deve produrre una query indovinata."""
    cfg = RagConfig()
    store = StructuredStore.from_manifest(MANIFEST)
    rotta = router.Route(router.STRUCTURED, intent="somma_importi", params={}, signals={})
    res = serve_structured(cfg, store, "quanto in totale", rotta)
    assert res.refused is True
    assert res.route == "uncovered"


def test_rifiuto_dichiarato_spiega_perche():
    """«non so» è meno utile di «so riconoscere la domanda, non ho la tabella»."""
    cfg = RagConfig()
    rotta = router.classify("Quanto è stato speso per le ferrovie?")
    res = serve_uncovered(cfg, "Quanto è stato speso per le ferrovie?", rotta)
    assert res.refused is True
    assert res.route == "uncovered"
    assert res.refusal_reason == "aggregazione fuori copertura: nessuna tabella degli importi"
    assert "non" in res.answer_text.lower()
    assert res.usage == {}


def test_il_rifiuto_dichiara_lo_stesso_perimetro_del_ramo_calcolato():
    """«su questo archivio» non è una variante stilistica di «sul corpus indicizzato».

    Il perimetro dichiarato esiste per non affermare nulla sul mondo fuori dal corpus. Un
    rifiuto che dice «non posso calcolarlo su questo archivio» promette una copertura —
    l'archivio CIPE/CIPESS 1967-2026 — che il corpus indicizzato non ha, e lo fa proprio
    nella frase che dovrebbe dichiarare i limiti del sistema. La risposta calcolata dice
    «corpus indicizzato»: il rifiuto deve dire la stessa cosa.
    """
    cfg = RagConfig()
    rotta = router.classify("Quanto è stato speso per le ferrovie?")
    res = serve_uncovered(cfg, "Quanto è stato speso per le ferrovie?", rotta)
    assert "corpus indicizzato" in res.answer_text
    assert "archivio" not in res.answer_text


def test_il_motivo_proprio_del_router_vince_sul_default():
    """Il multi-anno non è «non ho la tabella degli importi»: dirlo sarebbe falso."""
    cfg = RagConfig()
    rotta = router.classify("Quante delibere del CIPE nel 2019 e nel 2021?")
    res = serve_uncovered(cfg, "Quante delibere del CIPE nel 2019 e nel 2021?", rotta)
    assert res.refusal_reason == router.MOTIVO_ANNI_MULTIPLI
    assert res.router_signals["anni_multipli"] is True


def test_la_prosa_del_rifiuto_segue_il_motivo_e_non_lo_contraddice():
    """Il `refusal_reason` giusto con la prosa sbagliata è lo stesso guasto, un livello più su.

    Sul multi-anno il testo cablato sull'assenza degli importi diceva due cose false: che i
    metadati non bastano (bastano — è il semantic layer chiuso a non saper esprimere il filtro)
    e, in chiusura, offriva all'utente «posso invece contare le delibere per anno», cioè
    esattamente ciò che aveva appena chiesto.
    """
    cfg = RagConfig()
    domanda = "Quante delibere del CIPE nel 2019 e nel 2021?"
    res = serve_uncovered(cfg, domanda, router.classify(domanda))

    assert res.refusal_reason == router.MOTIVO_ANNI_MULTIPLI
    assert "tabella degli importi" not in res.answer_text
    assert "contare o elencare le delibere per anno" not in res.answer_text
    # Il perimetro dichiarato resta quello del ramo calcolato, come nel testo di default.
    assert "corpus indicizzato" in res.answer_text
    assert "archivio" not in res.answer_text


def test_un_motivo_non_mappato_cade_sul_testo_di_default_senza_esplodere():
    """La mappa motivo → prosa non deve diventare un `KeyError` in attesa di un motivo nuovo."""
    cfg = RagConfig()
    rotta = router.Route(router.UNCOVERED, reason="motivo inventato")
    res = serve_uncovered(cfg, "una domanda qualunque", rotta)

    assert res.refusal_reason == "motivo inventato"
    baseline = router.classify("Quanto è stato speso per le ferrovie?")
    atteso = serve_uncovered(cfg, "Quanto è stato speso per le ferrovie?", baseline)
    assert res.answer_text == atteso.answer_text
    assert "tabella degli importi" in res.answer_text


def test_ogni_motivo_emesso_dal_router_ha_una_prosa():
    """Un motivo nuovo senza testo deve rompere la suite qui, non mentire all'utente in produzione.

    `_TESTI_RIFIUTO` si legge con `.get(motivo, _TESTO_UNCOVERED)`: a runtime un motivo non
    mappato non esplode, mostra il testo di default — «non ho la tabella degli importi». Su un
    motivo diverso quel testo è falso, ed è di nuovo la divergenza audit/prosa che `reason`
    esiste per chiudere, silenziosa. La rete a runtime (un `warning`) la rende visibile a
    posteriori; questa asserzione la impedisce prima, in fase di sviluppo: `router.MOTIVI`
    elenca ciò che il router può emettere, e ogni voce deve avere qui la sua prosa.
    """
    assert set(router.MOTIVI) <= set(_TESTI_RIFIUTO)


def test_un_motivo_non_mappato_lascia_traccia_nei_log(caplog):
    """Il default silenzioso è il guasto invisibile: almeno deve nominarsi nei log."""
    cfg = RagConfig()
    rotta = router.Route(router.UNCOVERED, reason="motivo inventato")
    with caplog.at_level(logging.WARNING, logger="structured.service"):
        res = serve_uncovered(cfg, "una domanda qualunque", rotta)

    assert res.refused is True
    assert "motivo inventato" in caplog.text


def test_il_rifiuto_multi_anno_non_esemplifica_con_anni_finti():
    """Un esempio con anni concreti, su una domanda che ne cita altri, legge come un template.

    Gli anni veri non sono disponibili su questo ramo (`rotta.params` è vuoto): mostrare «dal
    2019 al 2021» a chi ha chiesto del 2015 e del 2023 non chiarisce la sintassi, segnala che la
    frase è precotta. La forma va detta come forma.
    """
    cfg = RagConfig()
    domanda = "Quante delibere del CIPE nel 2015 e nel 2023?"
    res = serve_uncovered(cfg, domanda, router.classify(domanda))

    assert res.refusal_reason == router.MOTIVO_ANNI_MULTIPLI
    assert re.search(r"\b(?:19|20)\d{2}\b", res.answer_text) is None


def test_il_degrado_a_rifiuto_resta_leggibile_nei_params():
    """Rotta servita e rotta proposta dal router divergono nel degrado: devono vedersi entrambe."""
    cfg = RagConfig()
    store = StructuredStore.from_manifest(MANIFEST)
    rotta = router.Route(router.STRUCTURED, intent="somma_importi", params={}, signals={})
    res = serve_structured(cfg, store, "quanto in totale", rotta)

    assert res.route == "uncovered"
    assert res.params["route"] == "uncovered"
    assert res.params["route_proposta"] == "structured"


def test_senza_degrado_rotta_servita_e_proposta_coincidono():
    cfg = RagConfig()
    store = StructuredStore.from_manifest(MANIFEST)
    domanda = "Quante delibere ha adottato il CIPESS nel 2024?"
    res = serve_structured(cfg, store, domanda, router.classify(domanda))

    assert res.params["route"] == "structured"
    assert res.params["route_proposta"] == "structured"
