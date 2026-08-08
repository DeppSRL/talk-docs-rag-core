"""Il router agentico: il modello propone, la pipeline valida.

Il contratto sotto test è il confine di fiducia: qualunque cosa il modello risponda —
JSON rotto, route inventata, intento fuori insieme, parametri non legabili, eccezione di
rete — l'esito è la classificazione lessicale, mai un errore e mai una query costruita
su una proposta non conforme. E la traccia della proposta resta nel risultato anche
quando viene annullata: una proposta scartata è un dato.
"""

import json

import pytest

from config import RagConfig
from rag import router
from rag.agentic_router import AgenticRouter, _sanifica_params
from rag.corpus_card import CorpusCard

CARD = CorpusCard(sections=(("00-contesto", "Delibere CIPE/CIPESS."),))


class _Risposta:
    def __init__(self, contenuto: str):
        msg = type("Msg", (), {"content": contenuto})()
        self.choices = [type("Scelta", (), {"message": msg, "finish_reason": "stop"})()]
        self.usage = type(
            "Usage", (), {"prompt_tokens": 700, "completion_tokens": 20, "total_tokens": 720,
                          "prompt_tokens_details": None},
        )()


class _ClientFinto:
    """Restituisce il payload preparato, o solleva se `errore` è valorizzato."""

    def __init__(self, payload=None, errore: Exception | None = None):
        self._payload = payload
        self._errore = errore
        self.chiamate = 0
        self.opzioni: dict = {}
        self.chat = type("Chat", (), {"completions": self})()

    def with_options(self, **kwargs):
        """Come il client OpenAI: ritorna una vista con altre opzioni. Qui la stessa
        istanza, annotata — così il test può verificare il budget di retry richiesto."""
        self.opzioni.update(kwargs)
        return self

    def create(self, **kwargs):
        self.chiamate += 1
        self.kwargs = kwargs
        if self._errore is not None:
            raise self._errore
        contenuto = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return _Risposta(contenuto)


def _router(payload=None, errore=None):
    client = _ClientFinto(payload, errore)
    return AgenticRouter(RagConfig(), client, CARD, corpus_version="v-test"), client


def _lessicale(route=router.POINTWISE):
    return router.Route(route, signals={"forma_conteggio": False, "delibera_specifica": False})


def test_proposta_structured_valida_viene_servita():
    agentic, client = _router({"route": "structured", "intent": "count_delibere",
                               "params": {"anno": "2024", "comitato": "cipess"}})
    r = agentic.classify("Nel 2024 il Comitato quante ne ha approvate?", _lessicale())
    assert r.route == router.STRUCTURED and r.source == "llm"
    # I parametri arrivano sanificati: anno intero, comitato normalizzato.
    assert r.params == {"anno": 2024, "comitato": "CIPESS"}
    assert r.llm["error"] is None
    assert r.llm["usage"]["prompt_tokens"] == 700
    # I segnali lessicali restano: il disaccordo si conta in audit senza rigiocare.
    assert r.signals == _lessicale().signals


def test_proposta_meta_viene_servita():
    agentic, _ = _router({"route": "meta", "intent": None, "params": {}})
    r = agentic.classify("Ma qui dentro che roba c'è?", _lessicale())
    assert r.route == router.META and r.source == "llm"


@pytest.mark.parametrize(
    "payload",
    [
        "non è json {",                                                  # JSON rotto
        {"route": "sql_libero"},                                         # route inventata
        {"route": "structured", "intent": "somma_importi", "params": {}},  # intento fuori insieme
        {"route": "structured", "intent": "count_delibere",
         "params": {"anno": 2024, "anno_da": 2019, "anno_a": 2021}},     # anno E intervallo
        {"route": "structured", "intent": "count_delibere", "params": {"anno": "duemila"}},
        {"route": "structured", "intent": "count_delibere", "params": {"comitato": "CNEL"}},
        ["una", "lista"],                                                # non è un oggetto
    ],
)
def test_proposta_non_conforme_ricade_sul_lessicale(payload):
    agentic, _ = _router(payload)
    lessicale = _lessicale()
    r = agentic.classify("domanda", lessicale)
    assert r is lessicale and r.source == "lexical"
    assert r.llm["error"] is not None       # la proposta annullata resta tracciata


def test_errore_di_rete_non_fa_fallire_la_risposta():
    agentic, _ = _router(errore=TimeoutError("boom"))
    r = agentic.classify("domanda", _lessicale())
    assert r.route == router.POINTWISE and r.source == "lexical"
    assert "chiamata fallita" in r.llm["error"]


def test_il_router_ha_un_budget_di_retry_proprio():
    """Misurato su `eval-20260808T122852Z`: 4 chiamate su 55 morte in 429, ricadute sul
    lessicale, con il richiamo del router che finiva per misurare la rete. Il ramo
    agentico raddoppia le richieste al provider: il suo budget di retry non è quello
    della generazione."""
    cfg = RagConfig()
    _, client = _router({"route": "pointwise"})
    assert client.opzioni["max_retries"] == cfg.router_llm_max_retries
    assert cfg.router_llm_max_retries > cfg.http_max_retries


def test_la_chiamata_e_riproducibile_e_cache_friendly():
    agentic, client = _router({"route": "pointwise", "intent": None, "params": {}})
    agentic.classify("domanda", _lessicale())
    assert client.kwargs["temperature"] == 0.0
    assert client.kwargs["response_format"] == {"type": "json_object"}
    assert client.kwargs["extra_body"] == {"prompt_cache_key": "router:v-test"}
    # Il prefisso stabile porta scheda e manifest delle capacità.
    system = client.kwargs["messages"][0]["content"]
    assert "Delibere CIPE/CIPESS." in system and "count_delibere" in system


def test_sanifica_params_scarta_le_chiavi_ignote_senza_invalidare():
    assert _sanifica_params({"anno": 2024, "regione": "Lazio"}) == {"anno": 2024}


@pytest.mark.parametrize(
    "grezzi",
    [
        {"anno": 1492},                       # fuori plausibilità
        {"anno_da": 2021},                    # intervallo monco
        {"anno_da": 2024, "anno_a": 2019},    # intervallo rovesciato
        {"comitato": 42},
        "non è un dict",
    ],
)
def test_sanifica_params_rifiuta_il_non_conforme(grezzi):
    assert _sanifica_params(grezzi) is None
