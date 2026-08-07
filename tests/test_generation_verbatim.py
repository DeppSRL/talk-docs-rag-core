"""Quarto ramo della cascata: la risposta non passa la verifica → astensione.

La guardia nasce SPENTA (`verbatim_min_valid_ratio = 0`): questi test coprono entrambe le
configurazioni, perché la run di misura gira a soglia 0 e la taratura arriva dopo.
"""

import json

from config import RagConfig
from rag.generation import MistralGenerator
from vendor.talkdocs.services.hybrid_search import HybridSearchResult

TESTO = (
    "Il Comitato assegna in via programmatica la somma di euro 295.178.000 a valere "
    "sulle risorse del Fondo sviluppo e coesione per l'anno 2020."
)


class RispostaFinta:
    def __init__(self, payload: dict):
        self.choices = [type("C", (), {
            "message": type("M", (), {"content": json.dumps(payload, ensure_ascii=False)})(),
            "finish_reason": "stop",
        })()]
        self.usage = None


class ClientFinto:
    def __init__(self, payload):
        self.payload = payload
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        return RispostaFinta(self.payload)


def _risultati():
    # `rank` è obbligatorio nella dataclass vendored: ometterlo dà TypeError, non un default.
    return [HybridSearchResult(content=TESTO, score=0.9, vector_score=0.9, keyword_score=0.5,
                               source="delibere/2020/E200001.txt", chunk_id="c1", metadata={}, rank=1)]


def _payload(verbatim: str):
    return {"answer": "Assegna 295 milioni [1].",
            "claims": [{"statement": "Assegna 295 milioni", "passages": [1], "verbatim": verbatim}]}


def test_schema_richiede_il_verbatim():
    from rag.schema import STRUCTURED_RESPONSE_SCHEMA

    claim = STRUCTURED_RESPONSE_SCHEMA["json_schema"]["schema"]["properties"]["claims"]["items"]
    assert "verbatim" in claim["properties"]
    assert "verbatim" in claim["required"]


def test_a_guardia_spenta_misura_ma_serve_la_risposta():
    cfg = RagConfig(support_threshold=0.0, abstention_idf_threshold=0.0, verbatim_min_valid_ratio=0.0)
    gen = MistralGenerator(cfg, ClientFinto(_payload("la somma di euro 32.500.000 per le quote premiali")))
    res = gen.generate("quanto assegna?", _risultati(), cache_key="k")
    assert res.uncertain is False               # servita
    assert res.verbatim.n_not_found == 1        # ma il difetto è registrato
    assert res.verbatim.valid_ratio == 0.0


def test_a_guardia_accesa_degrada_ad_astensione():
    cfg = RagConfig(support_threshold=0.0, abstention_idf_threshold=0.0, verbatim_min_valid_ratio=1.0)
    gen = MistralGenerator(cfg, ClientFinto(_payload("la somma di euro 32.500.000 per le quote premiali")))
    res = gen.generate("quanto assegna?", _risultati(), cache_key="k")
    assert res.uncertain is True
    assert res.uncertain_reason == "verbatim"
    assert "non" in res.answer_text.lower()


def test_span_valido_passa_anche_a_guardia_accesa():
    cfg = RagConfig(support_threshold=0.0, abstention_idf_threshold=0.0, verbatim_min_valid_ratio=1.0)
    gen = MistralGenerator(cfg, ClientFinto(_payload("assegna in via programmatica la somma di euro 295.178.000")))
    res = gen.generate("quanto assegna?", _risultati(), cache_key="k")
    assert res.uncertain is False
    assert res.verbatim.valid_ratio == 1.0


def test_senza_claim_la_guardia_non_scatta():
    """Il salvage da JSON troncato produce `claims=[]`: il rapporto non è definito."""
    cfg = RagConfig(support_threshold=0.0, abstention_idf_threshold=0.0, verbatim_min_valid_ratio=1.0)
    gen = MistralGenerator(cfg, ClientFinto({"answer": "prosa senza claim", "claims": []}))
    res = gen.generate("domanda", _risultati(), cache_key="k")
    assert res.uncertain is False
    assert res.verbatim.valid_ratio is None


def test_verbatim_disattivato_non_produce_esito():
    cfg = RagConfig(support_threshold=0.0, abstention_idf_threshold=0.0, verbatim_enabled=False)
    gen = MistralGenerator(cfg, ClientFinto(_payload("qualunque cosa")))
    res = gen.generate("domanda", _risultati(), cache_key="k")
    assert res.verbatim is None
