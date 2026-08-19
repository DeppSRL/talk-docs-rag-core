"""Il costo di un'astensione va dove il costo è stato sostenuto.

Delle due astensioni, una esce **prima** della chiamata al modello (`termini_mancanti`, IDF)
e l'altra **dopo** (`verbatim`: i claim sono già stati generati). Se entrambe registrassero
`usage={}`, con la guardia accesa il costo delle risposte scartate sparirebbe dalle
metriche — e sparirebbe proprio sulle domande peggiori, quelle su cui si decide se la
guardia conviene. È un errore di segno opposto a quello di contare i rifiuti deterministici
come chiamate al modello, già misurato su `eval-20260805T061415Z`.
"""

import json

from talk_docs_rag_core.config import RagConfig
from talk_docs_rag_core.rag.generation import MistralGenerator
from talk_docs_rag_core.rag.guard import TermStats
from talk_docs_rag_core.retrieval.services.hybrid_search import HybridSearchResult

TESTO = (
    "Il Comitato assegna in via programmatica la somma di euro 295.178.000 a valere "
    "sulle risorse del Fondo sviluppo e coesione per l'anno 2020."
)


class _Usage:
    prompt_tokens = 2000
    completion_tokens = 120
    total_tokens = 2120
    prompt_tokens_details = type("D", (), {"cached_tokens": 512})()


class RispostaFinta:
    def __init__(self, payload: dict):
        self.choices = [type("C", (), {
            "message": type("M", (), {"content": json.dumps(payload, ensure_ascii=False)})(),
            "finish_reason": "stop",
        })()]
        self.usage = _Usage()


class ClientFinto:
    def __init__(self, payload):
        self.payload = payload
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        return RispostaFinta(self.payload)


def _risultati():
    return [HybridSearchResult(content=TESTO, score=0.9, vector_score=0.9, keyword_score=0.5,
                               source="delibere/2020/E200001.txt", chunk_id="c1", metadata={}, rank=1)]


def _payload(verbatim: str):
    return {"answer": "Assegna 295 milioni [1].",
            "claims": [{"statement": "Assegna 295 milioni", "passages": [1], "verbatim": verbatim}]}


def _generatore(**kw):
    cfg = RagConfig(support_threshold=0.0, abstention_idf_threshold=0.0, **kw)
    return MistralGenerator(cfg, ClientFinto(_payload("la somma di euro 32.500.000 per le quote premiali")))


def test_l_astensione_verbatim_porta_il_costo_della_chiamata_fatta():
    res = _generatore(verbatim_min_valid_ratio=1.0).generate("quanto assegna?", _risultati(), cache_key="k")

    assert res.uncertain is True and res.uncertain_reason == "verbatim"
    assert res.usage["prompt_tokens"] == 2000
    assert res.usage["completion_tokens"] == 120
    assert res.usage["cached_tokens"] == 512
    # Senza `raw_output` la tupla di audit non permette di rigiocare la risposta scartata:
    # è l'unico posto in cui restano i claim che non hanno superato la guardia.
    assert '"claims"' in res.raw_output


def test_l_astensione_per_termini_mancanti_resta_a_costo_zero():
    """Deterministica, esce prima della chiamata: `usage={}` è la verità, non una svista."""
    cfg = RagConfig(support_threshold=0.0, abstention_idf_threshold=0.01)
    # Senza `TermStats` il segnale IDF è 0 per costruzione e il ramo non scatterebbe mai.
    stats = TermStats(n_chunks=1000, df={})
    gen = MistralGenerator(cfg, ClientFinto(_payload("qualunque cosa")), term_stats=stats)
    res = gen.generate("quante pratiche catastali sono state protocollate?", _risultati(), cache_key="k")

    assert res.uncertain is True and res.uncertain_reason == "termini_mancanti"
    assert res.usage == {}
    assert res.raw_output == ""


def test_la_risposta_servita_porta_lo_stesso_usage():
    """L'estrazione dell'usage è una sola: i due rami non possono divergere."""
    res = _generatore(verbatim_min_valid_ratio=0.0).generate("quanto assegna?", _risultati(), cache_key="k")

    assert res.uncertain is False
    assert res.usage["total_tokens"] == 2120
    assert res.usage["cached_tokens"] == 512
