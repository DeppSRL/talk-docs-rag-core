"""Il costo di indicizzare un corpus è misurato, non stimato.

Prima ``response.usage`` delle chiamate di embedding si buttava: il costo di indicizzazione
si poteva leggere **solo nella console del fornitore**. Con l'obiettivo dichiarato «quotare
in fretta con costi certi» — e col criterio «un secondo corpus in una giornata» — è la cifra
che serve a preventivare un corpus nuovo, quindi deve stare nel report della run.

Due proprietà che questi test difendono, e che è facile perdere:

- **un `usage` assente non fa fallire l'ingest.** Il campo è opzionale nella risposta: se
  manca si conta la chiamata e non i token. Perdere una metrica è meno grave che perdere
  un'indicizzazione da 13.670 chunk;
- **prezzo non configurato → costo `None`, non zero.** Uno zero si legge «gratis», che è una
  risposta sbagliata; `None` si legge «non lo sappiamo», che è quella giusta.

I test async girano con ``asyncio.run`` e non con ``pytest.mark.asyncio``: il repo non ha
``pytest-asyncio`` fra le dipendenze, e un test non è una ragione per aggiungerne una.
"""

import asyncio

from talkdocs_rag_core.config import RagConfig, _get_float
from talkdocs_rag_core.retrieval.services.embeddings import EmbeddingService


class _Usage:
    def __init__(self, prompt_tokens, total_tokens):
        self.prompt_tokens = prompt_tokens
        self.total_tokens = total_tokens


class _Dato:
    def __init__(self, embedding):
        self.embedding = embedding


class _Risposta:
    def __init__(self, n, dim, usage=None):
        self.data = [_Dato([0.1] * dim) for _ in range(n)]
        if usage is not None:
            self.usage = usage


class _Embeddings:
    def __init__(self, dim, usage_factory):
        self.dim = dim
        self._usage_factory = usage_factory
        self.chiamate = 0

    def create(self, model, input):  # noqa: A002 — è la firma del client OpenAI
        self.chiamate += 1
        return _Risposta(len(input), self.dim, self._usage_factory(len(input)))


class _Client:
    def __init__(self, dim=4, usage_factory=lambda n: _Usage(n * 10, n * 10)):
        self.embeddings = _Embeddings(dim, usage_factory)


class TestUsageAccumulato:
    def test_parte_da_zero(self):
        svc = EmbeddingService(_Client(), "mistral-embed-2312", 4)
        assert svc.usage == {"calls": 0, "prompt_tokens": 0, "total_tokens": 0}

    def test_accumula_su_piu_chiamate(self):
        svc = EmbeddingService(_Client(), "mistral-embed-2312", 4)
        asyncio.run(svc.get_embeddings(["a", "b"]))
        asyncio.run(svc.get_embeddings(["c"]))
        assert svc.usage["calls"] == 2
        # 2 testi × 10 + 1 testo × 10
        assert svc.usage["total_tokens"] == 30
        assert svc.usage["prompt_tokens"] == 30

    def test_usage_assente_non_fa_fallire(self):
        """Perdere la metrica è meno grave che perdere l'indicizzazione."""
        svc = EmbeddingService(_Client(usage_factory=lambda n: None), "m", 4)
        out = asyncio.run(svc.get_embeddings(["a", "b"]))
        assert len(out) == 2
        assert svc.usage["calls"] == 1
        assert svc.usage["total_tokens"] == 0

    def test_lista_vuota_non_chiama_e_non_conta(self):
        svc = EmbeddingService(_Client(), "m", 4)
        assert asyncio.run(svc.get_embeddings([])) == []
        assert svc.usage["calls"] == 0

    def test_i_testi_vuoti_non_arrivano_al_provider(self):
        """Erano già filtrati; qui si difende che il filtro non gonfi il conto."""
        svc = EmbeddingService(_Client(), "m", 4)
        asyncio.run(svc.get_embeddings(["a", "   ", ""]))
        assert svc.usage["calls"] == 1
        assert svc.usage["total_tokens"] == 10  # un solo testo non vuoto


class TestPrezzoEmbedding:
    def test_default_zero_quando_non_configurato(self, monkeypatch):
        monkeypatch.delenv("PRICE_EMBED_PER_MTOK", raising=False)
        assert RagConfig.from_env().price_embed_per_mtok == 0.0

    def test_valore_vuoto_vale_assente(self, monkeypatch):
        """`.env.example` lo lascia vuoto di proposito: non deve esplodere."""
        monkeypatch.setenv("PRICE_EMBED_PER_MTOK", "")
        assert _get_float("PRICE_EMBED_PER_MTOK", 0.0) == 0.0

    def test_letto_dall_ambiente(self, monkeypatch):
        monkeypatch.setenv("PRICE_EMBED_PER_MTOK", "0.10")
        assert RagConfig.from_env().price_embed_per_mtok == 0.10

    def test_e_separato_dai_prezzi_di_chat(self, monkeypatch):
        """Indicizzare è una-tantum per corpus, generare è ricorrente per domanda:
        sommarli nasconderebbe la cifra che serve a quotare un corpus nuovo."""
        monkeypatch.setenv("PRICE_EMBED_PER_MTOK", "0.10")
        monkeypatch.setenv("PRICE_INPUT_PER_MTOK", "0.20")
        cfg = RagConfig.from_env()
        assert cfg.price_embed_per_mtok == 0.10
        assert cfg.price_input_per_mtok == 0.20
