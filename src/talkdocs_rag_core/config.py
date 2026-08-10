"""RagConfig — configurazione iniettata che sostituisce il singleton ``app.config.settings``
di talk-docs.

Razionale (spec §3): il singleton letto all'import è la sola colla trasversale del
nucleo vendored. Sostituirlo con una dataclass *frozen* passata ai costruttori non è un
ripiego: rende chunk size, pesi hybrid, RRF-k, modello, soglie e temperatura **parametri
che l'eval spazza per run**. Nessun modulo vendored deve leggere l'ambiente all'import.

Precedenza dei valori: parametri espliciti > variabili d'ambiente (``.env`` via
``python-dotenv``) > default qui sotto.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv è opzionale a runtime (env già iniettato nell'environment)
    pass


def _get(name: str, default: str | None = None) -> str | None:
    """Valore dall'ambiente, altrimenti il default.

    ``python-dotenv`` ripulisce il commento inline solo quando la variabile ha un valore:
    su ``SUPPORT_THRESHOLD=   # da tarare`` il commento *diventa* il valore. Nessuno di
    questi parametri può iniziare con ``#``, quindi un valore così è un residuo di
    commento e vale come assente — meglio il default che un ValueError a metà config, e
    meglio "chiave assente" che una chiave spazzatura spedita all'API.
    """
    val = os.environ.get(name)
    if val is not None:
        val = val.strip()
        if val.startswith("#"):
            val = ""
    return val if val not in (None, "") else default


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    return float(raw) if raw is not None else default


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    return int(raw) if raw is not None else default


def _get_opt_float(name: str) -> float | None:
    raw = _get(name)
    return float(raw) if raw is not None else None


_TRUE_VALUES = ("1", "true", "yes", "on", "si", "sì")
_FALSE_VALUES = ("0", "false", "no", "off")


def _get_bool(name: str, default: bool) -> bool:
    """Flag booleano, con lo stesso patto di ``_get_float``/``_get_int``: assente → default,
    non interpretabile → ``ValueError``.

    Un valore non riconosciuto **non** vale ``False``. Un flag scritto male
    (``ROUTER_ENABLED=tru``, ``VERBATIM_ENABLED=Y``) sarebbe indistinguibile da uno
    spegnimento voluto: spegnerebbe in silenzio il meccanismo che governa, ed è
    esattamente la classe di guasto già pagata con ``SUPPORT_THRESHOLD`` lasciata vuota —
    il rifiuto deterministico non scattava mai e nessuno se ne accorgeva.

    Le forme italiane sono accettate perché l'``.env`` si compila a mano.
    """
    raw = _get(name)
    if raw is None:
        return default
    val = raw.lower()
    if val in _TRUE_VALUES:
        return True
    if val in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name}: valore booleano non riconosciuto {raw!r}. "
        f"Attesi {'/'.join(_TRUE_VALUES)} oppure {'/'.join(_FALSE_VALUES)}."
    )


@dataclass(frozen=True)
class RagConfig:
    """Parametri di una run. Frozen: per variare un parametro si usa ``with_overrides``."""

    # --- Provider Mistral La Plateforme (diretto, sovranità EU) ---
    mistral_api_key: str = ""
    mistral_base_url: str = "https://api.mistral.ai/v1"
    # Stringhe PINNATE (M1). Mai alias -latest.
    mistral_model: str = "ministral-14b-2512"
    mistral_embed_model: str = "mistral-embed-2312"
    embed_dim: int = 1024

    # --- Inferenza (riproducibilità) ---
    llm_temperature: float = 0.0
    # Tarato sui dati, non scelto a occhio: nella run `eval-20260807T170947Z` le risposte
    # non troncate avevano mediana 207 token, 90° percentile 362, massimo 476 — cioè una
    # risposta lunga legittima sfiorava il vecchio tetto di 512, e 8 andavano a sbattere.
    # 1024 è ~2,1× il 90° percentile. Il tetto resta, e il troncamento resta un esito
    # dichiarato (`truncated`/`finish_reason` in audit): non è stato tolto, è stato alzato.
    max_output_tokens: int = 1024
    llm_seed: int | None = None

    # --- Trasporto HTTP verso La Plateforme ---
    # Il default dell'SDK OpenAI è 600 s di read timeout: su una connessione del pool
    # morta (es. il portatile è andato in suspend a metà run) una singola richiesta blocca
    # la run per 10 minuti *senza output*, e la latenza registrata diventa un artefatto.
    # 60 s è oltre il doppio della coda peggiore osservata su 512 token di output.
    http_timeout_s: float = 60.0
    http_connect_timeout_s: float = 10.0
    http_max_retries: int = 3

    # --- Vector store (Chroma) ---
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_retrieval: str = "corpus"
    chroma_collection_cache: str = "semantic_cache"

    # --- Chunking (C1) — variabili d'eval ---
    chunk_tokens: int = 400
    chunk_overlap_ratio: float = 0.12

    # --- Retrieval hybrid (RRF) — variabili d'eval ---
    hybrid_vector_weight: float = 0.7
    hybrid_keyword_weight: float = 0.3
    rrf_constant: int = 60
    rag_top_k: int = 5
    whoosh_index_dir: str = "data/whoosh_index"

    # --- Soglie di policy ---
    # Sotto SUPPORT_THRESHOLD (similarità densa del miglior chunk, 0..1) → rifiuto
    # deterministico (C3), ramo di codice non istruzione nel prompt. Da tarare sull'eval.
    support_threshold: float = 0.55
    # Cache semantica: prudente, meglio un miss (C4). Similarità coseno minima per hit.
    cache_sim_threshold: float = 0.92
    # Guardiano di astensione (C3b): IDF minimo del termine mancante più raro perché la
    # pipeline si astenga invece di rispondere. `support_threshold` non basta — misurato su
    # ic-07-bis: support 0,878 sopra soglia, e nessun passaggio conteneva «quote premiali».
    # 5,5 ≈ «un termine presente in meno di ~55 chunk su 13.670 è del tutto assente dai
    # passaggi». Tarato sui 33 item: precisione 1,00, richiamo 0,85. 0 = guardiano spento.
    abstention_idf_threshold: float = 5.5
    # Document frequency dei termini del corpus, rigenerata dall'ingest (gitignorata).
    term_df_path: str = "data/term_df.json"

    # --- Router aggregativo e guardia verbatim (incremento 1) ---
    # Il router riconosce le domande di conteggio/elenco e le instrada su una query
    # calcolata sui metadati. Spegnibile per rigiocare le run precedenti.
    router_enabled: bool = True
    # Governa la *richiesta* dello span letterale al modello (cambia lo schema di output,
    # e quindi il prefisso cache-friendly: spegnendolo si torna al contratto precedente).
    verbatim_enabled: bool = True
    # Quota minima di claim con verbatim verificato perché la risposta venga servita.
    # 0 = guardia SPENTA, si misura soltanto. Come `abstention_idf_threshold`, si tara su
    # una run di misura invece di sceglierla a occhio.
    verbatim_min_valid_ratio: float = 0.0
    # Sotto questa lunghezza uno span conta come NON valido anche se la substring esiste:
    # il test di appartenenza diventa banalmente soddisfacibile.
    verbatim_min_chars: int = 40
    # Righe mostrate nella risposta aggregativa. L'audit le porta comunque tutte.
    structured_max_rows: int = 20

    # --- Scheda del corpus e router agentico (incremento 1b) ---
    # Directory della scheda (contesto semantico scritto a mano, file NN-*.md). Assente
    # o vuota → niente risposta meta dalla scheda, la pipeline degrada senza esplodere.
    corpus_card_dir: str = "corpus/delibere/card"
    # Classificatore LLM a valle del router lessicale. **ACCESO** dal 2026-08-08, dopo
    # tre misure e non per intuizione: nasceva spento, la prima run lo bocciava (quattro
    # risposte perse), la correzione della scheda lo ha risanato, e la validazione su 16
    # domande **mai usate per correggere** ha confermato che il guadagno generalizza —
    # routing 13/16 contro 10/16, e aggregative colloquiali 4/4 contro 0/4, che è la
    # classe che le regex non possono prendere. Due difetti residui dichiarati in
    # STATUS.md: la classe «importo scritto in un documento» è instradata bene 3 volte su
    # 5, e la route meta attira 1 domanda sull'ente su 2.
    # La proposta del modello resta validata contro il semantic layer chiuso, e su
    # qualunque violazione si ricade sul lessicale: il numero non lo scrive mai il modello.
    router_llm_enabled: bool = True
    # Tetto di output della chiamata di classificazione: un JSON di route sta in poche
    # decine di token, il tetto è una guardia contro derive di prosa.
    router_llm_max_tokens: int = 256
    # Retry della sola chiamata di classificazione, più alti di `http_max_retries` (3).
    # Il ramo agentico raddoppia le richieste al provider ed è il primo punto a toccare il
    # rate limit: misurato, 4 chiamate su 55 morte in 429 e ricadute sul lessicale, con il
    # richiamo del router che finiva per misurare la rete. Un fallimento qui non è un
    # errore visibile — si traveste da decisione di instradamento.
    router_llm_max_retries: int = 8

    # --- Provenienza delle frasi ricorrenti ---
    # Certi fatti non stanno in un documento: stanno in una formula che centinaia di
    # delibere ricopiano (la legge istitutiva, la presidenza, una definizione). La
    # citazione a una di esse è corretta e arbitraria insieme, e il giudizio umano l'ha
    # segnalato tre volte. La nota lo dichiara con un numero calcolato.
    #
    # `frase_min_documenti` è la soglia con cui si COSTRUISCE l'indice in ingest: sotto,
    # la frase non è boilerplate del corpus. `provenienza_min_quota` è la soglia con cui
    # la nota SCATTA, espressa in quota del corpus perché è l'unica forma che si trasporta
    # su un corpus di dimensione diversa — su 511 delibere il 5% fa 26.
    provenienza_enabled: bool = True
    frase_min_documenti: int = 10
    provenienza_min_quota: float = 0.05
    frasi_index_path: str = "corpus/frasi_ricorrenti.json"

    # --- Resilienza dell'eval ---
    # Il retry dell'SDK (`http_max_retries`) copre il 429 istantaneo, non il rate limit
    # *sostenuto*: i suoi backoff stanno nell'ordine dei secondi, e una finestra di quota
    # esaurita dura di più. Misurato il 9 agosto: la run `eval-20260809T130308Z` è morta
    # all'item 104 su 110 — 104 risposte già pagate al provider, zero report, zero bundle,
    # perché una singola eccezione risaliva fino a `run_eval`. Il numero di chiamate era
    # cresciuto lo stesso giorno (il ramo meta ha smesso di essere a zero chiamate).
    #
    # Due meccanismi distinti, e servono entrambi: qui la pausa lunga fra i tentativi, e
    # in `_run_condition` il fatto che l'esaurimento dei tentativi produca una **riga di
    # errore** invece di uccidere la run. Un item morto è un dato mancante, non un esito:
    # nel CSV ha la colonna `errore` valorizzata e tutte le colonne di giudizio vuote.
    eval_item_max_retries: int = 4
    eval_item_backoff_s: float = 20.0

    # --- Pricing (M1: da confermare in console La Plateforme) — EUR/USD per 1M token ---
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0
    price_cached_per_mtok: float = 0.0

    # --- Logging / audit ---
    log_level: str = "info"
    audit_log_dir: str = "logs"

    def with_overrides(self, **kwargs) -> RagConfig:
        """Ritorna una copia con i campi indicati sovrascritti (per lo sweep d'eval)."""
        return replace(self, **kwargs)

    @property
    def chunk_overlap_tokens(self) -> int:
        return max(0, round(self.chunk_tokens * self.chunk_overlap_ratio))

    @classmethod
    def from_env(cls) -> RagConfig:
        """Costruisce la config leggendo l'ambiente (una sola volta, esplicitamente)."""
        return cls(
            mistral_api_key=_get("MISTRAL_API_KEY", "") or "",
            mistral_base_url=_get("MISTRAL_BASE_URL", cls.mistral_base_url),
            mistral_model=_get("MISTRAL_MODEL", cls.mistral_model),
            mistral_embed_model=_get("MISTRAL_EMBED_MODEL", cls.mistral_embed_model),
            embed_dim=_get_int("EMBED_DIM", cls.embed_dim),
            llm_temperature=_get_float("LLM_TEMPERATURE", cls.llm_temperature),
            max_output_tokens=_get_int("MAX_OUTPUT_TOKENS", cls.max_output_tokens),
            llm_seed=(int(_get("LLM_SEED")) if _get("LLM_SEED") else None),
            http_timeout_s=_get_float("HTTP_TIMEOUT_S", cls.http_timeout_s),
            http_connect_timeout_s=_get_float("HTTP_CONNECT_TIMEOUT_S", cls.http_connect_timeout_s),
            http_max_retries=_get_int("HTTP_MAX_RETRIES", cls.http_max_retries),
            chroma_persist_dir=_get("CHROMA_PERSIST_DIR", cls.chroma_persist_dir),
            chroma_collection_retrieval=_get("CHROMA_COLLECTION_RETRIEVAL", cls.chroma_collection_retrieval),
            chroma_collection_cache=_get("CHROMA_COLLECTION_CACHE", cls.chroma_collection_cache),
            chunk_tokens=_get_int("CHUNK_TOKENS", cls.chunk_tokens),
            chunk_overlap_ratio=_get_float("CHUNK_OVERLAP_RATIO", cls.chunk_overlap_ratio),
            hybrid_vector_weight=_get_float("HYBRID_VECTOR_WEIGHT", cls.hybrid_vector_weight),
            hybrid_keyword_weight=_get_float("HYBRID_KEYWORD_WEIGHT", cls.hybrid_keyword_weight),
            rrf_constant=_get_int("RRF_CONSTANT", cls.rrf_constant),
            rag_top_k=_get_int("RAG_TOP_K", cls.rag_top_k),
            whoosh_index_dir=_get("WHOOSH_INDEX_DIR", cls.whoosh_index_dir),
            support_threshold=_get_float("SUPPORT_THRESHOLD", cls.support_threshold),
            cache_sim_threshold=_get_float("CACHE_SIM_THRESHOLD", cls.cache_sim_threshold),
            abstention_idf_threshold=_get_float("ABSTENTION_IDF_THRESHOLD", cls.abstention_idf_threshold),
            term_df_path=_get("TERM_DF_PATH", cls.term_df_path),
            router_enabled=_get_bool("ROUTER_ENABLED", cls.router_enabled),
            verbatim_enabled=_get_bool("VERBATIM_ENABLED", cls.verbatim_enabled),
            verbatim_min_valid_ratio=_get_float("VERBATIM_MIN_VALID_RATIO", cls.verbatim_min_valid_ratio),
            verbatim_min_chars=_get_int("VERBATIM_MIN_CHARS", cls.verbatim_min_chars),
            structured_max_rows=_get_int("STRUCTURED_MAX_ROWS", cls.structured_max_rows),
            corpus_card_dir=_get("CORPUS_CARD_DIR", cls.corpus_card_dir),
            router_llm_enabled=_get_bool("ROUTER_LLM_ENABLED", cls.router_llm_enabled),
            router_llm_max_tokens=_get_int("ROUTER_LLM_MAX_TOKENS", cls.router_llm_max_tokens),
            router_llm_max_retries=_get_int("ROUTER_LLM_MAX_RETRIES", cls.router_llm_max_retries),
            provenienza_enabled=_get_bool("PROVENIENZA_ENABLED", cls.provenienza_enabled),
            frase_min_documenti=_get_int("FRASE_MIN_DOCUMENTI", cls.frase_min_documenti),
            provenienza_min_quota=_get_float("PROVENIENZA_MIN_QUOTA", cls.provenienza_min_quota),
            frasi_index_path=_get("FRASI_INDEX_PATH", cls.frasi_index_path),
            eval_item_max_retries=_get_int("EVAL_ITEM_MAX_RETRIES", cls.eval_item_max_retries),
            eval_item_backoff_s=_get_float("EVAL_ITEM_BACKOFF_S", cls.eval_item_backoff_s),
            price_input_per_mtok=_get_float("PRICE_INPUT_PER_MTOK", cls.price_input_per_mtok),
            price_output_per_mtok=_get_float("PRICE_OUTPUT_PER_MTOK", cls.price_output_per_mtok),
            price_cached_per_mtok=_get_float("PRICE_CACHED_PER_MTOK", cls.price_cached_per_mtok),
            log_level=_get("LOG_LEVEL", cls.log_level),
            audit_log_dir=_get("AUDIT_LOG_DIR", cls.audit_log_dir),
        )


# Radice del repo, utile a ingest/audit per path relativi stabili.
REPO_ROOT = Path(__file__).resolve().parent
