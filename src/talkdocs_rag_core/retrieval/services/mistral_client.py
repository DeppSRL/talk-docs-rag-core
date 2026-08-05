"""Client factory Mistral La Plateforme (OpenAI-compatible).

Vendorizzato da ``services/openrouter_client.py`` @6dd976c e **generalizzato al
base_url Mistral** (spec §3). La modifica del vendoring: niente singleton ``settings``
letto all'import — il ``base_url`` e la ``api_key`` arrivano da ``RagConfig`` iniettato.

Un solo client OpenAI-compatible serve sia LLM (chat) sia embeddings.
"""

import logging

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)


def create_mistral_client(
    api_key: str,
    base_url: str = "https://api.mistral.ai/v1",
    timeout_s: float = 60.0,
    connect_timeout_s: float = 10.0,
    max_retries: int = 3,
) -> OpenAI | None:
    """Crea il client OpenAI-compatible puntato a Mistral La Plateforme.

    Args:
        api_key: MISTRAL_API_KEY (da RagConfig, mai hardcoded).
        base_url: endpoint La Plateforme (diretto, sovranità EU — non OpenRouter).
        timeout_s: read/write timeout per richiesta. **Non lasciare il default dell'SDK
            (600 s)**: una connessione del pool morta blocca la run per 10 minuti senza
            output e inquina la latenza misurata.
        connect_timeout_s: timeout di sola connessione (deve essere basso: se l'endpoint
            non è raggiungibile lo si sa in pochi secondi).
        max_retries: tentativi dell'SDK con backoff esponenziale. Il retry apre una
            connessione nuova, ed è ciò che salva la run dopo un resume da suspend.

    Returns:
        Client ``OpenAI`` configurato, o ``None`` se manca la chiave.
    """
    if not api_key:
        logger.warning("MISTRAL_API_KEY non configurata — funzioni LLM/embedding disabilitate")
        return None

    timeout = httpx.Timeout(timeout_s, connect=connect_timeout_s)
    logger.info(
        "Creazione client Mistral La Plateforme: %s (timeout=%.0fs connect=%.0fs retries=%d)",
        base_url,
        timeout_s,
        connect_timeout_s,
        max_retries,
    )
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
