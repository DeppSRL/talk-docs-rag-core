"""Embedding service (vendorizzato @6dd976c).

Modifica del vendoring (spec §3): niente ``settings`` letto all'import. Il client
OpenAI-compatible, la stringa modello e la dimensione sono **iniettati** dal costruttore
(vengono da ``RagConfig``), così l'eval può spazzare modello/dimensione per run.
"""

from openai import OpenAI


class EmbeddingService:
    """Service for generating embeddings via a OpenAI-compatible client (Mistral)."""

    def __init__(self, client: OpenAI, model: str, dimension: int):
        """Initialize with an injected client + model + dimension.

        Args:
            client: client OpenAI-compatible (create_mistral_client).
            model: stringa modello embedding PINNATA (es. mistral-embed-2312).
            dimension: dimensione dei vettori (mistral-embed → 1024).
        """
        if client is None:
            raise ValueError("EmbeddingService richiede un client valido (MISTRAL_API_KEY assente?)")
        self.client = client
        self.model = model
        self.dimension = dimension

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts."""
        if not texts:
            return []

        # Remove empty texts and track indices
        non_empty_texts = []
        original_indices = []

        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_texts.append(text.strip())
                original_indices.append(i)

        if not non_empty_texts:
            return [[0.0] * self.dimension] * len(texts)

        try:
            response = self.client.embeddings.create(model=self.model, input=non_empty_texts)

            # Create result list with proper ordering
            embeddings = [[0.0] * self.dimension] * len(texts)

            for i, embedding_data in enumerate(response.data):
                original_index = original_indices[i]
                embeddings[original_index] = embedding_data.embedding

            return embeddings

        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}") from e

    async def get_single_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings else [0.0] * self.dimension
