"""
LLM response generation utilities for RAG system (vendorizzato @6dd976c).

Modifiche del vendoring (spec §3):
- Niente ``from app.config import settings``: ``default_model`` è iniettato dal
  costruttore (viene da ``RagConfig``).
- Niente ``from app.i18n import t``: il solo testo fisso usato ("no_context") è
  inline-ato nella costante ``NO_CONTEXT_IT`` (mercato PA italiano = default).

Path di generazione con citazioni ``[n]`` + estrazione marker: riusato dal PoC come
riferimento. Il ramo di *rifiuto deterministico* e l'output strutturato con
``source_id`` vivono invece in ``rag/`` (C3), non nel testo del prompt.
"""

import logging
import re
from dataclasses import dataclass, field

from openai import OpenAI

logger = logging.getLogger(__name__)

# Fallback fisso "nessun contesto" — inline dell'unico messaggio i18n usato qui.
NO_CONTEXT_IT = "Non ho trovato informazioni rilevanti per rispondere alla tua domanda."


@dataclass
class ContextChunk:
    """Represents a chunk of retrieved context."""

    content: str
    source: str
    similarity: float
    distance: float
    chunk_id: str
    found_by_query: str | None = None


@dataclass
class LLMResponse:
    """Represents the response from LLM generation."""

    response_text: str
    sources_used: list[str]
    chunks_used: int
    total_chunks_available: int
    model_used: str
    success: bool = True
    error_message: str | None = None


@dataclass
class CitedLLMResponse:
    """Risposta sintetica con citazioni inline `[n]` mappate ai chunk usati."""

    response_text: str
    used_markers: list[int] = field(default_factory=list)
    chunks_used: int = 0
    model_used: str = ""
    success: bool = True
    error_message: str | None = None


class LLMGenerationService:
    """Service for generating LLM responses from retrieved context."""

    SYSTEM_PROMPT = (
        "You are an expert assistant that answers questions based EXCLUSIVELY on the "
        "provided context.\n"
        "The context comes from the client's indexed documents.\n"
        "\n"
        "Rules:\n"
        "1. Reply in the SAME LANGUAGE as the user's question.\n"
        "2. Base your answer EXCLUSIVELY on the provided context.\n"
        "3. If the context does not contain enough information, say so clearly.\n"
        "4. Cite the sources when possible.\n"
        "5. Be precise and concise.\n"
        "6. Keep a professional but accessible tone.\n"
        "7. Take the conversation history into account for more contextual answers."
    )

    CITATION_SYSTEM_PROMPT = (
        "You answer questions based EXCLUSIVELY on the provided passages, each numbered "
        "as [n].\n"
        "The passages come from the client's indexed documents.\n"
        "\n"
        "Rules:\n"
        "1. Reply in the SAME LANGUAGE as the user's question, precisely and concisely.\n"
        "2. Use only the information contained in the numbered passages: do not add "
        "anything that is not in the context.\n"
        "3. ALWAYS cite the source by placing the number in square brackets right after "
        "the statement it supports, e.g. [1] or [2][3]. Cite only the passages you "
        "actually used.\n"
        "4. If the passages do not contain enough information, say so clearly and do not "
        "invent anything.\n"
        "5. Do not list the sources at the end: inline citations are enough.\n"
        "6. Take the conversation history into account for more contextual answers."
    )

    _MARKER_RE = re.compile(r"\[(\d+)\]")

    def __init__(self, openai_client: OpenAI, default_model: str | None = None):
        """Initialize the service with an OpenAI client and injected default model."""
        self.client = openai_client
        self.default_model = default_model

    def _format_numbered_context(self, chunks: list[ContextChunk]) -> str:
        """Format chunks as a numbered context block (`[1]`, `[2]`, …)."""
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(f"[{i}] (documento: {chunk.source})\n{chunk.content}")
        return "\n\n---\n\n".join(parts)

    def _build_citation_messages(
        self, query: str, numbered_context: str, conversation_history: list[dict] | None = None
    ) -> list[dict]:
        """Build the messages array for a citation-aware generation call."""
        messages: list[dict] = [{"role": "system", "content": self.CITATION_SYSTEM_PROMPT}]

        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        user_prompt = (
            f"Passaggi disponibili:\n\n{numbered_context}\n\n---\n\n"
            f"Domanda: {query}\n\n"
            "Rispondi basandoti esclusivamente sui passaggi numerati sopra, citando le "
            "fonti con [n]."
        )
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _extract_used_markers(self, text: str, max_marker: int) -> list[int]:
        """Extract the citation markers actually used in the answer (deduped, ordered)."""
        seen: dict[int, None] = {}
        for match in self._MARKER_RE.findall(text):
            n = int(match)
            if 1 <= n <= max_marker and n not in seen:
                seen[n] = None
        return list(seen.keys())

    async def generate_answer_with_citations(
        self,
        query: str,
        context_chunks: list[ContextChunk],
        model: str | None = None,
        max_tokens: int = 1000,
        conversation_history: list[dict] | None = None,
    ) -> CitedLLMResponse:
        """Generate a synthetic answer with inline `[n]` citations."""
        model = model or self.default_model

        if not context_chunks:
            return CitedLLMResponse(
                response_text=NO_CONTEXT_IT,
                used_markers=[],
                chunks_used=0,
                model_used=model or "",
                success=True,
            )

        numbered_context = self._format_numbered_context(context_chunks)
        messages = self._build_citation_messages(query, numbered_context, conversation_history)

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            response_text = response.choices[0].message.content or ""
            used_markers = self._extract_used_markers(response_text, len(context_chunks))

            logger.info(
                "Generated cited answer: %d chunks in context, %d cited",
                len(context_chunks),
                len(used_markers),
            )

            return CitedLLMResponse(
                response_text=response_text,
                used_markers=used_markers,
                chunks_used=len(context_chunks),
                model_used=model,
                success=True,
            )
        except Exception as e:
            logger.error(f"Cited LLM generation failed: {e}")
            return CitedLLMResponse(
                response_text="",
                used_markers=[],
                chunks_used=len(context_chunks),
                model_used=model or "",
                success=False,
                error_message=str(e),
            )


def convert_dict_to_context_chunk(chunk_dict: dict) -> ContextChunk:
    """Convert a dictionary representation to a ContextChunk."""
    return ContextChunk(
        content=chunk_dict["content"],
        source=chunk_dict["source"],
        similarity=chunk_dict["similarity"],
        distance=chunk_dict["distance"],
        chunk_id=chunk_dict["chunk_id"],
        found_by_query=chunk_dict.get("found_by_query"),
    )
