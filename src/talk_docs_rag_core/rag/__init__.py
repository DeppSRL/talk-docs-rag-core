"""C3 — generazione grounded: output strutturato + rifiuto deterministico."""

from .generation import MistralGenerator, RagResult
from .schema import StructuredAnswer

__all__ = ["MistralGenerator", "RagResult", "StructuredAnswer"]
