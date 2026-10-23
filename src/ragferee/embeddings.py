"""Turn text into embedding vectors."""

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384


class Embedder(Protocol):
    """Anything that can turn passages and questions into vectors."""

    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for storage."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a question for searching."""
        ...


class FastEmbedder:
    """Embeds text locally with the BGE small English model via fastembed."""

    dimension = EMBEDDING_DIMENSION

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._model: TextEmbedding | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for storage."""
        vectors = self._get_model().passage_embed(texts)
        return [_to_list(vector) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a question for searching."""
        vector = next(iter(self._get_model().query_embed(text)))
        return _to_list(vector)

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(MODEL_NAME, cache_dir=str(self._cache_dir))
        return self._model


def _to_list(vector: Iterable[float]) -> list[float]:
    return [float(value) for value in vector]
