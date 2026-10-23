import hashlib

import pytest

from ragferee.embeddings import Embedder


class FakeEmbedder:
    """Deterministic stand-in for a real embedder, with no model download."""

    dimension = 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for storage."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a question for searching."""
        digest = hashlib.sha256(text.encode()).digest()
        return [byte / 255 for byte in digest[: self.dimension]]


@pytest.fixture
def fake_embedder() -> Embedder:
    return FakeEmbedder()
