import hashlib
import os
from collections.abc import Iterator

import psycopg
import pytest
from psycopg.rows import TupleRow
from pydantic import ValidationError

from ragferee.config import get_settings
from ragferee.embeddings import Embedder
from ragferee.storage import ChunkStore


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


@pytest.fixture
def connection() -> Iterator[psycopg.Connection[TupleRow]]:
    try:
        conn = psycopg.connect(get_settings().get_postgres_url(), connect_timeout=2)
    except (ValidationError, psycopg.OperationalError) as error:
        if os.environ.get("RAGFEREE_REQUIRE_DB") == "1":
            pytest.fail(f"Postgres is required but not available: {error}")
        pytest.skip("Postgres not available")
    with conn:
        conn.execute("CREATE SCHEMA ragferee_test")
        conn.execute("SET search_path TO ragferee_test, public")
        yield conn
        conn.rollback()


@pytest.fixture
def store(connection: psycopg.Connection[TupleRow], fake_embedder: Embedder) -> ChunkStore:
    chunk_store = ChunkStore(connection, fake_embedder.dimension)
    chunk_store.create_schema()
    return chunk_store
