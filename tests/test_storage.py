from collections.abc import Iterator

import psycopg
import pytest
from psycopg.rows import TupleRow
from pydantic import ValidationError

from ragferee.config import get_settings
from ragferee.embeddings import Embedder
from ragferee.models import Chunk
from ragferee.storage import ChunkStore


@pytest.fixture
def connection() -> Iterator[psycopg.Connection[TupleRow]]:
    try:
        conn = psycopg.connect(get_settings().get_postgres_url(), connect_timeout=2)
    except (ValidationError, psycopg.OperationalError):
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


def make_chunk(section: str, text: str) -> Chunk:
    return Chunk(law_number=11, law_title="Offside", section=section, page=109, text=text)


CHUNKS = [
    make_chunk("1. Offside position", "It is not an offence to be in an offside position."),
    make_chunk("2. Offside offence", "A player is penalised for interfering with play."),
]


def test_replace_all_stores_every_chunk(store: ChunkStore, fake_embedder: Embedder) -> None:
    store.replace_all(CHUNKS, fake_embedder.embed_documents([c.text for c in CHUNKS]))

    assert store.count() == 2


def test_replace_all_replaces_previous_chunks(store: ChunkStore, fake_embedder: Embedder) -> None:
    store.replace_all(CHUNKS, fake_embedder.embed_documents([c.text for c in CHUNKS]))
    store.replace_all(CHUNKS[:1], fake_embedder.embed_documents([CHUNKS[0].text]))

    assert store.count() == 1


def test_mismatched_lengths_raise_and_keep_data(store: ChunkStore, fake_embedder: Embedder) -> None:
    store.replace_all(CHUNKS, fake_embedder.embed_documents([c.text for c in CHUNKS]))

    with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
        store.replace_all(CHUNKS, fake_embedder.embed_documents(["only one"]))

    assert store.count() == 2


def test_search_column_supports_keyword_queries(
    store: ChunkStore, connection: psycopg.Connection[TupleRow], fake_embedder: Embedder
) -> None:
    store.replace_all(CHUNKS, fake_embedder.embed_documents([c.text for c in CHUNKS]))

    rows = connection.execute(
        "SELECT section FROM chunks WHERE search @@ plainto_tsquery('english', 'interfering')"
    ).fetchall()

    assert rows == [("2. Offside offence",)]


def test_create_schema_rejects_table_with_other_dimension(
    connection: psycopg.Connection[TupleRow],
) -> None:
    ChunkStore(connection, 3).create_schema()

    with pytest.raises(ValueError, match=r"expected vector\(4\)"):
        ChunkStore(connection, 4).create_schema()


def test_replace_all_rejects_wrong_embedding_length(store: ChunkStore) -> None:
    with pytest.raises(ValueError, match="must have 3 values"):
        store.replace_all(CHUNKS[:1], [[0.1, 0.2]])


def test_replace_all_rejects_empty_input(store: ChunkStore) -> None:
    with pytest.raises(ValueError, match="No chunks"):
        store.replace_all([], [])


def test_count_is_zero_before_schema_exists(connection: psycopg.Connection[TupleRow]) -> None:
    connection.execute("SET search_path TO ragferee_test")  # hide any real public.chunks

    assert ChunkStore(connection, 3).count() == 0
