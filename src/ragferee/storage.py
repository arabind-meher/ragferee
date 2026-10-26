"""Store chunks and their embeddings in Postgres with pgvector."""

from collections.abc import Sequence
from typing import Any, LiteralString

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import TupleRow

from ragferee.models import Chunk

_INSERT = """
    INSERT INTO chunks (law_number, law_title, section, page, text, embedding)
    VALUES (%s, %s, %s, %s, %s, %s)
"""


class ChunkStore:
    """Keeps the indexed chunks and their embeddings in a Postgres table."""

    def __init__(self, connection: psycopg.Connection[TupleRow], dimension: int) -> None:
        self._connection = connection
        self._dimension = dimension

    def create_schema(self) -> None:
        """Create the vector extension, the chunks table and its index if missing."""
        self._connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._connection.execute(
            sql.SQL("""
            CREATE TABLE IF NOT EXISTS chunks (
                id          serial PRIMARY KEY,
                law_number  integer NOT NULL,
                law_title   text    NOT NULL,
                section     text    NOT NULL,
                page        integer NOT NULL,
                text        text    NOT NULL,
                embedding   vector({dimension}) NOT NULL,
                search      tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
            )
            """).format(dimension=sql.Literal(self._dimension))
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS chunks_search_idx ON chunks USING gin (search)"
        )
        self._check_dimension()

    def replace_all(self, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]) -> None:
        """Replace every stored chunk with the given chunks and their embeddings."""
        if not chunks:
            raise ValueError("No chunks to store")
        if len(chunks) != len(embeddings):
            raise ValueError(f"Got {len(chunks)} chunks but {len(embeddings)} embeddings")
        if any(len(embedding) != self._dimension for embedding in embeddings):
            raise ValueError(f"Every embedding must have {self._dimension} values")

        register_vector(self._connection)
        rows = [
            (c.law_number, c.law_title, c.section, c.page, c.text, Vector(embedding))
            for c, embedding in zip(chunks, embeddings, strict=True)
        ]
        with self._connection.transaction(), self._connection.cursor() as cursor:
            cursor.execute("TRUNCATE chunks RESTART IDENTITY")
            cursor.executemany(_INSERT, rows)

    def count(self) -> int:
        """Return the number of stored chunks, or 0 if the table does not exist yet."""
        if not self._scalar("SELECT to_regclass('chunks') IS NOT NULL"):
            return 0
        return int(self._scalar("SELECT count(*) FROM chunks"))

    def sections(self) -> set[tuple[int, str]]:
        """Return every (Law number, section) pair in the index."""
        rows = self._connection.execute(
            "SELECT DISTINCT law_number, section FROM chunks"
        ).fetchall()
        return {(row[0], row[1]) for row in rows}

    def _check_dimension(self) -> None:
        actual = self._scalar(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        )
        expected = f"vector({self._dimension})"
        if actual != expected:
            raise ValueError(
                f"chunks.embedding is {actual}, expected {expected}; drop the table to rebuild it"
            )

    def _scalar(self, query: LiteralString) -> Any:
        row = self._connection.execute(query).fetchone()
        if row is None:
            return None
        return row[0]
