"""Find the chunks most relevant to a question."""

from collections.abc import Sequence
from enum import StrEnum

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import TupleRow
from pydantic import BaseModel, ConfigDict

from ragferee.embeddings import Embedder
from ragferee.models import Chunk

RRF_K = 60  # standard Reciprocal Rank Fusion constant (Cormack et al., 2009)
# Keyword search uses OR matching, so it matches most chunks and ranks them noisily; it
# counts for a quarter of vector search. Chosen on eval/retrieval.jsonl for the best hit@5
# (every top-5 chunk reaches the LLM), then confirmed on eval/retrieval-holdout.jsonl.
KEYWORD_WEIGHT = 0.25
_CANDIDATE_FACTOR = 10  # candidates taken from each search per requested result

_VECTOR_SQL = "SELECT id FROM chunks ORDER BY embedding <=> %s, id LIMIT %s"
_KEYWORD_SQL = """
    WITH q AS (
        SELECT replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')::tsquery AS query
    )
    SELECT id FROM chunks, q
    WHERE search @@ q.query
    ORDER BY ts_rank(search, q.query) DESC, id  -- id breaks ties, so results are repeatable
    LIMIT %s
"""
_FETCH_SQL = """
    SELECT id, law_number, law_title, section, page, text FROM chunks WHERE id = ANY(%s)
"""


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """Merge ranked id lists into one ranking, the best first.

    Each id scores weight / (k + rank) for every list it appears in; without weights,
    every list counts equally. Ties keep the order in which ids were first seen, so the
    result is deterministic.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("Need one weight per ranking")
    if any(weight < 0 for weight in weights):
        raise ValueError("Weights must not be negative")

    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda entry: entry[1], reverse=True)


class SearchMode(StrEnum):
    """Which searches to run: both combined, or one on its own for comparison."""

    HYBRID = "hybrid"
    VECTOR = "vector"
    KEYWORD = "keyword"


class SearchResult(BaseModel):
    """A chunk found for a question, with its fused relevance score.

    The score only orders results (higher is better); it is not a probability.
    """

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float


class Retriever:
    """Hybrid search over the stored chunks: vector similarity plus keywords."""

    def __init__(
        self,
        connection: psycopg.Connection[TupleRow],
        embedder: Embedder,
        *,
        keyword_weight: float = KEYWORD_WEIGHT,
    ) -> None:
        if keyword_weight < 0:
            raise ValueError("keyword_weight must not be negative")
        self._connection = connection
        self._embedder = embedder
        self._keyword_weight = keyword_weight

    def search(
        self, question: str, *, limit: int = 5, mode: SearchMode = SearchMode.HYBRID
    ) -> list[SearchResult]:
        """Return the chunks most relevant to the question, best first."""
        if not question.strip():
            raise ValueError("Question must not be empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        register_vector(self._connection)
        candidates = limit * _CANDIDATE_FACTOR
        rankings: list[list[int]] = []
        weights: list[float] = []
        if mode in (SearchMode.HYBRID, SearchMode.VECTOR):
            rankings.append(self._vector_ids(question, candidates))
            weights.append(1.0)
        if mode in (SearchMode.HYBRID, SearchMode.KEYWORD):
            rankings.append(self._keyword_ids(question, candidates))
            weights.append(self._keyword_weight)
        fused = reciprocal_rank_fusion(rankings, weights=weights)[:limit]
        chunks = self._fetch([item for item, _ in fused])
        return [SearchResult(chunk=chunks[item], score=score) for item, score in fused]

    def _vector_ids(self, question: str, count: int) -> list[int]:
        query = Vector(self._embedder.embed_query(question))
        rows = self._connection.execute(_VECTOR_SQL, (query, count)).fetchall()
        return [row[0] for row in rows]

    def _keyword_ids(self, question: str, count: int) -> list[int]:
        rows = self._connection.execute(_KEYWORD_SQL, (question, count)).fetchall()
        return [row[0] for row in rows]

    def _fetch(self, ids: list[int]) -> dict[int, Chunk]:
        rows = self._connection.execute(_FETCH_SQL, (ids,)).fetchall()
        return {
            row[0]: Chunk(
                law_number=row[1], law_title=row[2], section=row[3], page=row[4], text=row[5]
            )
            for row in rows
        }
