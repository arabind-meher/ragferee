import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg.rows import TupleRow

from ragferee.answering import NOT_COVERED_MESSAGE
from ragferee.api import create_app, get_connection, get_llm, get_searcher
from ragferee.embeddings import Embedder
from ragferee.llm import LLM, NOT_COVERED, LLMError
from ragferee.storage import ChunkStore
from tests.fakes import RESULTS, FakeLLM, FakeSearcher


class FailingLLM:
    def complete(self, system: str, prompt: str) -> str:
        raise LLMError("OpenAI request failed: rate limit")


def _client(llm: LLM) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_searcher] = lambda: FakeSearcher(RESULTS)
    app.dependency_overrides[get_llm] = lambda: llm
    return TestClient(app)


def test_ask_returns_answer_with_citations() -> None:
    client = _client(FakeLLM("The other team gets a corner kick [1]."))

    response = client.post("/ask", json={"question": "Goalkeeper holds the ball too long?"})

    assert response.status_code == 200
    assert response.json() == {
        "text": "The other team gets a corner kick [1].",
        "citations": [{"law_number": 12, "section": "3. Corner kick", "page": 119}],
    }


def test_ask_limit_caps_sources() -> None:
    llm = FakeLLM("A corner kick [1].")

    _client(llm).post("/ask", json={"question": "Corner kick?", "limit": 2})

    assert "[2]" in llm.prompts[0]
    assert "[3]" not in llm.prompts[0]


def test_ask_not_covered() -> None:
    response = _client(FakeLLM(NOT_COVERED)).post("/ask", json={"question": "Who won?"})

    assert response.status_code == 200
    assert response.json() == {"text": NOT_COVERED_MESSAGE, "citations": []}


def test_ask_rejects_blank_question() -> None:
    response = _client(FakeLLM("unused")).post("/ask", json={"question": "   "})

    assert response.status_code == 422


@pytest.mark.parametrize("limit", [0, 21])
def test_ask_rejects_limit_out_of_range(limit: int) -> None:
    response = _client(FakeLLM("unused")).post(
        "/ask", json={"question": "Offside?", "limit": limit}
    )

    assert response.status_code == 422


def test_ask_llm_failure_returns_502() -> None:
    response = _client(FailingLLM()).post("/ask", json={"question": "Offside?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "OpenAI request failed: rate limit"}


def test_ask_database_failure_returns_503() -> None:
    def unavailable() -> None:
        raise psycopg.OperationalError("connection refused at 10.0.0.1")

    app = create_app()
    app.dependency_overrides[get_searcher] = unavailable

    response = TestClient(app).post("/ask", json={"question": "Offside?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}


def _health_app(connection: psycopg.Connection[TupleRow]) -> FastAPI:
    app = create_app()
    app.dependency_overrides[get_connection] = lambda: connection
    return app


def test_health_reports_chunk_count(
    connection: psycopg.Connection[TupleRow], store: ChunkStore, fake_embedder: Embedder
) -> None:
    chunks = [result.chunk for result in RESULTS]
    store.replace_all(chunks, fake_embedder.embed_documents([chunk.text for chunk in chunks]))

    response = TestClient(_health_app(connection)).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunks": 3}


def test_health_without_chunks_returns_503(
    connection: psycopg.Connection[TupleRow], store: ChunkStore
) -> None:
    response = TestClient(_health_app(connection)).get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "No chunks indexed"}
