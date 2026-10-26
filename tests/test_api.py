import pytest
from fastapi.testclient import TestClient

from ragferee.answering import NOT_COVERED_MESSAGE
from ragferee.api import create_app, get_llm, get_searcher
from ragferee.llm import LLM, NOT_COVERED
from tests.fakes import RESULTS, FakeLLM, FakeSearcher


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
