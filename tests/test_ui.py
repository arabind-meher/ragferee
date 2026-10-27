import json
from collections.abc import Callable

import httpx
import pytest

from ragferee.models import Answer, Citation
from ragferee.ui import ApiError, ask, chunk_count, format_sources

CITATIONS = [
    {"law_number": 11, "section": "3. No offence", "page": 111},
    {"law_number": 11, "section": "1. Offside position", "page": 109},
]
ANSWER = {
    "text": "No offence from a throw-in [1]. Otherwise offside applies [1] [2].",
    "citations": CITATIONS,
}


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(base_url="http://api.test", transport=httpx.MockTransport(handler))


def respond(status: int, body: object) -> httpx.Client:
    return make_client(lambda request: httpx.Response(status, json=body))


def test_ask_returns_answer() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=ANSWER)

    answer = ask(make_client(handler), "Offside from a throw-in?", limit=3)

    assert answer.text == ANSWER["text"]
    assert answer.citations[1] == Citation(law_number=11, section="1. Offside position", page=109)
    assert requests[0].url.path == "/ask"
    assert json.loads(requests[0].content) == {"question": "Offside from a throw-in?", "limit": 3}


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (422, {"detail": []}, "Please enter a question"),
        (
            502,
            {"detail": "OpenAI request failed: rate limit"},
            "failed: OpenAI request failed: rate",
        ),
        (503, {"detail": "Database unavailable"}, "Is Postgres running"),
        (500, {"detail": "Internal Server Error"}, r"error \(500\)"),
    ],
)
def test_ask_error_status_raises(status: int, body: object, message: str) -> None:
    with pytest.raises(ApiError, match=message):
        ask(respond(status, body), "Offside?", limit=5)


def test_ask_unreachable_api_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ApiError, match="Can't reach the ragferee API"):
        ask(make_client(handler), "Offside?", limit=5)


def test_ask_timeout_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(ApiError, match="took too long"):
        ask(make_client(handler), "Offside?", limit=5)


def test_chunk_count_when_healthy() -> None:
    assert chunk_count(respond(200, {"status": "ok", "chunks": 135})) == 135


def test_chunk_count_when_unhealthy() -> None:
    assert chunk_count(respond(503, {"detail": "No chunks indexed"})) is None


def test_chunk_count_when_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    assert chunk_count(make_client(handler)) is None


def test_format_sources_numbers_citations_in_order() -> None:
    answer = Answer.model_validate(ANSWER)

    assert format_sources(answer) == (
        "1. Law 11, 3. No offence, p. 111\n2. Law 11, 1. Offside position, p. 109"
    )
