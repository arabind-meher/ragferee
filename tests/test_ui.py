import json
from collections.abc import Callable

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from ragferee import ui
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


# The Streamlit app, run headless with the API calls replaced


def _app() -> None:
    from ragferee.ui import main

    main()


def run_app(
    monkeypatch: pytest.MonkeyPatch, *, chunks: int | None, reply: Answer | ApiError
) -> tuple[AppTest, list[tuple[str, int]]]:
    asked: list[tuple[str, int]] = []

    def fake_ask(client: httpx.Client, question: str, *, limit: int) -> Answer:
        asked.append((question, limit))
        if isinstance(reply, ApiError):
            raise reply
        return reply

    monkeypatch.setattr(ui, "chunk_count", lambda client: chunks)
    monkeypatch.setattr(ui, "ask", fake_ask)
    return AppTest.from_function(_app).run(), asked


def test_app_shows_api_status(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _ = run_app(monkeypatch, chunks=135, reply=Answer.model_validate(ANSWER))

    assert not app.exception
    assert app.title[0].value == "Ask about the Laws of the Game 2026/27"
    assert app.sidebar.success[0].value == "API online · 135 Law passages indexed"
    assert len(app.chat_input) == 1


def test_app_shows_answer_with_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    app, asked = run_app(monkeypatch, chunks=135, reply=Answer.model_validate(ANSWER))

    app.sidebar.slider[0].set_value(3)
    app.chat_input[0].set_value("Offside from a throw-in?").run()

    assert asked == [("Offside from a throw-in?", 3)]
    user, assistant = app.chat_message
    assert user.markdown[0].value == "Offside from a throw-in?"
    assert assistant.markdown[0].value == ANSWER["text"]
    assert "1. Law 11, 3. No offence, p. 111" in assistant.markdown[1].value


def test_app_keeps_history_until_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _ = run_app(monkeypatch, chunks=135, reply=Answer.model_validate(ANSWER))

    app.chat_input[0].set_value("Offside from a throw-in?").run()
    app.run()
    assert len(app.chat_message) == 2

    next(b for b in app.sidebar.button if b.label == "Clear chat").click().run()
    assert len(app.chat_message) == 0


def test_app_example_not_covered(monkeypatch: pytest.MonkeyPatch) -> None:
    not_covered = Answer(text="The Laws don't cover this.", citations=())
    app, asked = run_app(monkeypatch, chunks=135, reply=not_covered)

    app.sidebar.button[3].click().run()

    assert asked == [(ui.EXAMPLES[3], 5)]
    assert app.chat_message[1].info[0].value == "The Laws don't cover this."


def test_app_when_api_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _ = run_app(monkeypatch, chunks=None, reply=ApiError("Can't reach the ragferee API"))

    app.chat_input[0].set_value("Offside?").run()

    assert app.sidebar.warning[0].value == "API starting or offline"
    assert app.chat_message[1].error[0].value == "Can't reach the ragferee API"
