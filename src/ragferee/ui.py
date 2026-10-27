"""Web chat for asking questions about the Laws of the Game, over the ragferee API."""

import os

import httpx
import streamlit as st

from ragferee.models import Answer

# Read from the environment, not Settings, so the UI needs no database credentials.
API_URL = os.environ.get("RAGFEREE_API_URL", "http://127.0.0.1:8000")

EXAMPLES = (
    "What happens if a goalkeeper holds the ball too long?",
    "Can a player be offside from a throw-in?",
    "Do players have to wear shin pads?",
    "Who won the 2022 World Cup?",
)

_ERRORS = {
    422: "Please enter a question (up to 500 characters).",
    503: "The database is unavailable. Is Postgres running (`make db-up`)?",
}


class ApiError(Exception):
    """A request to the ragferee API failed; the message is shown to the user."""


def ask(client: httpx.Client, question: str, *, limit: int) -> Answer:
    """Ask the API a question and return its cited answer."""
    try:
        response = client.post("/ask", json={"question": question, "limit": limit})
    except httpx.TimeoutException as error:
        raise ApiError("The API took too long to answer. Try again.") from error
    except httpx.TransportError as error:
        raise ApiError(
            f"Can't reach the ragferee API at {API_URL}. Is `make run` still running?"
        ) from error

    if response.status_code == 502:
        raise ApiError(f"The language model failed: {_detail(response)}")
    if response.status_code in _ERRORS:
        raise ApiError(_ERRORS[response.status_code])
    if response.is_error:
        raise ApiError(f"The API returned an error ({response.status_code}).")
    return Answer.model_validate_json(response.content)


def chunk_count(client: httpx.Client) -> int | None:
    """Return the number of indexed Law passages, or None if the API is not ready."""
    try:
        response = client.get("/health")
    except httpx.TransportError:
        return None
    if response.is_error:
        return None
    return int(response.json()["chunks"])


def format_sources(answer: Answer) -> str:
    """Return the answer's citations as a numbered Markdown list, matching its [n] markers."""
    return "\n".join(
        f"{number}. {citation}" for number, citation in enumerate(answer.citations, start=1)
    )


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json()["detail"])
    except (ValueError, KeyError, TypeError):
        return response.text or "no details"


# Streamlit app

Reply = Answer | ApiError


@st.cache_resource
def _client() -> httpx.Client:
    """One HTTP client, reused across Streamlit's reruns."""
    return httpx.Client(base_url=API_URL, timeout=90)


def _show_reply(reply: Reply) -> None:
    if isinstance(reply, ApiError):
        st.error(str(reply))
    elif not reply.citations:
        st.info(reply.text)
    else:
        st.markdown(reply.text)
        st.markdown(f"**Sources**\n\n{format_sources(reply)}")


def _sidebar() -> tuple[int, str | None]:
    """Draw the sidebar, and return the number of sources and any example question clicked."""
    example = None
    with st.sidebar:
        st.header("ragferee")
        chunks = chunk_count(_client())
        if chunks is None:
            st.warning("API starting or offline")
        else:
            st.success(f"API online · {chunks} Law passages indexed")

        limit = st.slider("Sources", min_value=1, max_value=10, value=5)

        st.subheader("Try")
        for question in EXAMPLES:
            if st.button(question, width="stretch"):
                example = question

        if st.button("Clear chat"):
            st.session_state.messages = []

        st.caption(
            "Not an official IFAB product. "
            "[Laws of the Game](https://www.theifab.com/laws/) · "
            f"[API docs]({API_URL}/docs)"
        )
    return limit, example


def main() -> None:
    """Run the chat page."""
    st.set_page_config(page_title="ragferee", page_icon="⚽")
    st.title("Ask about the Laws of the Game 2026/27")
    st.caption(
        "Answers come only from the IFAB Laws, with the Law, section and page for every "
        "statement. Each question is answered on its own."
    )

    limit, example = _sidebar()
    messages: list[tuple[str, str | Reply]] = st.session_state.setdefault("messages", [])

    for role, content in messages:
        with st.chat_message(role):
            if isinstance(content, str):
                st.markdown(content)
            else:
                _show_reply(content)

    question = st.chat_input("Ask a question…", max_chars=500) or example
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        reply: Reply
        with st.spinner("Reading the Laws…"):
            try:
                reply = ask(_client(), question, limit=limit)
            except ApiError as error:
                reply = error
        _show_reply(reply)
    messages += [("user", question), ("assistant", reply)]


if __name__ == "__main__":
    main()
