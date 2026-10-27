"""Web chat for asking questions about the Laws of the Game, over the ragferee API."""

import os

import httpx

from ragferee.models import Answer

# Read from the environment, not Settings, so the UI needs no database credentials.
API_URL = os.environ.get("RAGFEREE_API_URL", "http://127.0.0.1:8000")

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
