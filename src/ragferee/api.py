"""Web API for asking questions about the Laws of the Game."""

from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import version
from typing import Annotated, cast

import psycopg
from fastapi import APIRouter, Depends, FastAPI, Request
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field, StringConstraints

from ragferee.answering import Answerer, Searcher
from ragferee.embeddings import Embedder
from ragferee.llm import LLM
from ragferee.models import Answer
from ragferee.retrieval import Retriever

# Request models


class AskRequest(BaseModel):
    """A question to answer, and how many Law passages to answer from."""

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    limit: int = Field(default=5, ge=1, le=20)


# Shared services


@dataclass(frozen=True)
class Services:
    """Objects created once at startup and shared by every request."""

    pool: ConnectionPool[psycopg.Connection[TupleRow]]
    embedder: Embedder
    llm: LLM


def _services(request: Request) -> Services:
    return cast(Services, request.app.state.services)


# Dependencies


def get_connection(request: Request) -> Iterator[psycopg.Connection[TupleRow]]:
    """Lend a pooled database connection for the length of one request."""
    with _services(request).pool.connection() as connection:
        yield connection


def get_searcher(
    request: Request,
    connection: Annotated[psycopg.Connection[TupleRow], Depends(get_connection)],
) -> Searcher:
    """Return a retriever that searches with the request's connection."""
    return Retriever(connection, _services(request).embedder)


def get_llm(request: Request) -> LLM:
    """Return the shared language model."""
    return _services(request).llm


# Routes

router = APIRouter()


@router.post("/ask")
def ask(
    body: AskRequest,
    searcher: Annotated[Searcher, Depends(get_searcher)],
    llm: Annotated[LLM, Depends(get_llm)],
) -> Answer:
    """Answer a question from the Laws of the Game, citing Law, section and page."""
    return Answerer(searcher, llm, limit=body.limit).answer(body.question)


def create_app() -> FastAPI:
    """Create the web application."""
    app = FastAPI(
        title="ragferee",
        summary="Cited answers to questions about the IFAB Laws of the Game.",
        version=version("ragferee"),
    )
    app.include_router(router)
    return app


app = create_app()
