"""Web API for asking questions about the Laws of the Game."""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import version
from typing import Annotated, Literal, cast

import psycopg
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field, StringConstraints, ValidationError

from ragferee.answering import Answerer, Searcher
from ragferee.config import get_settings
from ragferee.embeddings import EMBEDDING_DIMENSION, MODELS_DIR, Embedder, FastEmbedder
from ragferee.llm import LLM, LLMError, make_llm
from ragferee.models import Answer
from ragferee.retrieval import Retriever
from ragferee.storage import ChunkStore

# Request and response models


class AskRequest(BaseModel):
    """A question to answer, and how many Law passages to answer from."""

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    limit: int = Field(default=5, ge=1, le=20)


class HealthResponse(BaseModel):
    """The service is up, with the number of indexed chunks."""

    status: Literal["ok"]
    chunks: int


# Shared services


@dataclass(frozen=True)
class Services:
    """Objects created once at startup and shared by every request."""

    pool: ConnectionPool[psycopg.Connection[TupleRow]]
    embedder: Embedder
    llm: LLM


def _services(request: Request) -> Services:
    return cast(Services, request.app.state.services)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the shared services at startup, and close the pool at shutdown."""
    try:
        settings = get_settings()
    except ValidationError as error:
        raise RuntimeError("Invalid settings; run `ragferee check` for details") from error

    pool: ConnectionPool[psycopg.Connection[TupleRow]] = ConnectionPool(
        settings.get_postgres_url(),
        min_size=1,
        max_size=4,
        kwargs={"connect_timeout": 5},
        check=ConnectionPool.check_connection,
        open=False,
    )
    try:
        pool.open(wait=True, timeout=10)
    except psycopg.OperationalError as error:
        raise RuntimeError("Cannot reach Postgres; start it with `make db-up`") from error

    try:
        with pool.connection() as connection:
            if ChunkStore(connection, EMBEDDING_DIMENSION).count() == 0:
                raise RuntimeError("No chunks indexed; run `ragferee index` first")
        embedder = FastEmbedder(MODELS_DIR)
        embedder.embed_query("warm up")  # load the model now, not on the first request
        app.state.services = Services(pool=pool, embedder=embedder, llm=make_llm(settings))
        yield
    finally:
        pool.close()


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


@router.get("/health", responses={503: {"description": "Database unavailable or empty"}})
def health(
    connection: Annotated[psycopg.Connection[TupleRow], Depends(get_connection)],
) -> HealthResponse:
    """Check that the database is reachable and chunks are indexed."""
    chunks = ChunkStore(connection, EMBEDDING_DIMENSION).count()
    if chunks == 0:
        raise HTTPException(status_code=503, detail="No chunks indexed")
    return HealthResponse(status="ok", chunks=chunks)


@router.post("/ask")
def ask(
    body: AskRequest,
    searcher: Annotated[Searcher, Depends(get_searcher)],
    llm: Annotated[LLM, Depends(get_llm)],
) -> Answer:
    """Answer a question from the Laws of the Game, citing Law, section and page."""
    return Answerer(searcher, llm, limit=body.limit).answer(body.question)


# Errors


def _llm_failed(request: Request, error: Exception) -> JSONResponse:
    """The API worked but the language model behind it failed, so 502 Bad Gateway."""
    return JSONResponse(status_code=502, content={"detail": str(error)})


def _database_unavailable(request: Request, error: Exception) -> JSONResponse:
    """Hide connection details from the client."""
    return JSONResponse(status_code=503, content={"detail": "Database unavailable"})


def create_app() -> FastAPI:
    """Create the web application."""
    app = FastAPI(
        title="ragferee",
        summary="Cited answers to questions about the IFAB Laws of the Game.",
        version=version("ragferee"),
        lifespan=_lifespan,
    )
    app.include_router(router)
    app.add_exception_handler(LLMError, _llm_failed)
    app.add_exception_handler(psycopg.OperationalError, _database_unavailable)
    return app


app = create_app()
