"""Command-line interface."""

import argparse
import sys
import textwrap
from importlib.metadata import version
from pathlib import Path

import httpx
import psycopg
from pydantic import ValidationError

from ragferee.config import get_settings
from ragferee.download import LAWS_OF_THE_GAME, Downloader, DownloadError
from ragferee.embeddings import FastEmbedder
from ragferee.ingest import LawsParser, load_chunks, read_lines, save_chunks
from ragferee.retrieval import Retriever
from ragferee.storage import ChunkStore

RAW_DATA_DIR = Path("data/raw")
CHUNKS_PATH = Path("data/processed/chunks.json")
MODELS_DIR = Path("data/models")


def main(argv: list[str] | None = None) -> int:
    """Run the ragferee command line and return an exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return _check()
    if args.command == "download":
        return _download(force=args.force)
    if args.command == "ingest":
        return _ingest()
    if args.command == "index":
        return _index()
    if args.command == "search":
        return _search(args.question, limit=args.limit)
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragferee",
        description="Q&A assistant for the IFAB Laws of the Game.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('ragferee')}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="validate settings from .env and the environment")
    download = commands.add_parser("download", help="download the Laws of the Game PDF")
    download.add_argument("--force", action="store_true", help="download again even if present")
    commands.add_parser("ingest", help="split the downloaded PDF into citable chunks")
    commands.add_parser("index", help="embed the chunks and store them in Postgres")
    search = commands.add_parser("search", help="show the chunks most relevant to a question")
    search.add_argument("question", help="the question to search for")
    search.add_argument("--limit", type=int, default=5, help="number of results (default: 5)")
    return parser


def _check() -> int:
    try:
        settings = get_settings()
    except ValidationError as error:
        print("Invalid settings:", file=sys.stderr)
        for detail in error.errors(include_input=False, include_url=False):
            field = ".".join(str(part) for part in detail["loc"]).upper()
            message = f"{field}: {detail['msg']}" if field else detail["msg"]
            print(f"  {message}", file=sys.stderr)
        return 1

    print("Settings OK")
    print(f"database: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
    print(f"llm provider: {settings.llm_provider}")
    return 0


def _download(*, force: bool) -> int:
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        downloader = Downloader(client, RAW_DATA_DIR)
        try:
            path = downloader.download(LAWS_OF_THE_GAME, force=force)
        except DownloadError as error:
            print(error, file=sys.stderr)
            return 1

    print(f"PDF ready at {path}")
    return 0


def _ingest() -> int:
    pdf = RAW_DATA_DIR / LAWS_OF_THE_GAME.filename
    if not pdf.exists():
        print(f"{pdf} not found; run `ragferee download` first", file=sys.stderr)
        return 1

    chunks = LawsParser().parse(read_lines(pdf))
    save_chunks(chunks, CHUNKS_PATH)
    print(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")
    return 0


def _index() -> int:
    if not CHUNKS_PATH.exists():
        print(f"{CHUNKS_PATH} not found; run `ragferee ingest` first", file=sys.stderr)
        return 1
    try:
        url = get_settings().get_postgres_url()
    except ValidationError:
        print("Invalid settings; run `ragferee check` for details", file=sys.stderr)
        return 1

    chunks = load_chunks(CHUNKS_PATH)
    embedder = FastEmbedder(MODELS_DIR)
    try:
        with psycopg.connect(url, connect_timeout=5) as connection:
            store = ChunkStore(connection, embedder.dimension)
            store.create_schema()
            print(f"Embedding {len(chunks)} chunks...")
            embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
            store.replace_all(chunks, embeddings)
            count = store.count()
    except psycopg.OperationalError as error:
        print("Cannot reach Postgres; start it with `make db-up`", file=sys.stderr)
        print(error, file=sys.stderr)
        return 1
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    print(f"Indexed {count} chunks")
    return 0


def _search(question: str, *, limit: int) -> int:
    try:
        url = get_settings().get_postgres_url()
    except ValidationError:
        print("Invalid settings; run `ragferee check` for details", file=sys.stderr)
        return 1

    embedder = FastEmbedder(MODELS_DIR)
    try:
        with psycopg.connect(url, connect_timeout=5) as connection:
            if ChunkStore(connection, embedder.dimension).count() == 0:
                print("No chunks indexed; run `ragferee index` first", file=sys.stderr)
                return 1
            results = Retriever(connection, embedder).search(question, limit=limit)
    except psycopg.OperationalError as error:
        print("Cannot reach Postgres; start it with `make db-up`", file=sys.stderr)
        print(error, file=sys.stderr)
        return 1
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result.chunk.citation()}  ({result.score:.4f})")
        print(f"   {textwrap.shorten(result.chunk.text, width=100, placeholder=' …')}")
    return 0
