"""Command-line interface."""

import argparse
import sys
import textwrap
from importlib.metadata import version
from pathlib import Path

import httpx
import psycopg
from psycopg.rows import TupleRow
from pydantic import ValidationError

from ragferee.answering import NOT_COVERED_MESSAGE, Answerer
from ragferee.config import get_settings
from ragferee.download import LAWS_OF_THE_GAME, Downloader, DownloadError
from ragferee.embeddings import EMBEDDING_DIMENSION, MODELS_DIR, FastEmbedder
from ragferee.evaluation import EvalReport, Evaluator, load_cases, unknown_sections
from ragferee.ingest import LawsParser, load_chunks, read_lines, save_chunks
from ragferee.llm import LLMError, make_llm
from ragferee.retrieval import KEYWORD_WEIGHT, Retriever, SearchMode
from ragferee.storage import ChunkStore

RAW_DATA_DIR = Path("data/raw")
CHUNKS_PATH = Path("data/processed/chunks.json")
EVAL_PATH = Path("eval/retrieval.jsonl")


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
    if args.command == "ask":
        return _ask(args.question, limit=args.limit)
    if args.command == "eval":
        return _evaluate(
            args.file,
            limit=args.limit,
            mode=SearchMode(args.mode),
            compare=args.compare,
            keyword_weight=args.keyword_weight,
        )
    if args.command == "serve":
        return _serve(host=args.host, port=args.port)
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
    ask = commands.add_parser("ask", help="answer a question with citations to the Laws")
    ask.add_argument("question", help="the question to answer")
    ask.add_argument("--limit", type=int, default=5, help="sources to answer from (default: 5)")
    evaluate = commands.add_parser("eval", help="measure how well search finds expected sections")
    evaluate.add_argument("--limit", type=int, default=5, help="results per question (default: 5)")
    evaluate.add_argument(
        "--mode", choices=list(SearchMode), default=SearchMode.HYBRID, help="search mode"
    )
    evaluate.add_argument("--compare", action="store_true", help="compare all search modes")
    evaluate.add_argument(
        "--file", type=Path, default=EVAL_PATH, help=f"question file (default: {EVAL_PATH})"
    )
    evaluate.add_argument(
        "--keyword-weight",
        type=float,
        default=KEYWORD_WEIGHT,
        help=f"weight of keyword results in hybrid search (default: {KEYWORD_WEIGHT})",
    )
    serve = commands.add_parser("serve", help="run the web API")
    serve.add_argument("--host", default="127.0.0.1", help="address to bind (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
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
    connection = _connect()
    if connection is None:
        return 1

    chunks = load_chunks(CHUNKS_PATH)
    embedder = FastEmbedder(MODELS_DIR)
    with connection:
        store = ChunkStore(connection, embedder.dimension)
        try:
            store.create_schema()
            print(f"Embedding {len(chunks)} chunks...")
            embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
            store.replace_all(chunks, embeddings)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        count = store.count()

    print(f"Indexed {count} chunks")
    return 0


def _search(question: str, *, limit: int) -> int:
    connection = _connect()
    if connection is None:
        return 1

    embedder = FastEmbedder(MODELS_DIR)
    with connection:
        if not _has_index(connection, embedder.dimension):
            return 1
        try:
            results = Retriever(connection, embedder).search(question, limit=limit)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result.chunk.citation()}  ({result.score:.4f})")
        print(f"   {textwrap.shorten(result.chunk.text, width=100, placeholder=' …')}")
    return 0


def _ask(question: str, *, limit: int) -> int:
    connection = _connect()
    if connection is None:
        return 1

    embedder = FastEmbedder(MODELS_DIR)
    with connection:
        if not _has_index(connection, embedder.dimension):
            return 1
        try:
            answerer = Answerer(
                Retriever(connection, embedder), make_llm(get_settings()), limit=limit
            )
            answer = answerer.answer(question)
        except (ValueError, LLMError) as error:
            print(error, file=sys.stderr)
            return 1

    print(answer.text)
    if answer.text == NOT_COVERED_MESSAGE:
        return 0
    print()
    if not answer.citations:
        print("(no sources cited)")
        return 0
    print("Sources:")
    for number, citation in enumerate(answer.citations, start=1):
        print(f"[{number}] {citation}")
    return 0


def _evaluate(
    path: Path, *, limit: int, mode: SearchMode, compare: bool, keyword_weight: float
) -> int:
    if not path.exists():
        print(f"{path} not found; add evaluation questions first", file=sys.stderr)
        return 1
    try:
        cases = load_cases(path)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    connection = _connect()
    if connection is None:
        return 1

    embedder = FastEmbedder(MODELS_DIR)
    with connection:
        if not _has_index(connection, embedder.dimension):
            return 1
        unknown = unknown_sections(cases, ChunkStore(connection, embedder.dimension).sections())
        if unknown:
            print(f"Sections in {path} not found in the index:", file=sys.stderr)
            for ref in unknown:
                print(f"  Law {ref.law}, {ref.section}", file=sys.stderr)
            return 1
        modes = list(SearchMode) if compare else [mode]
        try:
            evaluator = Evaluator(Retriever(connection, embedder, keyword_weight=keyword_weight))
            reports = [evaluator.run(cases, limit=limit, mode=m) for m in modes]
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1

    if compare:
        _print_comparison(reports)
    else:
        _print_report(reports[0])
    return 0


def _serve(*, host: str, port: int) -> int:
    # The API checks these at startup too; checking first gives a short message, not a traceback.
    connection = _connect()
    if connection is None:
        return 1
    with connection:
        if not _has_index(connection, EMBEDDING_DIMENSION):
            return 1

    import uvicorn  # imported here: only needed to serve

    uvicorn.run("ragferee.api:app", host=host, port=port)
    return 0


def _print_report(report: EvalReport) -> None:
    print(f"Retrieval ({report.mode}, top {report.limit}) on {len(report.results)} questions")
    scores = [f"hit@{k} {report.hit_rate_at(k):.2f}" for k in _cutoffs(report.limit)]
    print("  " + "   ".join([*scores, f"MRR {report.mrr:.2f}"]))
    if report.misses:
        print(f"Misses ({len(report.misses)}):")
    for miss in report.misses:
        expected = "; ".join(f"Law {ref.law}, {ref.section}" for ref in miss.case.relevant)
        retrieved = " · ".join(str(citation) for citation in miss.retrieved[:3])
        print(f'  "{miss.case.question}"')
        print(f"     expected: {expected}")
        print(f"     got:      {retrieved or 'nothing'}")


def _print_comparison(reports: list[EvalReport]) -> None:
    cutoffs = _cutoffs(reports[0].limit)
    print(f"Retrieval on {len(reports[0].results)} questions")
    print("  " + " ".join([f"{'mode':<8}", *(f"{f'hit@{k}':>6}" for k in cutoffs), f"{'MRR':>6}"]))
    for report in reports:
        scores = (f"{report.hit_rate_at(k):>6.2f}" for k in cutoffs)
        print("  " + " ".join([f"{report.mode:<8}", *scores, f"{report.mrr:>6.2f}"]))


def _cutoffs(limit: int) -> list[int]:
    """Return the k values to report hit rate for: 1 and the limit, without repeats."""
    return sorted({1, limit})


def _connect() -> psycopg.Connection[TupleRow] | None:
    """Open a database connection, or print why not and return None."""
    try:
        url = get_settings().get_postgres_url()
    except ValidationError:
        print("Invalid settings; run `ragferee check` for details", file=sys.stderr)
        return None
    try:
        return psycopg.connect(url, connect_timeout=5)
    except psycopg.OperationalError as error:
        print("Cannot reach Postgres; start it with `make db-up`", file=sys.stderr)
        print(error, file=sys.stderr)
        return None


def _has_index(connection: psycopg.Connection[TupleRow], dimension: int) -> bool:
    """Return whether chunks are indexed, printing a hint if not."""
    if ChunkStore(connection, dimension).count() == 0:
        print("No chunks indexed; run `ragferee index` first", file=sys.stderr)
        return False
    return True
