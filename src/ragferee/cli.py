"""Command-line interface."""

import argparse
import sys
from importlib.metadata import version

from pydantic import ValidationError

from ragferee.config import get_settings


def main(argv: list[str] | None = None) -> int:
    """Run the ragferee command line and return an exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return _check()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragferee",
        description="Q&A assistant for the IFAB Laws of the Game.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('ragferee')}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="validate settings from .env and the environment")
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
