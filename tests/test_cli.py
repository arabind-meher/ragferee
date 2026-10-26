from collections.abc import Iterator
from pathlib import Path

import pytest

from ragferee.answering import NOT_COVERED_MESSAGE
from ragferee.cli import _print_answer, main
from ragferee.config import get_settings
from ragferee.models import Answer, Citation


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    for name in ("POSTGRES_USER", "POSTGRES_PASSWORD", "LLM_PROVIDER", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_check_valid_settings_returns_0(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("POSTGRES_USER", "user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "s3cret-pw")

    exit_code = main(["check"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Settings OK" in out
    assert "s3cret-pw" not in out


def test_check_invalid_settings_returns_1(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["check"])

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "POSTGRES_USER" in err


def test_search_with_invalid_settings_returns_1(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["search", "offside"])

    assert exit_code == 1
    assert "ragferee check" in capsys.readouterr().err


def test_ask_with_invalid_settings_returns_1(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["ask", "What is offside?"])

    assert exit_code == 1
    assert "ragferee check" in capsys.readouterr().err


def test_serve_with_invalid_settings_returns_1(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["serve"])

    assert exit_code == 1
    assert "ragferee check" in capsys.readouterr().err


def test_eval_without_question_file_returns_1(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["eval"])

    assert exit_code == 1
    assert "retrieval.jsonl not found" in capsys.readouterr().err


def test_eval_with_missing_file_returns_1(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["eval", "--file", "missing.jsonl"])

    assert exit_code == 1
    assert "missing.jsonl not found" in capsys.readouterr().err


def test_print_answer_lists_sources(capsys: pytest.CaptureFixture[str]) -> None:
    citation = Citation(law_number=12, section="3. Corner kick", page=119)

    _print_answer(Answer(text="A corner kick [1].", citations=(citation,)))

    assert capsys.readouterr().out == (
        "A corner kick [1].\n\nSources:\n[1] Law 12, 3. Corner kick, p. 119\n"
    )


def test_print_answer_not_covered_has_no_sources(capsys: pytest.CaptureFixture[str]) -> None:
    _print_answer(Answer(text=NOT_COVERED_MESSAGE, citations=()))

    assert capsys.readouterr().out == f"{NOT_COVERED_MESSAGE}\n"


def test_print_answer_without_citations_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    _print_answer(Answer(text="A corner kick.", citations=()))

    assert "(no sources cited)" in capsys.readouterr().out
