from collections.abc import Iterator
from pathlib import Path

import pytest

from ragferee.cli import main
from ragferee.config import get_settings


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
