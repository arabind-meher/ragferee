import pytest
from pydantic import SecretStr, ValidationError

from ragferee.config import Settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep settings from the shell out of these tests; each test sets what it needs."""
    for name in (
        "POSTGRES_DB",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_url_format() -> None:
    settings = Settings(
        _env_file=None, postgres_user="user", postgres_password=SecretStr("password")
    )

    url = settings.get_postgres_url()

    assert url == "postgresql://user:password@localhost:5432/ragferee"


def test_password_escaping() -> None:
    settings = Settings(
        _env_file=None, postgres_user="user", postgres_password=SecretStr("p@ss/w:rd")
    )

    url = settings.get_postgres_url()

    assert url == "postgresql://user:p%40ss%2Fw%3Ard@localhost:5432/ragferee"


def test_openai_without_key() -> None:
    with pytest.raises(ValidationError, match="OpenAI API key is required"):
        Settings(
            _env_file=None,
            postgres_user="user",
            postgres_password=SecretStr("password"),
            llm_provider="openai",
        )


def test_openai_with_empty_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(ValidationError, match="OpenAI API key is required"):
        Settings(
            _env_file=None,
            postgres_user="user",
            postgres_password=SecretStr("password"),
        )
