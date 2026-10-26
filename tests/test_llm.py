from pydantic import SecretStr

from ragferee.config import Settings
from ragferee.llm import NOT_COVERED, MockLLM, make_llm


def test_mock_cites_first_source() -> None:
    reply = MockLLM().complete("system", "Sources:\n\n[1] Law 12, 3. Corner kick, p. 119\n...")

    assert "[1]" in reply


def test_mock_without_sources_is_not_covered() -> None:
    reply = MockLLM().complete("system", "Question: Who won the World Cup?")

    assert reply == NOT_COVERED


def test_make_llm_returns_mock() -> None:
    settings = Settings(
        _env_file=None,
        postgres_user="user",
        postgres_password=SecretStr("password"),
        llm_provider="mock",
    )

    assert isinstance(make_llm(settings), MockLLM)
