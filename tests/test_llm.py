import json

import httpx2
import pytest
from openai import OpenAI
from pydantic import SecretStr

from ragferee.config import Settings
from ragferee.llm import NOT_COVERED, LLMError, MockLLM, OpenAILLM, make_llm


def _reply(text: str) -> dict[str, object]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5-mini",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }


def _openai_llm(status: int, body: dict[str, object], requests: list[httpx2.Request]) -> OpenAILLM:
    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(status, json=body)

    client = OpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handle)),
    )
    return OpenAILLM(SecretStr("test-key"), "gpt-5-mini", client=client)


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


def test_make_llm_returns_openai() -> None:
    settings = Settings(
        _env_file=None,
        postgres_user="user",
        postgres_password=SecretStr("password"),
        llm_provider="openai",
        openai_api_key=SecretStr("test-key"),
    )

    assert isinstance(make_llm(settings), OpenAILLM)


def test_openai_sends_prompt_and_returns_text() -> None:
    requests: list[httpx2.Request] = []
    llm = _openai_llm(200, _reply(" A corner kick [1]. \n"), requests)

    reply = llm.complete("system rules", "the prompt")

    sent = json.loads(requests[0].content)
    assert reply == "A corner kick [1]."
    assert sent["model"] == "gpt-5-mini"
    assert sent["instructions"] == "system rules"
    assert sent["input"] == "the prompt"


def test_openai_error_becomes_llm_error() -> None:
    llm = _openai_llm(401, {"error": {"message": "Incorrect API key"}}, [])

    with pytest.raises(LLMError, match="OpenAI request failed"):
        llm.complete("system", "prompt")


def test_openai_empty_reply_is_llm_error() -> None:
    llm = _openai_llm(200, _reply("  "), [])

    with pytest.raises(LLMError, match="no text"):
        llm.complete("system", "prompt")
