"""Generate text with a language model."""

from typing import TYPE_CHECKING, Protocol

from pydantic import SecretStr

from ragferee.config import Settings

if TYPE_CHECKING:
    from openai import OpenAI

NOT_COVERED = "NOT_COVERED"


class LLMError(Exception):
    """Raised when a language model request fails."""


class LLM(Protocol):
    """Anything that can complete a prompt with text."""

    def complete(self, system: str, prompt: str) -> str:
        """Return the model's reply to a prompt, following the system instructions."""
        ...


class MockLLM:
    """Returns a fixed answer without a network call, for trying the pipeline and tests."""

    def complete(self, system: str, prompt: str) -> str:
        """Cite the first source if the prompt has one, otherwise report not covered."""
        if "[1]" not in prompt:
            return NOT_COVERED
        return "This is a mock answer based on the retrieved Laws [1]."


class OpenAILLM:
    """Answers with an OpenAI model through the Responses API."""

    def __init__(self, api_key: SecretStr, model: str, *, client: "OpenAI | None" = None) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client

    def complete(self, system: str, prompt: str) -> str:
        """Return the model's reply to a prompt, following the system instructions."""
        from openai import OpenAIError

        try:
            response = self._get_client().responses.create(
                model=self._model, instructions=system, input=prompt
            )
        except OpenAIError as error:
            raise LLMError(f"OpenAI request failed: {error}") from error
        text = response.output_text.strip()
        if not text:
            raise LLMError("OpenAI model returned no text")
        return text

    def _get_client(self) -> "OpenAI":
        if self._client is None:
            from openai import OpenAI  # imported here: slow to load

            self._client = OpenAI(api_key=self._api_key.get_secret_value(), timeout=60.0)
        return self._client


def make_llm(settings: Settings) -> LLM:
    """Return the language model selected by the settings."""
    if settings.llm_provider == "mock":
        return MockLLM()
    if settings.openai_api_key is None:
        raise LLMError("OpenAI API key is required.")
    return OpenAILLM(settings.openai_api_key, settings.openai_model)
