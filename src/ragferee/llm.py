"""Generate text with a language model."""

from typing import Protocol

from ragferee.config import Settings

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


def make_llm(settings: Settings) -> LLM:
    """Return the language model selected by the settings."""
    if settings.llm_provider == "mock":
        return MockLLM()
    raise LLMError(f"LLM provider {settings.llm_provider!r} is not supported yet")
