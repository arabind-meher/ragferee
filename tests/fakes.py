"""Test doubles shared by the answering and API tests."""

from ragferee.models import Chunk
from ragferee.retrieval import SearchResult


def result(law: int, section: str, page: int, text: str) -> SearchResult:
    """Return a search result for a chunk with the given location and text."""
    chunk = Chunk(law_number=law, law_title=f"Law {law}", section=section, page=page, text=text)
    return SearchResult(chunk=chunk, score=0.0)


RESULTS = [
    result(12, "3. Corner kick", 119, "A corner kick is awarded if a goalkeeper holds the ball."),
    result(12, "1. Direct free kick", 116, "A direct free kick is awarded for a handball."),
    result(11, "1. Offside position", 108, "Being in an offside position is not an offence."),
]


class FakeSearcher:
    """Returns fixed results instead of searching a database."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(self, question: str, *, limit: int) -> list[SearchResult]:
        """Return the first results, up to the limit."""
        return self.results[:limit]


class FakeLLM:
    """Returns a fixed reply and records the prompts it receives."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str) -> str:
        """Record the prompt and return the fixed reply."""
        self.prompts.append(prompt)
        return self.reply
