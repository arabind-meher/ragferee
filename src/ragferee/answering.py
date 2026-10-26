"""Answer questions from retrieved Law text, with citations."""

import re
from collections.abc import Sequence
from typing import Protocol

from ragferee.llm import LLM, NOT_COVERED
from ragferee.models import Answer
from ragferee.retrieval import SearchResult

NOT_COVERED_MESSAGE = "The Laws of the Game don't seem to cover this. Try rephrasing the question."

_SYSTEM = f"""\
You answer questions about the IFAB Laws of the Game 2026/27 for players, fans and new referees.
Use only the numbered sources provided. Do not use outside knowledge.
After each statement, cite the source it comes from with its number in square brackets, \
e.g. [2]. Cite several sources as [1][3].
Explain in plain language, in a few sentences.
If the sources do not answer the question, reply exactly: {NOT_COVERED}"""

_MARKER = re.compile(r"[ \t]*\[(\d+)\]")


class Searcher(Protocol):
    """Anything that can find the chunks most relevant to a question."""

    def search(self, question: str, *, limit: int) -> list[SearchResult]:
        """Return the chunks most relevant to the question, best first."""
        ...


def build_prompt(question: str, results: Sequence[SearchResult]) -> tuple[str, str]:
    """Return the system instructions and a prompt with numbered sources and the question."""
    sources = "\n\n".join(
        f"[{number}] {result.chunk.citation()}\n{result.chunk.text}"
        for number, result in enumerate(results, start=1)
    )
    return _SYSTEM, f"Sources:\n\n{sources}\n\nQuestion: {question}"


def cite(text: str, results: Sequence[SearchResult]) -> Answer:
    """Turn the [n] source markers in a reply into citations.

    Markers for sources that were not given are removed. The rest are renumbered in
    order of first appearance, so [1] in the text is always the first citation.
    """
    cited: list[int] = []  # source indexes, in order of first appearance

    def renumber(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        if not 0 <= index < len(results):
            return ""
        if index not in cited:
            cited.append(index)
        return f" [{cited.index(index) + 1}]"

    renumbered = _MARKER.sub(renumber, text).strip()
    citations = tuple(results[index].chunk.citation() for index in cited)
    return Answer(text=renumbered, citations=citations)


class Answerer:
    """Answers questions from the most relevant chunks, citing only those chunks."""

    def __init__(self, searcher: Searcher, llm: LLM, *, limit: int = 5) -> None:
        self._searcher = searcher
        self._llm = llm
        self._limit = limit

    def answer(self, question: str) -> Answer:
        """Return a cited answer, or a not-covered answer if the Laws don't cover it."""
        if not question.strip():
            raise ValueError("Question must not be empty")

        results = self._searcher.search(question, limit=self._limit)
        if not results:
            return Answer(text=NOT_COVERED_MESSAGE, citations=())

        reply = self._llm.complete(*build_prompt(question, results))
        if reply.strip() == NOT_COVERED:
            return Answer(text=NOT_COVERED_MESSAGE, citations=())
        return cite(reply, results)
