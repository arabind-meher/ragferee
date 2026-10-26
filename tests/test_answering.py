import pytest

from ragferee.answering import NOT_COVERED_MESSAGE, Answerer, build_prompt, cite
from ragferee.llm import NOT_COVERED
from ragferee.models import Chunk, Citation
from ragferee.retrieval import SearchResult


def _result(law: int, section: str, page: int, text: str) -> SearchResult:
    chunk = Chunk(law_number=law, law_title=f"Law {law}", section=section, page=page, text=text)
    return SearchResult(chunk=chunk, score=0.0)


RESULTS = [
    _result(12, "3. Corner kick", 119, "A corner kick is awarded if a goalkeeper holds the ball."),
    _result(12, "1. Direct free kick", 116, "A direct free kick is awarded for a handball."),
    _result(11, "1. Offside position", 108, "Being in an offside position is not an offence."),
]


class FakeSearcher:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(self, question: str, *, limit: int) -> list[SearchResult]:
        return self.results[:limit]


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_build_prompt_numbers_sources() -> None:
    system, prompt = build_prompt("Who takes a corner kick?", RESULTS[:2])

    assert NOT_COVERED in system
    assert "[1] Law 12, 3. Corner kick, p. 119\nA corner kick is awarded" in prompt
    assert "[2] Law 12, 1. Direct free kick, p. 116" in prompt
    assert prompt.endswith("Question: Who takes a corner kick?")


def test_cite_renumbers_in_order_of_appearance() -> None:
    answer = cite("Offside first [3]. Again [3]. Then a corner [1].", RESULTS)

    assert answer.text == "Offside first [1]. Again [1]. Then a corner [2]."
    assert answer.citations == (
        Citation(law_number=11, section="1. Offside position", page=108),
        Citation(law_number=12, section="3. Corner kick", page=119),
    )


def test_cite_removes_unknown_sources() -> None:
    answer = cite("A corner kick [1][7].", RESULTS)

    assert answer.text == "A corner kick [1]."
    assert len(answer.citations) == 1


def test_cite_without_markers_has_no_citations() -> None:
    answer = cite("A corner kick is awarded.", RESULTS)

    assert answer.text == "A corner kick is awarded."
    assert answer.citations == ()


def test_answerer_without_results_skips_llm() -> None:
    llm = FakeLLM("should not be used")

    answer = Answerer(FakeSearcher([]), llm).answer("What is offside?")

    assert answer.text == NOT_COVERED_MESSAGE
    assert answer.citations == ()
    assert llm.prompts == []


def test_answerer_not_covered_reply() -> None:
    answer = Answerer(FakeSearcher(RESULTS), FakeLLM(f" {NOT_COVERED}\n")).answer("Who won?")

    assert answer.text == NOT_COVERED_MESSAGE
    assert answer.citations == ()


def test_answerer_cites_sources_from_prompt() -> None:
    llm = FakeLLM("The other team gets a corner kick [1].")

    answer = Answerer(FakeSearcher(RESULTS), llm, limit=2).answer("Goalkeeper holds the ball?")

    assert answer.text == "The other team gets a corner kick [1]."
    assert answer.citations == (RESULTS[0].chunk.citation(),)
    assert "[2] Law 12, 1. Direct free kick" in llm.prompts[0]
    assert "[3]" not in llm.prompts[0]


def test_answerer_rejects_blank_question() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Answerer(FakeSearcher(RESULTS), FakeLLM("unused")).answer("   ")
