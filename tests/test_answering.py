import pytest

from ragferee.answering import NOT_COVERED_MESSAGE, Answerer, build_prompt, cite
from ragferee.llm import NOT_COVERED
from ragferee.models import Citation
from tests.fakes import RESULTS, FakeLLM, FakeSearcher


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


def test_cite_with_only_unknown_markers_is_not_covered() -> None:
    answer = cite(" [7] ", RESULTS)

    assert answer.text == NOT_COVERED_MESSAGE
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
