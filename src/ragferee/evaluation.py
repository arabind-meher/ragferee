"""Measure how well retrieval finds the sections that answer a question."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragferee.models import Citation
from ragferee.retrieval import Retriever, SearchMode


class SectionRef(BaseModel):
    """A Law section that correctly answers an evaluation question."""

    model_config = ConfigDict(frozen=True)

    law: int = Field(ge=1, le=17)
    section: str = Field(min_length=1)


class EvalCase(BaseModel):
    """One evaluation question and the sections that answer it."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1)
    relevant: tuple[SectionRef, ...] = Field(min_length=1)


def load_cases(path: Path) -> list[EvalCase]:
    """Read evaluation cases from a JSON Lines file, one case per line."""
    cases: list[EvalCase] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.model_validate_json(line))
        except ValidationError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    if not cases:
        raise ValueError(f"{path} contains no evaluation cases")
    return cases


def hit_rate(ranks: Sequence[int | None], k: int) -> float:
    """Return the share of questions whose first relevant result is in the top k.

    A rank is 1 for the first result; None means no relevant result was found.
    """
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    """Return the mean of 1/rank over all questions, counting misses as 0."""
    if not ranks:
        return 0.0
    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)


def unknown_sections(cases: Sequence[EvalCase], known: set[tuple[int, str]]) -> list[SectionRef]:
    """Return expected sections that do not exist in the index, such as typos."""
    return [ref for case in cases for ref in case.relevant if (ref.law, ref.section) not in known]


class CaseResult(BaseModel):
    """How retrieval did on one evaluation question."""

    model_config = ConfigDict(frozen=True)

    case: EvalCase
    rank: int | None  # 1-based rank of the first relevant result; None if it was missed
    retrieved: tuple[Citation, ...]


class EvalReport(BaseModel):
    """The results of one evaluation run, with its metrics."""

    model_config = ConfigDict(frozen=True)

    mode: SearchMode
    limit: int
    results: tuple[CaseResult, ...]

    @property
    def ranks(self) -> list[int | None]:
        """Return the rank of the first relevant result for each question."""
        return [result.rank for result in self.results]

    @property
    def mrr(self) -> float:
        """Return the mean reciprocal rank of this run."""
        return mean_reciprocal_rank(self.ranks)

    @property
    def misses(self) -> list[CaseResult]:
        """Return the questions with no relevant result."""
        return [result for result in self.results if result.rank is None]

    def hit_rate_at(self, k: int) -> float:
        """Return the share of questions with a relevant result in the top k."""
        return hit_rate(self.ranks, k)


class Evaluator:
    """Runs evaluation questions through a retriever and records how each one went."""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def run(
        self, cases: Sequence[EvalCase], *, limit: int = 5, mode: SearchMode = SearchMode.HYBRID
    ) -> EvalReport:
        """Search for every question and report where the expected sections ranked."""
        results = tuple(self._evaluate(case, limit=limit, mode=mode) for case in cases)
        return EvalReport(mode=mode, limit=limit, results=results)

    def _evaluate(self, case: EvalCase, *, limit: int, mode: SearchMode) -> CaseResult:
        found = self._retriever.search(case.question, limit=limit, mode=mode)
        expected = {(ref.law, ref.section) for ref in case.relevant}
        rank = next(
            (
                position
                for position, result in enumerate(found, start=1)
                if (result.chunk.law_number, result.chunk.section) in expected
            ),
            None,
        )
        return CaseResult(
            case=case, rank=rank, retrieved=tuple(result.chunk.citation() for result in found)
        )
