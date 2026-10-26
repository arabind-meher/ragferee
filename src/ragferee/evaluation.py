"""Measure how well retrieval finds the sections that answer a question."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


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
