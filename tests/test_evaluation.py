from pathlib import Path

import pytest

from ragferee.evaluation import EvalCase, SectionRef, hit_rate, load_cases, mean_reciprocal_rank

RANKS = [1, 3, None, 6]


def test_hit_rate_counts_ranks_within_k() -> None:
    assert hit_rate(RANKS, k=1) == 0.25
    assert hit_rate(RANKS, k=5) == 0.5
    assert hit_rate(RANKS, k=10) == 0.75


def test_mean_reciprocal_rank_counts_misses_as_zero() -> None:
    assert mean_reciprocal_rank([1, 2, None]) == pytest.approx((1 + 0.5 + 0) / 3)


def test_metrics_of_no_questions_are_zero() -> None:
    assert hit_rate([], k=5) == 0.0
    assert mean_reciprocal_rank([]) == 0.0


def test_load_cases_reads_lines_and_skips_blank_ones(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.jsonl"
    path.write_text(
        '{"question": "When is a player offside?", '
        '"relevant": [{"law": 11, "section": "1. Offside position"}]}\n'
        "\n"
    )

    cases = load_cases(path)

    assert cases == [
        EvalCase(
            question="When is a player offside?",
            relevant=(SectionRef(law=11, section="1. Offside position"),),
        )
    ]


def test_load_cases_reports_the_bad_line_number(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.jsonl"
    path.write_text(
        '{"question": "Offside?", "relevant": [{"law": 11, "section": "1. Offside position"}]}\n'
        '{"question": "Bad law", "relevant": [{"law": 99, "section": "x"}]}\n'
    )

    with pytest.raises(ValueError, match=r"retrieval\.jsonl:2"):
        load_cases(path)


def test_load_cases_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.jsonl"
    path.write_text("\n")

    with pytest.raises(ValueError, match="no evaluation cases"):
        load_cases(path)
