from pathlib import Path

import psycopg
import pytest
from psycopg.rows import TupleRow

from ragferee.embeddings import Embedder
from ragferee.evaluation import (
    CaseResult,
    EvalCase,
    EvalReport,
    Evaluator,
    SectionRef,
    hit_rate,
    load_cases,
    mean_reciprocal_rank,
    unknown_sections,
)
from ragferee.models import Chunk
from ragferee.retrieval import Retriever, SearchMode
from ragferee.storage import ChunkStore

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


SAFETY = SectionRef(law=4, section="1. Safety")
PROCEDURE = SectionRef(law=17, section="1. Procedure")


def case(question: str, *relevant: SectionRef) -> EvalCase:
    return EvalCase(question=question, relevant=relevant)


def test_report_computes_metrics_and_misses() -> None:
    results = tuple(
        CaseResult(case=case(f"q{rank}", SAFETY), rank=rank, retrieved=()) for rank in (1, None, 3)
    )
    report = EvalReport(mode=SearchMode.HYBRID, limit=5, results=results)

    assert report.hit_rate_at(1) == pytest.approx(1 / 3)
    assert report.mrr == pytest.approx((1 + 1 / 3) / 3)
    assert [result.case.question for result in report.misses] == ["qNone"]


def test_unknown_sections_finds_typos() -> None:
    typo = SectionRef(law=4, section="1. Safty")
    cases = [case("Jewellery?", SAFETY), case("Earrings?", typo)]

    assert unknown_sections(cases, known={(4, "1. Safety")}) == [typo]


def test_evaluator_ranks_hits_and_records_misses(
    store: ChunkStore, connection: psycopg.Connection[TupleRow], fake_embedder: Embedder
) -> None:
    chunks = [
        Chunk(
            law_number=4,
            law_title="The Players' Equipment",
            section="1. Safety",
            page=67,
            text="Players must not wear anything dangerous, including jewellery.",
        ),
        Chunk(
            law_number=17,
            law_title="The Corner Kick",
            section="1. Procedure",
            page=149,
            text="The ball is placed in the corner area.",
        ),
    ]
    store.replace_all(chunks, fake_embedder.embed_documents([c.text for c in chunks]))
    evaluator = Evaluator(Retriever(connection, fake_embedder))

    report = evaluator.run([case("jewellery", SAFETY), case("jewellery", PROCEDURE)], limit=1)

    assert report.ranks == [1, None]
    assert report.results[0].retrieved[0].law_number == 4
