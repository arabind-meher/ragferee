from pathlib import Path

import pytest
from pydantic import ValidationError

from ragferee.ingest import LawsParser, Line, load_chunks, save_chunks
from ragferee.models import Chunk


def line(page: int, text: str, size: float = 9.0, *, bold: bool = False) -> Line:
    return Line(page=page, text=text, size=size, bold=bold)


def law_start(number: int, title: str, page: int) -> list[Line]:
    return [line(page, f"Law{number}", size=45.0), line(page + 1, title, size=33.0)]


def test_chunk_is_tagged_with_law_section_and_page() -> None:
    lines = [
        *law_start(12, "Fouls and Misconduct", page=114),
        line(115, "1. Direct free kick", bold=True),
        line(115, "A direct free kick is awarded if a player charges an opponent."),
    ]

    chunks = LawsParser().parse(lines)

    assert len(chunks) == 1
    assert chunks[0].law_number == 12
    assert chunks[0].law_title == "Fouls and Misconduct"
    assert chunks[0].section == "1. Direct free kick"
    assert chunks[0].page == 115


def test_section_across_pages_gives_one_chunk_per_page() -> None:
    lines = [
        *law_start(1, "The Field of Play", page=44),
        line(45, "2. Field markings", bold=True),
        line(45, "The field of play must be rectangular."),
        line(46, "The two longer boundary lines are touchlines."),
    ]

    chunks = LawsParser().parse(lines)

    assert [(c.section, c.page) for c in chunks] == [
        ("2. Field markings", 45),
        ("2. Field markings", 46),
    ]


def test_text_before_first_section_uses_law_title() -> None:
    lines = [
        *law_start(12, "Fouls and Misconduct", page=114),
        line(115, "Free kicks can only be awarded when the ball is in play."),
    ]

    chunks = LawsParser().parse(lines)

    assert chunks[0].section == "Fouls and Misconduct"


def test_footers_and_stray_labels_are_skipped() -> None:
    lines = [
        *law_start(11, "Offside", page=108),
        line(109, "1. Offside position", bold=True),
        line(109, "It is not an offence to be in an offside position."),
        line(109, "Laws of the Game 2026/27 | Law 11 | Offside", size=6.0),
        line(110, "11"),
    ]

    chunks = LawsParser().parse(lines)

    assert len(chunks) == 1
    assert "Laws of the Game" not in chunks[0].text


def test_saved_chunks_load_back_unchanged(tmp_path: Path) -> None:
    chunks = [
        Chunk(
            law_number=11,
            law_title="Offside",
            section="1. Offside position",
            page=109,
            text="It is not an offence to be in an offside position.",
        ),
    ]
    path = tmp_path / "processed" / "chunks.json"

    save_chunks(chunks, path)

    assert load_chunks(path) == chunks


def test_load_chunks_rejects_invalid_data(tmp_path: Path) -> None:
    path = tmp_path / "chunks.json"
    path.write_text(
        '[{"law_number": 99, "law_title": "x", "section": "x", "page": 1, "text": "x"}]'
    )

    with pytest.raises(ValidationError):
        load_chunks(path)
