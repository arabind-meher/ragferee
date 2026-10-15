from ragferee.models import Chunk, Citation


def test_chunk_citation_copies_location() -> None:
    chunk = Chunk(
        law_number=12,
        law_title="Fouls and Misconduct",
        section="3. Disciplinary action",
        page=112,
        text="A player who commits a cautionable offence is cautioned.",
    )

    citation = chunk.citation()

    assert citation == Citation(law_number=12, section="3. Disciplinary action", page=112)


def test_citation_str_format() -> None:
    citation = Citation(law_number=12, section="3. Disciplinary action", page=112)

    assert str(citation) == "Law 12, 3. Disciplinary action, p. 112"
