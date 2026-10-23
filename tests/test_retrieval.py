import psycopg
import pytest
from psycopg.rows import TupleRow

from ragferee.embeddings import Embedder
from ragferee.models import Chunk
from ragferee.retrieval import RRF_K, Retriever, reciprocal_rank_fusion
from ragferee.storage import ChunkStore

JEWELLERY = Chunk(
    law_number=4,
    law_title="The Players' Equipment",
    section="1. Safety",
    page=67,
    text="Players must not use equipment or wear anything that is dangerous, including jewellery.",
)
OFFSIDE = Chunk(
    law_number=11,
    law_title="Offside",
    section="1. Offside position",
    page=109,
    text="It is not an offence to be in an offside position.",
)
CORNER = Chunk(
    law_number=17,
    law_title="The Corner Kick",
    section="1. Procedure",
    page=149,
    text="The ball is placed in the corner area nearest to where it crossed the goal line.",
)
CHUNKS = [JEWELLERY, OFFSIDE, CORNER]


@pytest.fixture
def retriever(
    store: ChunkStore, connection: psycopg.Connection[TupleRow], fake_embedder: Embedder
) -> Retriever:
    store.replace_all(CHUNKS, fake_embedder.embed_documents([c.text for c in CHUNKS]))
    return Retriever(connection, fake_embedder)


def test_id_in_both_lists_beats_top_of_one_list() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [4, 2, 5]])

    assert fused[0][0] == 2


def test_single_list_keeps_its_order() -> None:
    fused = reciprocal_rank_fusion([[7, 3, 9]])

    assert [item for item, _ in fused] == [7, 3, 9]


def test_ties_keep_first_seen_order() -> None:
    fused = reciprocal_rank_fusion([[1], [2]])

    assert [item for item, _ in fused] == [1, 2]


def test_empty_input_gives_empty_result() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_score_adds_one_over_k_plus_rank_per_list() -> None:
    fused = dict(reciprocal_rank_fusion([[1], [1]]))

    assert fused[1] == pytest.approx(2 / (RRF_K + 1))


def test_keyword_match_ranks_first(retriever: Retriever) -> None:
    results = retriever.search("jewellery")

    assert results[0].chunk == JEWELLERY


def test_identical_text_ranks_first_by_vector(retriever: Retriever) -> None:
    results = retriever.search(OFFSIDE.text)

    assert results[0].chunk == OFFSIDE


def test_limit_caps_the_number_of_results(retriever: Retriever) -> None:
    assert len(retriever.search("ball", limit=2)) == 2


def test_stop_words_only_still_returns_vector_results(retriever: Retriever) -> None:
    assert len(retriever.search("what is the")) == len(CHUNKS)


def test_empty_index_returns_no_results(
    store: ChunkStore, connection: psycopg.Connection[TupleRow], fake_embedder: Embedder
) -> None:
    assert Retriever(connection, fake_embedder).search("offside") == []


def test_invalid_arguments_raise(retriever: Retriever) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        retriever.search("   ")
    with pytest.raises(ValueError, match="at least 1"):
        retriever.search("offside", limit=0)
