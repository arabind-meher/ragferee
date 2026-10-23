import pytest

from ragferee.retrieval import RRF_K, reciprocal_rank_fusion


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
