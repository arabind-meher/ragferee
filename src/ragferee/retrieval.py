"""Find the chunks most relevant to a question."""

from collections.abc import Sequence

RRF_K = 60  # standard Reciprocal Rank Fusion constant (Cormack et al., 2009)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = RRF_K
) -> list[tuple[int, float]]:
    """Merge ranked id lists into one ranking, the best first.

    Each id scores 1 / (k + rank) for every list it appears in. Ties keep the order
    in which ids were first seen, so the result is deterministic.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda entry: entry[1], reverse=True)
