"""Fixed-effect rank and cluster correction counts on a retained sample."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Exact multiway numerical rank is intentionally bounded: never silently create
# a huge dummy matrix. One-/two-way rank calculations do not use this budget.
_MAX_DUMMY_BYTES = 64 * 1024**2
_MAX_DUMMY_MIN_DIMENSION = 2000


@dataclass(frozen=True)
class EffectRank:
    rank: int
    exact: bool
    method: str


def is_nested(fine: np.ndarray, coarse: np.ndarray) -> bool:
    """Whether each observed fine category belongs to exactly one coarse group."""
    count = int(fine.max()) + 1
    minimum = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    maximum = np.full(count, -1, dtype=np.int64)
    np.minimum.at(minimum, fine, coarse)
    np.maximum.at(maximum, fine, coarse)
    observed = maximum >= 0
    return bool(np.all(minimum[observed] == maximum[observed]))


def _components(
    first: np.ndarray, second: np.ndarray, n_first: int, n_second: int
) -> int:
    """Connected components in the observed bipartite incidence graph."""
    graph = coo_matrix(
        (np.ones(len(first), dtype=bool), (first, n_first + second)),
        shape=(n_first + n_second, n_first + n_second),
    ).tocsr()
    return int(connected_components(graph, directed=False, return_labels=False))


def effect_rank(
    codes: np.ndarray, counts: np.ndarray, *, mode: str = "exact"
) -> EffectRank:
    """Rank of the full categorical dummy span (including its intercept).

    For general 3+ dimensions, exact mode uses a bounded dense SVD. Conservative
    mode uses pairwise connectivity to bound rank from above; it never labels
    that bound exact. Nested/duplicate dimensions are removed first.
    """
    n, dimensions = codes.shape
    if dimensions == 0:
        return EffectRank(0, True, "none")
    if dimensions == 1:
        return EffectRank(int(counts[0]), True, "category_count")
    # Two-way connectivity already accounts for nesting and duplicates.
    if dimensions == 2:
        components = _components(
            codes[:, 0], codes[:, 1], int(counts[0]), int(counts[1])
        )
        return EffectRank(int(counts.sum()) - components, True, "bipartite_components")

    retained: list[int] = []
    for candidate in range(dimensions):
        if any(is_nested(codes[:, index], codes[:, candidate]) for index in retained):
            continue
        retained = [
            index
            for index in retained
            if not is_nested(codes[:, candidate], codes[:, index])
        ]
        retained.append(candidate)
    if len(retained) < dimensions:
        result = effect_rank(codes[:, retained], counts[retained], mode=mode)
        return EffectRank(
            result.rank, result.exact, "nested_reduction+" + result.method
        )

    if mode == "conservative":
        rank = int(counts[0])
        for right in range(1, dimensions):
            redundancy = max(
                _components(
                    codes[:, left],
                    codes[:, right],
                    int(counts[left]),
                    int(counts[right]),
                )
                for left in range(right)
            )
            rank += int(counts[right]) - redundancy
        return EffectRank(min(n, rank), False, "pairwise_upper_bound")

    columns = int(counts.sum())
    if n * columns * 8 > _MAX_DUMMY_BYTES or min(n, columns) > _MAX_DUMMY_MIN_DIMENSION:
        raise ValueError(
            "exact rank of these 3+ fixed effects exceeds the numerical-rank budget; "
            "simplify redundant effects or explicitly use fe_dof='conservative' "
            "for a reported upper bound on absorbed rank"
        )
    dummy = np.zeros((n, columns), dtype=np.float64)
    rows = np.arange(n)
    offset = 0
    for dimension, count in enumerate(counts):
        dummy[rows, offset + codes[:, dimension]] = 1.0
        offset += int(count)
    return EffectRank(int(np.linalg.matrix_rank(dummy)), True, "dummy_svd")


def cluster_parameter_count(
    n_features: int,
    codes: np.ndarray,
    counts: np.ndarray,
    cluster_codes: list[np.ndarray],
    *,
    policy: str,
    full_rank: EffectRank,
    mode: str,
) -> tuple[int, tuple[int, ...], EffectRank]:
    """Choose correction K separately from the rank of the fitted model.

    With absorbed effects, n_features excludes the intercept. Nonnested mode
    retains an intercept even if every effect is excluded from correction K.
    An effect is excluded only if its entire partition nests in a requested
    primary cluster term (interaction terms count as single partitions).
    """
    if codes.shape[1] == 0:
        return n_features, (), EffectRank(0, True, "none")
    if policy == "full":
        return n_features + full_rank.rank, (), full_rank
    if policy == "none":
        return n_features, (), EffectRank(0, True, "excluded_by_policy")
    nested = tuple(
        dimension
        for dimension in range(codes.shape[1])
        if any(is_nested(codes[:, dimension], cluster) for cluster in cluster_codes)
    )
    retained = [
        dimension for dimension in range(codes.shape[1]) if dimension not in nested
    ]
    if not retained:
        correction_rank = EffectRank(1, True, "intercept_only")
    elif not nested:
        correction_rank = full_rank
    else:
        correction_rank = effect_rank(codes[:, retained], counts[retained], mode=mode)
    return n_features + correction_rank.rank, nested, correction_rank
