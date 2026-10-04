import numpy as np
import pytest
from research.budget_graph import candidates, select_pairs, random_pairs, directed


def test_exact_budget_subset_symmetry_and_distances():
    x = np.random.default_rng(4).random((80, 2))
    graph = candidates(x, .12, 1.6)
    scores = np.arange(80)
    for k in [0, 1, 7, 10000]:
        pairs = select_pairs(graph, scores, k)
        assert len(pairs) == len(graph.base) + min(k, len(graph.extra))
        pairset = {tuple(p) for p in pairs}
        assert len(pairset) == len(pairs)
        assert {tuple(p) for p in graph.base} <= pairset
        assert np.all(np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1) <= .12 * 1.6 + 1e-12)
        edges = directed(pairs)
        assert edges.shape[1] == 2 * len(pairs)
        assert set(map(tuple, edges.T)) == set(map(tuple, edges[::-1].T))
        assert np.all(edges[0] != edges[1])


def test_empty_and_tied_scores_are_deterministic():
    g = candidates(np.array([[0., 0.], [10., 10.]]), .1)
    assert directed(select_pairs(g, np.ones(2), 10)).shape == (2, 0)
    g = candidates(np.random.default_rng(1).random((30, 2)), .1, 4)
    assert np.array_equal(select_pairs(g, np.ones(30), 5), select_pairs(g, np.ones(30), 5))
    assert len(random_pairs(g, 5, np.random.default_rng(1))) == len(g.base) + min(5, len(g.extra))


def test_invalid_input():
    with pytest.raises(ValueError):
        candidates(np.array([[np.nan, 0]]), .1)
    with pytest.raises(ValueError):
        candidates(np.zeros((4, 2)), .1, .8)
