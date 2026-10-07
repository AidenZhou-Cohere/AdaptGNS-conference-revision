"""Particle-simulation methods and data utilities."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Candidates:
    base: np.ndarray
    extra: np.ndarray
    n_nodes: int


def candidates(position, radius, radius_factor=1.267):
    position = np.asarray(position)
    if position.ndim != 2 or not np.isfinite(position).all():
        raise ValueError("positions must be a finite [N,D] array")
    if radius <= 0 or radius_factor < 1:
        raise ValueError("radius > 0 and radius_factor >= 1 required")
    pairs = cKDTree(position).query_pairs(radius * radius_factor, output_type="ndarray")
    pairs = pairs.reshape(-1, 2)
    if len(pairs):
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    d2 = ((position[pairs[:, 0]] - position[pairs[:, 1]]) ** 2).sum(1)
    base = d2 <= radius ** 2
    return Candidates(pairs[base], pairs[~base], len(position))


def select_pairs(graph, node_scores, extra_budget):
    """Top-K pair priority max(s_i,s_j), with lexicographic tie breaking.

    K counts pairs, not selected particles. The priority is a heuristic, not
    an estimate or guarantee of actual error reduction. Ties depend on IDs.
    """
    score = np.asarray(node_scores).reshape(-1)
    if len(score) != graph.n_nodes or not np.isfinite(score).all():
        raise ValueError("one finite score required per particle")
    if int(extra_budget) != extra_budget or extra_budget < 0:
        raise ValueError("extra_budget must be a nonnegative integer")
    k = min(int(extra_budget), len(graph.extra))
    values = np.maximum(score[graph.extra[:, 0]], score[graph.extra[:, 1]])
    order = np.argsort(-values, kind="stable")[:k]
    return np.concatenate((graph.base, graph.extra[order]), axis=0)


def random_pairs(graph, extra_budget, rng):
    k = min(int(extra_budget), len(graph.extra))
    return np.concatenate((graph.base, graph.extra[rng.permutation(len(graph.extra))[:k]]))


def directed(pairs):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return np.concatenate((pairs.T, pairs[:, ::-1].T), axis=1)
