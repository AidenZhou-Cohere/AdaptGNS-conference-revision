"""Exact reference comparisons on synthetic CPU inputs; no official/test data."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from test_train_goop3d_graph_support_cuda_v1 import synthetic, entry

ROOT = Path(__file__).resolve().parent
REFERENCE_SHA = "7d43fe7d06ac450b6b9031181902cfc6d3d9754af3a26e17e96fd46c4a1bf64f"


def load(name):
    path = ROOT / (name + ".py")
    spec = importlib.util.spec_from_file_location("_exact_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reference = load("goop3d_graph_support")
candidate = load("goop3d_graph_support_vectorized_v1")


def test_reference_pin_and_unchanged_numerical_rng_native_ordering_functions():
    assert hashlib.sha256((ROOT / "goop3d_graph_support.py").read_bytes()).hexdigest() == REFERENCE_SHA
    def functions(path):
        return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    left = functions(ROOT / "goop3d_graph_support.py")
    right = functions(ROOT / "goop3d_graph_support_vectorized_v1.py")
    assert right.keys() - left.keys() == {"sorted_pair_difference"}
    for name in left.keys() - {"canonical_pairs", "strict_pairs", "append_optional_edges"}:
        assert ast.dump(left[name]) == ast.dump(right[name]), name
    # The optional-graph procedure changes only uncapped degree bookkeeping.
    old = "np.bincount(ordered_edges(local, graph.base, True)[1], minlength=len(local))"
    new = "np.bincount(graph.base.reshape(-1), minlength=len(local)) + 1"
    expected = ast.parse(ast.unparse(left["append_optional_edges"]).replace(old, new)).body[0]
    assert ast.dump(expected) == ast.dump(right["append_optional_edges"])


@pytest.mark.parametrize("pairs", [[], [[2, 1], [0, 4], [1, 2], [4, 0]],
    [[-5, 2], [3, -6], [-5, 2], [1, 1]],
    [[np.iinfo(np.int64).max, 0], [np.iinfo(np.int64).min, 5], [0, np.iinfo(np.int64).max]]])
def test_canonical_pair_order_dtype_shape_and_hash_are_identical(pairs):
    a, b = reference.canonical_pairs(pairs), candidate.canonical_pairs(pairs)
    assert reference.state_hash(a) == candidate.state_hash(b)
    assert np.array_equal(a, b) and a.dtype == b.dtype and a.shape == b.shape


def test_pair_difference_rejects_nonnested_and_preserves_order_with_overflow_fallback():
    for n in (5, 3_037_000_501):
        base = np.array([[0, 3], [2, n-1]], dtype=np.int64)
        expanded = np.array([[0, 2], [0, 3], [1, 2], [2, n-1]], dtype=np.int64)
        assert np.array_equal(candidate.sorted_pair_difference(base, expanded, n), expanded[[0, 2]])
        for bad in (np.array([[0, 4]], dtype=np.int64), np.array([[n-2, n-1]], dtype=np.int64)):
            with pytest.raises(RuntimeError, match="nonnested"):
                candidate.sorted_pair_difference(bad, expanded, n)
    empty = np.empty((0, 2), dtype=np.int64)
    assert np.array_equal(candidate.sorted_pair_difference(empty, expanded, n), expanded)
    with pytest.raises(RuntimeError, match="nonnested"):
        candidate.sorted_pair_difference(base, empty, n)


def clouds():
    radius = np.float32(.025)
    outer = np.float32(.025 * 1.267)
    boundary = np.array([[0, 0, 0]] + [[float(value), 0, 0] for value in
        (np.nextafter(radius, -np.inf, dtype=np.float32), radius,
         np.nextafter(radius, np.inf, dtype=np.float32),
         np.nextafter(outer, -np.inf, dtype=np.float32), outer,
         np.nextafter(outer, np.inf, dtype=np.float32))], dtype=np.float32)
    rng = np.random.default_rng(82450)
    return [np.zeros((1, 3), dtype=np.float32), np.zeros((130, 3), dtype=np.float32), boundary,
            np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
            rng.uniform(.2, .3, size=(127, 3)).astype(np.float32),
            rng.uniform(.2, .22, size=(140, 3)).astype(np.float32),
            rng.normal(size=(213, 3)).astype(np.float32)*.025]


@pytest.mark.parametrize("points", clouds())
def test_candidate_pair_sets_and_all_ordered_edge_hashes_are_exact(points):
    before = reference.state_hash(points)
    a, b = reference.strict_pairs(points, .025), candidate.strict_pairs(points, .025)
    assert a.n_nodes == b.n_nodes == len(points)
    for first, second in ((a.base, b.base), (a.extra, b.extra)):
        assert reference.state_hash(first) == candidate.state_hash(second)
        for loops, cap in ((False, None), (True, None), (True, 128), (True, 1)):
            left = reference.ordered_edges(points, first, loops, cap)
            right = candidate.ordered_edges(points, second, loops, cap)
            assert reference.state_hash(left) == candidate.state_hash(right)
    assert reference.state_hash(points) == before


@pytest.mark.parametrize("arm", ["base", "mix", "expanded25"])
@pytest.mark.parametrize("seed,step", [(0, 0), (1, 1), (2, 64), (0, 99999)])
def test_full_batch_prefix_suffix_rng_and_all_ledger_bytes_are_exact(arm, seed, step, monkeypatch):
    for local in clouds():
        points = np.concatenate((local, local))
        counts = torch.tensor([len(local), len(local)])
        noisy = torch.from_numpy(np.repeat(points[:, None, :], 6, axis=1))
        base = reference.strict_pairs(local, .025).base
        edges = reference.ordered_edges(local, base, True, 128)
        native = torch.from_numpy(np.concatenate((edges, edges + len(local)), axis=1))
        rng_before = torch.get_rng_state().clone()
        with monkeypatch.context() as m:
            m.setattr(torch.cuda, "_lazy_init", lambda: (_ for _ in ()).throw(AssertionError("CUDA forbidden")))
            left, ledger_a = reference.append_optional_edges(noisy, counts, native, .025, seed, step, arm)
            right, ledger_b = candidate.append_optional_edges(noisy, counts, native, .025, seed, step, arm)
        assert torch.equal(left, right) and ledger_a == ledger_b
        assert torch.equal(right[:, :native.shape[1]], native)
        assert torch.equal(rng_before, torch.get_rng_state())
        if all(row["selected_optional_pairs"] == 0 for row in ledger_b):
            assert left is native and right is native


@pytest.mark.parametrize("arm", ["base", "mix", "expanded25"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_tiny_cpu_updates_preserve_outputs_gradients_adam_and_ledger(synthetic, arm, seed):
    s = synthetic
    left, left_opt = s.make(seed)
    right, right_opt = s.make(seed)
    for step in range(2):
        noise = s.helpers.host_noise(s.batch[0].shape, s.batch[1], seed, step)
        old = reference.forward_batch(left, s.batch, noise, "cpu", seed, step, arm)
        new = candidate.forward_batch(right, s.batch, noise, "cpu", seed, step, arm)
        assert all(torch.equal(a, b) for a, b in zip(old[:3], new[:3])) and old[3] == new[3]
        losses = []
        for model, optimizer, output in ((left, left_opt, old), (right, right_opt, new)):
            optimizer.param_groups[0]["lr"] = entry.learning_rate(step, 37)
            optimizer.zero_grad(set_to_none=True)
            losses.append(entry.guarded_update(s.helpers, model, optimizer, *output[:3], s.batch[1] != 3,
                                              "faithful", step + 1, 37))
        assert torch.equal(*losses)
        assert entry.tree_equal(torch, left.state_dict(), right.state_dict())
        assert entry.tree_equal(torch, left_opt.state_dict(), right_opt.state_dict())
        assert all(torch.equal(a.grad, b.grad) for a, b in zip(left.parameters(), right.parameters()))
