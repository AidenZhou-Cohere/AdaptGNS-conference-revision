"""Separate exact-semantics CPU bookkeeping candidate; not scientifically admitted.

Derived from goop3d_graph_support.py SHA256
7d43fe7d06ac450b6b9031181902cfc6d3d9754af3a26e17e96fd46c4a1bf64f.
Only canonical pair sorting/difference and uncapped degree accounting change.
Native/model/RNG/edge ordering and all numeric operations remain unchanged.
No frozen source is overwritten; CPU equivalence is not a CUDA numerical gate.
"""
from dataclasses import dataclass
import hashlib
import math
import numpy as np
from scipy.spatial import cKDTree
import torch

RADIUS_FACTOR = 1.267
MAX_PAIRS = 2_000_000
MAX_BATCH_EDGES = 5_000_000
NOISE = 6.7e-4
KINEMATIC = 3


@dataclass
class Candidates:
    base: np.ndarray
    extra: np.ndarray
    n_nodes: int


def require(condition, message):
    if not condition:
        raise ValueError(message)


def state_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def canonical_pairs(pairs):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs):
        pairs = np.sort(pairs, axis=1)
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        unique = np.r_[True, np.any(pairs[1:] != pairs[:-1], axis=1)]
        pairs = pairs[unique]
    return np.ascontiguousarray(pairs)


def pair_set(pairs):
    return {tuple(pair) for pair in pairs}


def check_candidate_bound(points, radius):
    points = np.asarray(points, dtype=np.float32)
    require(points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()
            and math.isfinite(radius) and radius > 0, "Finite 3D points and positive radius required")
    tree = cKDTree(points)
    count = int((tree.count_neighbors(tree, radius * RADIUS_FACTOR) - len(points)) // 2)
    require(count <= MAX_PAIRS, f"3D benchmark candidate_pair_resource_guard: {count} > {MAX_PAIRS}")
    return tree


def strict_pairs(position, radius):
    """Native float32 norm and strict radius, symmetric pair universe."""
    points = np.asarray(position, dtype=np.float32)
    tree = check_candidate_bound(points, radius)
    # Query matches native cKDTree candidates; its subsequent norm is float32.
    def query(r):
        found = canonical_pairs(tree.query_pairs(r, output_type="ndarray"))
        distance = np.linalg.norm(points[found[:, 0]] - points[found[:, 1]], axis=1)
        return found[distance < r]
    base, expanded = query(radius), query(radius * RADIUS_FACTOR)
    extra = sorted_pair_difference(base, expanded, len(points))
    return Candidates(base, extra, len(points))


def sorted_pair_difference(base, expanded, n_nodes):
    """Exact lexicographically sorted expanded-minus-base; reject nonnested sets.

    Internal inputs are canonical sorted unique pairs from the two queries.
    Int64 keys preserve their order where multiplication is safe. A structured
    lexicographic fallback preserves semantics for the otherwise infeasible
    very large node IDs, avoiding overflow and any new graph-size restriction.
    """
    if not len(base):
        return np.ascontiguousarray(expanded)
    if n_nodes <= np.iinfo(np.int64).max // max(n_nodes, 1):
        base_keys = base[:, 0] * n_nodes + base[:, 1]
        expanded_keys = expanded[:, 0] * n_nodes + expanded[:, 1]
    else:
        pair_dtype = np.dtype([("source", np.int64), ("target", np.int64)])
        base_keys = np.ascontiguousarray(base).view(pair_dtype).reshape(-1)
        expanded_keys = np.ascontiguousarray(expanded).view(pair_dtype).reshape(-1)
    index = np.searchsorted(expanded_keys, base_keys)
    if np.any(index >= len(expanded_keys)) or not np.array_equal(expanded_keys[index], base_keys):
        raise RuntimeError("Strict radius pair sets are unexpectedly nonnested")
    keep = np.ones(len(expanded), dtype=bool)
    keep[index] = False
    return np.ascontiguousarray(expanded[keep])


def ordered_edges(position, pairs, loops, cap=None):
    """Receiver-major, distance then source-ID order; exactly one self edge.

    Cap is only used for base/dense factorial cases. It includes the self edge,
    exactly as native training does, and may create asymmetric directed edges.
    No policy applies a post-selection cap.
    """
    points = np.asarray(position, dtype=np.float32)
    require(points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all(), "Finite 3D points required")
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    n = len(points)
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError("Cap must be a positive integer or None")
    if len(pairs) and (np.any(pairs[:, 0] >= pairs[:, 1]) or np.any(pairs < 0) or np.any(pairs >= n)
                       or len(np.unique(pairs, axis=0)) != len(pairs)):
        raise ValueError("Unique canonical non-self pairs required")
    edges = np.concatenate((pairs.T, pairs[:, ::-1].T), axis=1)
    if loops:
        ids = np.arange(n, dtype=np.int64)
        edges = np.concatenate((edges, np.stack((ids, ids))), axis=1)
    distances = np.linalg.norm(points[edges[0]] - points[edges[1]], axis=1)
    order = np.lexsort((edges[0], distances, edges[1]))
    edges = edges[:, order]
    if cap is not None and edges.shape[1]:
        _, first = np.unique(edges[1], return_index=True)
        within_receiver = np.arange(edges.shape[1]) - np.repeat(first, np.diff(np.r_[first, edges.shape[1]]))
        edges = edges[:, within_receiver < cap]
    return np.ascontiguousarray(edges)


def graph_rng(seed, absolute_step, example_slot):
    require(seed >= 0 and absolute_step >= 0 and example_slot >= 0, "Nonnegative RNG identity required")
    coin_material = [20261005, seed, absolute_step, example_slot, 4409]
    pair_material = [20261005, seed, absolute_step, example_slot, 5501]
    coin = bool(np.random.default_rng(np.random.SeedSequence(coin_material)).random() < .5)
    rng = np.random.default_rng(np.random.SeedSequence(pair_material))
    return coin, rng, coin_material, pair_material


def append_optional_edges(noisy_sequence, counts, native_edges, radius, seed, absolute_step, arm):
    """Native batch graph prefix plus exact symmetric optional suffix.

    Mandatory edges are obtained from the native model, not reconstructed or
    symmetrized. The independent per-example graph RNG never touches torch.
    """
    require(arm in ("base", "mix", "expanded25"), "Unknown 3D graph arm")
    counts = torch.as_tensor(counts).detach().cpu().numpy().astype(np.int64)
    points = noisy_sequence[:, -1].detach().cpu().numpy().astype(np.float32, copy=False)
    native = native_edges.detach().cpu().numpy()
    require(counts.ndim == 1 and np.all(counts > 0) and int(counts.sum()) == len(points), "Invalid batch partition")
    require(native.shape[0] == 2 and native.dtype.kind in "iu", "Invalid native edges")
    offsets = np.r_[0, np.cumsum(counts)]
    require(np.all(native >= 0) and np.all(native < len(points)), "Native edge index out of range")
    membership = np.repeat(np.arange(len(counts)), counts)
    require(np.all(membership[native[0]] == membership[native[1]]), "Native cross-example edge")
    additions, ledger = [], []
    for slot, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        start, end = int(start), int(end)
        local = points[start:end]
        graph = strict_pairs(local, radius)
        coin, rng, coin_material, pair_material = graph_rng(seed, absolute_step, slot)
        budget = math.floor(.25 * len(graph.extra))
        active = arm == "expanded25" or (arm == "mix" and coin)
        optional = graph.extra[rng.permutation(len(graph.extra))[:budget]] if active else np.empty((0, 2), dtype=np.int64)
        optional = canonical_pairs(optional)
        extra_edges = ordered_edges(local, optional, False)
        require(extra_edges.shape[1] == 2 * (budget if active else 0), "Exact 2B optional count violated")
        require(not np.any(extra_edges[0] == extra_edges[1]), "Optional self edge")
        mask = (native[1] >= start) & (native[1] < end)
        local_native = native[:, mask] - start
        expected_native = ordered_edges(local, graph.base, True, 128)
        require(np.array_equal(local_native, expected_native), "Native graph differs from strict/capped/self convention")
        if extra_edges.size:
            additions.append(extra_edges + start)
        # Every canonical undirected pair contributes one receiver degree at
        # each endpoint; native self candidates contribute one at every node.
        # This exactly counts the uncapped graph without sorting its edges.
        full_degree = np.bincount(graph.base.reshape(-1), minlength=len(local)) + 1
        ledger.append({"example_slot": slot, "n_particles": len(local), "exposure_coin": coin,
            "expanded": active, "coin_seed_material": coin_material, "pair_seed_material": pair_material,
            "native_directed_edges": int(local_native.shape[1]), "native_self_edges": int(np.sum(local_native[0] == local_native[1])),
            "receivers_above_native_cap": int(np.sum(full_degree > 128)), "annulus_pairs": len(graph.extra),
            "optional_budget_if_exposed": budget, "selected_optional_pairs": len(optional),
            "native_edge_sha256": state_hash(local_native),
            "optional_pair_sha256": state_hash(optional), "noisy_current_sha256": state_hash(local)})
    all_edges = np.concatenate((native, *additions), axis=1) if additions else native
    require(np.array_equal(all_edges[:, :native.shape[1]], native), "Native mandatory prefix changed")
    require(all_edges.shape[1] == native.shape[1] + 2 * sum(row["selected_optional_pairs"] for row in ledger), "Batch optional budget differs")
    require(np.all(membership[all_edges[0]] == membership[all_edges[1]]), "Augmented cross-example edge")
    require(all_edges.shape[1] <= MAX_BATCH_EDGES, "3D benchmark batch edge resource guard")
    # Returning the original tensor in the no-addition case also retains its
    # exact edge order and avoids a needless device round trip.
    edges = native_edges if not additions else torch.as_tensor(all_edges, dtype=torch.long, device=native_edges.device)
    return edges, ledger


def forward_batch(model, batch, noise, device, seed, absolute_step, arm):
    """Direct normalized output; never decode positions then invert them."""
    position, types, counts, labels = batch
    require(position.shape[1:] == (6, 3) and labels.shape == (len(position), 3), "Six-frame 3D history and 3D labels required")
    require(float(model._connectivity_radius) == .025, "Goop-3D radius must be .025")
    position, noise = position.to(device), noise.to(device)
    types, counts, labels = types.to(device), counts.to(device), labels.to(device)
    noisy = position + noise
    # Check the host pair bound before native query_ball_point materializes its
    # neighbor lists. This is a 3D benchmark guard, not the old 2D guard value.
    local_counts = counts.detach().cpu().tolist()
    require(all(value > 0 for value in local_counts) and sum(local_counts) == len(noisy), "Invalid batch partition")
    start = 0
    for count in local_counts:
        check_candidate_bound(noisy[start:start + count, -1].detach().cpu().numpy(), float(model._connectivity_radius))
        start += count
    nodes, native_edges, native_features = model._encoder_preprocessor(
        noisy, counts, types, None, augment_radius_prob=0., augment_radius_factor=1.267)
    edges, ledger = append_optional_edges(noisy, counts, native_edges, float(model._connectivity_radius), seed, absolute_step, arm)
    if edges is native_edges:
        features = native_features
    else:
        current = noisy[:, -1]
        displacements = (current[edges[0]] - current[edges[1]]) / model._connectivity_radius
        features = torch.cat((displacements, torch.norm(displacements, dim=-1, keepdim=True)), dim=-1)
    prediction, head = model._encode_process_decode(nodes, edges, features)
    target = model._inverse_decoder_postprocessor(labels + noise[:, -1], noisy)
    return prediction, head, target, ledger


def host_noise(shape, particle_types, seed, step, noise_std=NOISE):
    """Original twice-integrated random-walk noise, generated only on CPU."""
    n, history, dim = shape
    if history < 2 or n != len(particle_types) or noise_std < 0:
        raise ValueError("Invalid history/noise configuration")
    generator = torch.Generator(device="cpu")
    state = np.random.SeedSequence([seed, step, 2207]).generate_state(1, dtype=np.uint64)
    generator.manual_seed(int(state[0]) % (2**63-1))
    increments = torch.randn((n, history-1, dim), generator=generator, dtype=torch.float32)
    increments *= noise_std/math.sqrt(history-1)
    velocities = increments.cumsum(1)
    noise = torch.cat((torch.zeros((n,1,dim)), velocities.cumsum(1)), 1)
    noise *= (torch.as_tensor(particle_types).cpu() != KINEMATIC).view(-1,1,1)
    return noise


def normalized_statistics(prediction, head, target, mask):
    """D3 normalized-acceleration diagnostics, never calibration/rollout claims."""
    require(prediction.ndim == 2 and prediction.shape[1] == 3 and target.shape == prediction.shape,
            "Three-dimensional acceleration required")
    require(head.shape == (len(prediction),) and mask.shape == (len(prediction),) and bool(mask.any()),
            "One scalar variance and mask per particle required")
    residual = prediction[mask] - target[mask]
    squared = residual.square().sum(-1)
    q = head[mask].clamp_min(1e-6)
    return {"coordinate_mse": float((squared / 3).mean().detach().cpu()),
            "vector_mse": float(squared.mean().detach().cpu()),
            "mean_coordinate_variance_q": float(q.mean().detach().cpu()),
            "mean_vector_risk_3q": float((3 * q).mean().detach().cpu()),
            "constant_free_gaussian_nll": float((squared / (2 * q) + 1.5 * q.log()).mean().detach().cpu())}
