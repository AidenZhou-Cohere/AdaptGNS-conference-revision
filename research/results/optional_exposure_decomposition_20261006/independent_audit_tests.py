"""Pure arithmetic audit self-checks; no scientific files are opened."""
import copy
import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("independent_exposure_audit", Path(__file__).with_name("audit_optional_exposure_decomposition.py"))
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def fixture(zero_degrees=False):
    r = np.tile([1., 2.], (8, 1))
    base_error = np.einsum("ij,ij->i", r, r) / 2
    degrees = [np.array([0, 2, 0, 1, 1, 2, 1, 1]), np.array([0, 0, 2, 1, 2, 1, 1, 1]), np.ones(8, dtype=int)]
    q = {}
    for i, policy in enumerate(module.SPARSE):
        delta = np.arange(16).reshape(8, 2).astype(float) / (40 + i)
        q[policy] = {"error": np.einsum("ij,ij->i", delta-r, delta-r) / 2,
            "alignment": np.einsum("ij,ij->i", r, delta),
            "cost": np.einsum("ij,ij->i", delta, delta) / 2,
            "degree": np.zeros(8, dtype=int) if zero_degrees else degrees[i]}
    return q, base_error


def test_independent_coordinate_partition_identity_and_empty_group_rules():
    q, base = fixture(); a = module.Audit()
    expected = module.core(q, base, {}, a, "synthetic")
    module.partition(expected["metrics"], 8, a, "synthetic")
    for p in module.PAIRS:
        groups = expected["pairs"][p]["groups"]
        assert sum(groups[g]["particles"] for g in module.PRIMARY) == 8
        assert sum(groups[g]["particles"] for g in module.BOTH) == groups["both"]["particles"]
        assert expected["metrics"][p + "/whole/degree_difference"] == 0.
    q, base = fixture(True); empty = module.core(q, base, {}, a, "empty")
    for p in module.PAIRS:
        assert empty["metrics"][p + "/neither/particle_fraction"] == 1.
        for g in module.GROUPS[1:]:
            assert empty["metrics"][f"{p}/{g}/particle_fraction"] == 0.
            for quantity in module.Q:
                assert empty["metrics"][f"{p}/{g}/{quantity}_contribution"] == 0.
                key = f"{p}/{g}/conditional_mean_{quantity}_difference"
                assert empty["metrics"][key] is None and empty["undefined_reasons"][key] == "empty_group"


def test_conditional_nulls_and_weighted_ratios_do_not_drop_empty_frames():
    a = module.Audit(); q, base = fixture(); q0, b0 = fixture(True)
    full, empty = module.core(q, base, {}, a, "full"), module.core(q0, b0, {}, a, "empty")
    result = module.aggregate({(3, 7): full, (3, 106): empty}, sources=(3,), targets=(7, 106))
    p, g = "speed_minus_random", "left_only"
    assert full["metrics"][f"{p}/{g}/particle_fraction"] > 0
    key = f"{p}/{g}/conditional_mean_error_difference"
    assert result["metrics"][key] is None
    assert result["metric_coverage"][key]["defined_frames"] == 1
    assert result["metric_coverage"][key]["undefined_reason_counts"] == {"empty_group": 1}
    ratio = result["weighted_conditionals"][f"{p}/{g}/error"]
    assert ratio["reason"] is None and ratio["value"] == ratio["weighted_contribution"] / ratio["weighted_particle_fraction"]


def test_missing_action_propagates_only_to_required_comparisons_and_all_required_aggregates():
    a = module.Audit(); q, base = fixture()
    full = module.core(q, base, {}, a, "full")
    del q[module.POLICIES[-1]]
    partial = module.core(q, base, {}, a, "partial")
    assert partial["status"] == "partial"
    assert partial["metrics"]["risk_minus_random/whole/error_difference"] is None
    assert partial["metrics"]["speed_minus_random/whole/error_difference"] is not None
    combined = module.aggregate({(3, 7): full, (3, 106): partial}, sources=(3,), targets=(7, 106))
    assert combined["metrics"]["risk_minus_random/whole/error_difference"] is None
    assert combined["weighted_conditionals"]["risk_minus_random/both/error"]["reason"] == "missing_required_input"
    assert module.mean([1., None, 3.]) is None
