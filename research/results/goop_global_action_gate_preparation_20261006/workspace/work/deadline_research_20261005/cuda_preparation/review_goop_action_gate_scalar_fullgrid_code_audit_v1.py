#!/usr/bin/env python3
"""One full120-cell inert driver/NPZ/core grid plus narrow admission variants.

The separately checked original-protocol validator is stubbed because every
source/model input here is synthetic. Numerical validation is never stubbed.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location("_independent_fullgrid_fixtures", HERE / "test_summarize_goop_action_gate_v1.py")
    w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
    m, t = w.M, w.T
    sources = {name: sha(HERE / name) for name in ("goop_action_gate_scalar_core_v1.py", "summarize_goop_action_gate_v1.py",
               "test_goop_action_gate_scalar_core_v1.py", "test_summarize_goop_action_gate_v1.py")}
    assert sources["goop_action_gate_scalar_core_v1.py"] == "a0c0098e72c073b3468a2cc9a0d7a23c723250148f1dc1a550f35e02425081b1"
    assert sources["summarize_goop_action_gate_v1.py"] == "2f1f5fa7085be29d573aadd098eedb764259d06dde79308526052127966eb85f"
    output = HERE / "goop_action_gate_scalar_fullgrid_fixture_code_audit_v1"
    output.mkdir(exist_ok=False)
    with pytest.MonkeyPatch.context() as mp:
        fixture = w.stage_fixture(output, mp, policy="base", collection=False)
        root, directory, entry, outcome, context, bindings, mods = fixture
        protocol = json.loads((directory / "protocol.json").read_bytes())
        protocol_pin = sha(directory / "protocol.json")
        committed, cases = [], []
        original_rollout = t.D.rollout_row
        for source in range(30):
            for policy in m.POLICIES:
                name = f"source_{source:06d}_{policy}"
                started = time.perf_counter()
                if source == 0 and policy == "base":
                    row = json.loads((directory / (name + ".json")).read_bytes())
                    record = {**row, "row_file": name + ".json", "row_sha256": sha(directory / (name + ".json"))}
                    wall = 123.0  # Explicit synthetic timing of the reusable starter fixture.
                else:
                    def source_rollout(*args, **kwargs):
                        args = list(args); args[6] = {**args[6], "source_index": source}
                        return original_rollout(*args, **kwargs)
                    t.D.rollout_row = source_rollout
                    try:
                        row, arrays, _, _, _, _ = t.numeric_fixture(policy)
                    finally:
                        t.D.rollout_row = original_rollout
                    row.update(source_index=source, protocol_sha256=protocol_pin)
                    record = t.D.publish_row(directory, name, row, arrays)
                    wall = time.perf_counter() - started
                committed.append(record)
                cases.append({"source_index": source, "policy": policy, "wall_seconds": wall})
        assert len(committed) == 120
        collection = {"schema": t.D.ROLLOUT_COLLECTION, "mode": "test-rollout", "split": "test",
                      "driver_protocol_sha256": protocol_pin, "model": protocol["model"], "expected_cells": m.EXPECTED,
                      "required_rows": 120, "committed_rows": 120, "complete_rows": 120,
                      "all_inputs_reverified": True, "model_state_verified_unchanged": True,
                      "status": "complete", "all_required_outcomes_complete": True,
                      "abort_reason": None, "current_at_stop": None, "rows": committed,
                      "coverage": t.D.collection_coverage(m.EXPECTED, committed), "case_timings": cases,
                      "runtime": {"synthetic_inert_grid": True, "setup_seconds": 0.0}}
        outcome.update(stopped_and_reaped=True, exit_code=0)

        def publish(value):
            status = {"schema": t.D.SCHEMA, "state": value["status"], "current": value["current_at_stop"],
                      **{k: value[k] for k in ("required_rows", "committed_rows", "complete_rows", "all_inputs_reverified",
                         "model_state_verified_unchanged", "abort_reason")}}
            w.write(directory / "status.json", status)
            tree = m.tree_snapshot(directory)
            tree["files"].pop("rollout_collection.json", None)
            tree["output_tree_entries"] = [v for v in tree["output_tree_entries"] if v["path"] != "rollout_collection.json"]
            value.update(tree)
            w.write(directory / "rollout_collection.json", value)

        def summarize(value):
            other = [{"seed": seed, "scientific_verification_passed": False, "rows": [],
                      "cells": [{**v, "state": "never_started", "failure": None} for v in m.EXPECTED],
                      "outcome": {"started": False}, "runtime": None} for seed in (1, 2)]
            return mods.V.summarize([value] + other, mods.P)

        publish(collection)
        baseline_status = (directory / "status.json").read_bytes()
        baseline_collection = (directory / "rollout_collection.json").read_bytes()
        proof = {"path": str(directory / "rollout_collection.json"), "sha256": sha(directory / "rollout_collection.json"),
                 "status": "complete", "required_rows": 120, "committed_rows": 120}
        outcome["complete_collection"] = proof
        results = []
        for case in ("complete_original120", "missing_original_proof", "changed_original_proof", "model_audit_failed"):
            (directory / "status.json").write_bytes(baseline_status)
            (directory / "rollout_collection.json").write_bytes(baseline_collection)
            outcome.update(exit_code=0, complete_collection=copy.deepcopy(proof))
            if case == "missing_original_proof": outcome.pop("complete_collection")
            elif case == "changed_original_proof": outcome["complete_collection"]["sha256"] = "0" * 64
            elif case == "model_audit_failed":
                outcome.update(exit_code=1); outcome.pop("complete_collection")
                changed = {**collection, "model_state_verified_unchanged": False, "status": "stopped_incomplete_or_failed",
                           "all_required_outcomes_complete": False,
                           "abort_reason": {"category": "execution_error", "error": "synthetic final model-state mismatch"}}
                publish(changed)
            stage = w.stage(fixture)
            assert len(stage["rows"]) == 120 and not stage["evidence"]["invalid_records"]
            assert all(v["validated"]["complete"] for v in stage["rows"])
            verified = case == "complete_original120"
            assert stage["scientific_verification_passed"] is verified
            summary = summarize(stage)
            for metric in mods.V.METRICS + ("end_to_end_case_wall_seconds",):
                for policy in m.POLICIES:
                    value = summary["absolute"][metric][policy]["seed_values"]["0"]
                    assert (value is not None) is verified
            assert summary["runtime"][0]["per_policy"]["base"]["consumed_committed_work_totals"]["deployed_forward_attempts"] == 30 * 395
            results.append({"case": case, "passed": True, "scientific_verification_passed": verified,
                            "numerically_complete_retained_rows": 120, "evidence": stage["evidence"]["scientific_verification"],
                            "collection_sha256": sha(directory / "rollout_collection.json"),
                            "source0_base_H395_seed_mean": summary["absolute"]["mean_rollout_mse"]["base"]["seed_values"]["0"]})
        (directory / "status.json").write_bytes(baseline_status)
        (directory / "rollout_collection.json").write_bytes(baseline_collection)
        for name, pin in sources.items(): assert sha(HERE / name) == pin
        receipt = {"schema": "adaptgns_goop_action_gate_scalar_fullgrid_independent_code_audit_v1", "status": "passed",
                   "source_sha256": sources, "probe_source_sha256": sha(Path(__file__)), "inert_synthetic_only": True,
                   "complete_byte_validated_rows": 120, "accepted_forecasts": 120 * 395, "policies": list(m.POLICIES),
                   "numeric_validation_stubbed": False, "protocol_validation_stubbed_for_synthetic_lineage": True,
                   "original_protocol_and_driver_context_separately_tested": True,
                   "all_original_numeric_artifacts_retained": True, "passed": len(results), "cases": results,
                   "baseline_collection_sha256": sha(directory / "rollout_collection.json")}
        target = HERE / "goop_action_gate_scalar_fullgrid_independent_code_audit_v1.json"
        target.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"path": str(target), "sha256": sha(target), "passed": len(results), "rows": 120}, indent=2))


if __name__ == "__main__":
    main()
