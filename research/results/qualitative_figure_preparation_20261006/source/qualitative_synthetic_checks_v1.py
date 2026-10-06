#!/usr/bin/env python3
"""Inert synthetic fixtures and contract checks; no external data or inference."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import render_qualitative_v1 as renderer


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return bound(path)


def bound(path):
    return {"path": str(path.resolve()), "sha256": renderer.sha(path.read_bytes())}


def make_fixture(directory, study):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    dataset, dims, horizon, updates = renderer.STUDIES[study]
    count = 33
    bounds = np.array([[0., 1.]] * dims)
    manifest = {"dataset": dataset, "split": "test", "record_count": 4,
                "metadata": {"bounds": bounds.tolist()},
                "records": [{"source_index": i, "id": f"test:{i:06d}",
                             "positions": {"shape": [horizon + 6, n, dims]}}
                            for i, n in enumerate((41, 33, 20, 56))]}
    manifest_ref = write_json(directory / "synthetic_manifest.json", manifest)
    t = np.arange(count, dtype=np.float32)
    points = np.stack([.15 + .7 * ((t * (d + 3)) % count) / (count-1)
                       for d in range(dims)], axis=-1)
    initial = np.stack([points.copy() for _ in range(6)])
    all_steps = np.array([1, 10, 50, 200, horizon], dtype=np.int64)
    def truth(step):
        answer = points.copy()
        answer[:, 1] = .14 + .72 * (points[:, 1] - .15) / .7 * (1. - .55 * step/horizon)
        return answer.astype(np.float32)
    cells = []
    for number, (arm, policy, _) in enumerate(renderer.METHODS):
        if study == "Goop2D" and policy == "laggedrisk25":
            cells.append({"arm": arm, "policy": policy, "coverage_state": "timed_out_current",
                          "reason": "Synthetic invocation ended without a committed row"})
            continue
        model = {"arm": arm, "seed": 0, "objective": "faithful", "checkpoint_sha256": ("1" if arm == "base" else "2") * 64}
        protocol = {"schema": renderer.PROTOCOL_SCHEMAS[study], "purpose": "final_evaluation",
                    "mode": "full-rollout", "split": "test", "input_files_sha256":
                    {manifest_ref["path"]: manifest_ref["sha256"]}}
        if study == "Goop3D":
            protocol.update(**model, checkpoint_updates=updates)
        else:
            protocol["model"] = dict(model, completed_updates=updates)
        protocol_ref = write_json(directory / f"{arm}_{policy}_protocol.json", protocol)
        prefix = 173 if study == "Goop2D" and policy == "random25" else 0 if study == "Goop3D" and policy == "laggedrisk25" else horizon
        steps = all_steps[all_steps <= prefix]
        actual = np.stack([truth(s) for s in steps]) if len(steps) else np.empty((0, count, dims), dtype=np.float32)
        predicted = actual.copy()
        if len(steps):
            predicted[:, :, 0] += np.array(steps, dtype=np.float32)[:, None] / horizon * .018 * (number+1)
            if horizon in steps and number == 0:
                predicted[-1, 0, 0] = 1.16
                predicted[-1, 1, 1] = -.06
        npz = directory / f"{arm}_{policy}_trace.npz"
        np.savez_compressed(npz, forecast_steps=steps, predicted_positions=predicted,
                            ground_truth_positions=actual, initial_observed_positions=initial,
                            particle_types=np.full(count, 7, dtype=np.int64), bounds=bounds,
                            rejected_prediction=np.full((count, dims), np.nan, dtype=np.float32))
        row = {"source_index": 1, "trajectory_id": "test:000001", "particles": count,
               "arm": arm, "policy": policy, "training_seed": 0, "horizon": horizon, "objective": "faithful",
               "status": "complete" if prefix == horizon else "failed", "completed_steps": prefix,
               "failure": None if prefix == horizon else {"category": "synthetic_numerical_guard"},
               "protocol_sha256": protocol_ref["sha256"]}
        key = "artifact" if study == "Goop3D" else "trace"
        row.update({key + "_file": npz.name, key + "_sha256": bound(npz)["sha256"]})
        row_ref = write_json(directory / f"{arm}_{policy}_row.json", row)
        cells.append({"arm": arm, "policy": policy, "row": row_ref, "protocol": protocol_ref})
    request = {"schema": "qualitative-render-request-v1", "study": study, "synthetic": True,
               "selection_plan": bound(renderer.PLAN), "manifest": manifest_ref, "cells": cells}
    request_path = directory / "request.json"
    write_json(request_path, request)
    return request_path


class QualitativeChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic_renderer_check_", dir=renderer.HERE)
        self.path = make_fixture(Path(self.temporary.name) / "fixture", "Goop2D")

    def tearDown(self):
        self.temporary.cleanup()

    def mutate_row(self, function, index=0):
        request = json.loads(self.path.read_text())
        cell = request["cells"][index]
        row_path = Path(cell["row"]["path"])
        row = json.loads(row_path.read_text())
        function(row)
        cell["row"] = write_json(row_path, row)
        write_json(self.path, request)

    def test_lower_median_is_not_source_order_median(self):
        data = renderer.load_request(self.path)
        self.assertEqual((data["source"]["source_index"], data["count"]), (1, 33))

    def test_forecast_lookup_and_ground_truth(self):
        data = renderer.load_request(self.path)
        self.assertEqual(sorted(data["truth"]), [1, 10, 50, 200, 395])
        self.assertFalse(np.array_equal(data["truth"][200], data["truth"][50]))
        self.assertEqual(renderer.panel_state(data["panels"][1], 200)[0], "FAILED BEFORE FRAME")
        self.assertNotIn(200, data["panels"][1]["frames"])

    def test_committed_prefix_and_unreturned_unknown_remain_distinct(self):
        data = renderer.load_request(self.path)
        self.assertEqual(data["panels"][1]["prefix"], 173)
        self.assertEqual(renderer.panel_state(data["panels"][1], 1)[0], "SAVED PREFIX")
        self.assertIsNone(data["panels"][2]["prefix"])
        self.assertIn("unknown", renderer.panel_state(data["panels"][2], 1)[1])

    def test_three_dimensions_and_zero_prefix(self):
        path = make_fixture(Path(self.temporary.name) / "goop3d", "Goop3D")
        data = renderer.load_request(path)
        self.assertEqual((data["dims"], data["horizon"], data["updates"]), (3, 295, 25000))
        self.assertEqual(data["panels"][2]["prefix"], 0)
        self.assertEqual(data["panels"][2]["frames"], {})
        self.assertIn("prefix: 0", renderer.panel_state(data["panels"][2], 1)[1])

    def test_sand_endpoint(self):
        path = make_fixture(Path(self.temporary.name) / "sand", "Sand")
        data = renderer.load_request(path)
        self.assertEqual((data["horizon"], data["updates"]), (314, 100000))
        self.assertIn(314, data["truth"])

    def test_checksum_failure(self):
        request = json.loads(self.path.read_text())
        with Path(request["cells"][0]["row"]["path"]).open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            renderer.load_request(self.path)

    def test_wrong_source_rejected(self):
        self.mutate_row(lambda row: row.update(source_index=0))
        with self.assertRaisesRegex(ValueError, "example identity"):
            renderer.load_request(self.path)

    def test_wrong_seed_rejected(self):
        self.mutate_row(lambda row: row.update(training_seed=1))
        with self.assertRaisesRegex(ValueError, "example identity"):
            renderer.load_request(self.path)

    def test_wrong_objective_rejected(self):
        self.mutate_row(lambda row: row.update(objective="corrected_nll"))
        with self.assertRaisesRegex(ValueError, "Faithful final"):
            renderer.load_request(self.path)

    def test_different_mix_checkpoints_rejected(self):
        path = make_fixture(Path(self.temporary.name) / "goop3d", "Goop3D")
        request = json.loads(path.read_text())
        cell = request["cells"][2]
        protocol_path = Path(cell["protocol"]["path"])
        protocol = json.loads(protocol_path.read_text())
        protocol["checkpoint_sha256"] = "9" * 64
        cell["protocol"] = write_json(protocol_path, protocol)
        row_path = Path(cell["row"]["path"])
        row = json.loads(row_path.read_text())
        row["protocol_sha256"] = cell["protocol"]["sha256"]
        cell["row"] = write_json(row_path, row)
        write_json(path, request)
        with self.assertRaisesRegex(ValueError, "same checkpoint"):
            renderer.load_request(path)

    def test_same_state_control_mislabelling_rejected(self):
        request = json.loads(self.path.read_text())
        request["cells"][2]["policy"] = "previous-observed-base-risk25"
        write_json(self.path, request)
        with self.assertRaisesRegex(ValueError, "fixed order"):
            renderer.load_request(self.path)

    def test_wrong_training_endpoint_rejected(self):
        request = json.loads(self.path.read_text())
        cell = request["cells"][0]
        protocol_path = Path(cell["protocol"]["path"])
        protocol = json.loads(protocol_path.read_text())
        protocol["model"]["completed_updates"] = 512
        cell["protocol"] = write_json(protocol_path, protocol)
        row_path = Path(cell["row"]["path"])
        row = json.loads(row_path.read_text())
        row["protocol_sha256"] = cell["protocol"]["sha256"]
        cell["row"] = write_json(row_path, row)
        write_json(self.path, request)
        with self.assertRaisesRegex(ValueError, "training budget"):
            renderer.load_request(self.path)

    def test_missing_saved_frame_is_not_backfilled(self):
        data = renderer.load_request(self.path)
        del data["panels"][0]["frames"][200]
        self.assertEqual(renderer.panel_state(data["panels"][0], 200)[0], "MISSING SAVED FRAME")

    def test_inconsistent_completion_rejected(self):
        self.mutate_row(lambda row: row.update(completed_steps=394))
        with self.assertRaisesRegex(ValueError, "committed status"):
            renderer.load_request(self.path)

    def test_truth_disagreement_rejected(self):
        request = json.loads(self.path.read_text())
        cell = request["cells"][1]
        row_path = Path(cell["row"]["path"])
        row = json.loads(row_path.read_text())
        npz_path = row_path.parent / row["trace_file"]
        with np.load(npz_path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        arrays["ground_truth_positions"][0, 0, 0] += .1
        np.savez_compressed(npz_path, **arrays)
        row["trace_sha256"] = bound(npz_path)["sha256"]
        cell["row"] = write_json(row_path, row)
        write_json(self.path, request)
        with self.assertRaisesRegex(ValueError, "ground truth differs"):
            renderer.load_request(self.path)

    def test_no_truth_is_invented_for_all_missing(self):
        request = json.loads(self.path.read_text())
        request["cells"] = [dict(arm=arm, policy=policy, coverage_state="never_started", reason="Synthetic missing")
                            for arm, policy, _ in renderer.METHODS]
        write_json(self.path, request)
        data = renderer.load_request(self.path)
        self.assertEqual(data["truth"], {})
        self.assertTrue(all(panel["prefix"] is None for panel in data["panels"]))

    def test_false_outside_grid_claim_rejected(self):
        request = json.loads(self.path.read_text())
        request["cells"][2]["coverage_state"] = "outside_evaluated_source_grid"
        write_json(self.path, request)
        with self.assertRaisesRegex(ValueError, "Outside-grid"):
            renderer.load_request(self.path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-fixtures", type=Path)
    args = parser.parse_args()
    if args.make_fixtures:
        directory = args.make_fixtures.resolve()
        renderer.require(renderer.HERE in directory.parents and not directory.exists(), "Fresh presentation fixture directory required")
        directory.mkdir(parents=True)
        for study in ("Goop2D", "Goop3D"):
            path = make_fixture(directory / study, study)
            renderer.render(renderer.load_request(path), directory / f"synthetic_{study.lower()}")
        print(json.dumps({"synthetic_only": True, "fixture_directory": str(directory)}, sort_keys=True))
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(QualitativeChecks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
