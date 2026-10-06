"""Synthetic committed-array checks; no scientific model/data is opened."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import fit_goop_action_gate_v1 as fit
import goop_global_action_gate_core_v1 as core


def descriptors(arrays):
    return {k:dict(shape=list(v.shape), dtype=v.dtype.str,
        value_sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()) for k, v in arrays.items()}


class FitChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); self.out = self.root / "labels"; self.out.mkdir()
        source = self.root / "synthetic_input.bin"; source.write_bytes(b"synthetic only")
        self.input_path = source
        self.ids = [(0, 6), (0, 62)]
        # Only the schedule cardinality is reduced for this artifact fixture.
        # Production main loads the byte-pinned real core; its 8000/150 identity
        # checks and full numerical fit are tested separately with real counts.
        def checked_rows(ids, split):
            fit.require(ids == [list(v) for v in self.ids], "Synthetic ordered ledger differs")
        self.helper = SimpleNamespace(row_ids=lambda split:self.ids, checked_rows=checked_rows,
            features=core.features, action_errors=core.action_errors, rng_material=core.rng_material)
        fit.publish(self.out / "protocol.json", {"fixture": "synthetic"})
        h = np.full((6, 2, 2), .5, dtype=np.float32); h[:, 1, 0] = .517
        a = {"history": h, "particle_types": np.full(2, 7, dtype=np.int64),
            "features": core.features(h), "base_edges": np.array([[0, 1], [0, 1]], dtype=np.int64),
            "random25_edges": np.array([[0, 1, 0, 1], [0, 1, 1, 0]], dtype=np.int64),
            "random25_selected_optional_pairs": np.array([[0, 1]], dtype=np.int64),
            "base_prediction": h[-1] + np.float32(.002), "random25_prediction": h[-1] + np.float32(.001),
            "base_risk": np.ones(2, dtype=np.float32), "base_raw_risk": np.zeros(2, dtype=np.float32),
            "random25_risk": np.ones(2, dtype=np.float32), "random25_raw_risk": np.zeros(2, dtype=np.float32),
            "target": h[-1].copy()}
        errors = core.action_errors(a["base_prediction"], a["random25_prediction"], a["target"])
        a["action_errors"] = np.array(list(errors.values()), dtype=np.float64)
        self.arrays = a; rows = []
        for source_index, target in self.ids:
            name = f"source_{source_index:06d}_target_{target:03d}"
            np.savez_compressed(self.out / (name + ".npz"), **a)
            row = dict(schema=fit.ROW_SCHEMA, model_seed=0, arm="mix", checkpoint_updates=100000,
                checkpoint_sha256="b" * 64, split="train", source_index=source_index, target_frame=target,
                status="complete", failure=None, gate_protocol_sha256=fit.PROTOCOL_SHA,
                protocol_sha256=fit.sha(self.out / "protocol.json"), source_manifest_sha256="a" * 64,
                source_trajectory_content_sha256="c" * 64, history_sha256=descriptors(a)["history"]["value_sha256"],
                features=a["features"].tolist(), **errors, pair_rng_material=core.rng_material("train", 0, source_index, target),
                base_graph={}, random25_graph={}, network_passes=2, timing={},
                prediction_sha256={p:descriptors(a)[p + "_prediction"]["value_sha256"] for p in ("base", "random25")},
                artifact_file=name + ".npz", artifact_sha256=fit.sha(self.out / (name + ".npz")), numeric_arrays=descriptors(a))
            pin = fit.publish(self.out / (name + ".json"), row)
            rows.append(dict(row, row_file=name + ".json", row_sha256=pin))
        files, entries = fit.inventory(self.out)
        self.collection = dict(schema=fit.COLLECTION_SCHEMA, status="complete", mode="train-labels", split="train",
            all_required_labels_complete=True, all_inputs_reverified=True, required_rows=2, committed_rows=2, complete_rows=2,
            model=dict(arm="mix", seed=0, objective="faithful", completed_updates=100000, checkpoint_sha256="b" * 64, run_config_sha256="e" * 64),
            gate_protocol_sha256=fit.PROTOCOL_SHA, original_training_protocol_sha256="f" * 64, core_sha256=fit.CORE_SHA,
            driver_sha256="d" * 64, cohort_sha256="1" * 64, cohort_audit_sha256="2" * 64,
            training_admission_sha256="3" * 64, source_manifest_sha256="a" * 64, metadata_sha256="4" * 64,
            expected_row_ids=[list(v) for v in self.ids], rows=rows, coverage=[dict(source_index=s,target_frame=t,state="committed_complete") for s,t in self.ids],
            input_sha256={str(source):fit.sha(source)}, files=files, output_tree_entries=entries)
        self.path = self.out / "label_collection.json"; self.path.write_bytes(fit.encode(self.collection))

    def read(self):
        return fit.read_collection(dict(seed=0,split="train",path=str(self.path),sha256=fit.sha(self.path)),
            dict(driver_sha256="d"*64),np,self.helper,{},lambda:None)

    def test_committed_actions_are_independently_recomputed(self):
        collection, values = self.read()
        self.assertEqual(collection["complete_rows"], 2)
        np.testing.assert_array_equal(values["features"], np.tile(self.arrays["features"], (2, 1)))
        np.testing.assert_array_equal(values["benefits"], self.arrays["action_errors"][2])

    def test_numeric_bytes_and_descriptor_tampering_are_rejected(self):
        row = copy.deepcopy(self.collection["rows"][0]); row["numeric_arrays"]["history"]["shape"] = [6, 1, 2]
        with self.assertRaises(ValueError):
            fit.array_check(np, core, row, self.out / row["artifact_file"])
        with (self.out / row["artifact_file"]).open("ab") as f:f.write(b"changed")
        with self.assertRaises(ValueError):self.read()

    def test_consistently_rehashed_false_labels_are_rejected(self):
        a = {k:v.copy() for k,v in self.arrays.items()}; a["features"][0] += 1
        row = copy.deepcopy(self.collection["rows"][0]); row["features"] = a["features"].tolist()
        path = self.out / row["artifact_file"]; np.savez_compressed(path, **a)
        row.update(artifact_sha256=fit.sha(path), numeric_arrays=descriptors(a))
        with self.assertRaisesRegex(ValueError, "Saved features"):
            fit.array_check(np, core, row, path)

    def test_base_prefix_and_optional_orientations_are_checked(self):
        a = {k:v.copy() for k,v in self.arrays.items()}; a["random25_edges"][:, 2] = [0, 0]
        row = copy.deepcopy(self.collection["rows"][0]); path = self.out / row["artifact_file"]
        np.savez_compressed(path, **a); row.update(artifact_sha256=fit.sha(path), numeric_arrays=descriptors(a))
        with self.assertRaisesRegex(ValueError, "orientations"):
            fit.array_check(np, core, row, path)

    def test_failed_missing_and_reordered_rows_block_fitting(self):
        for field, value in (("all_required_labels_complete", False), ("committed_rows", 1),
                             ("expected_row_ids", list(reversed(self.collection["expected_row_ids"]))),
                             ("mode", "train-label-capacity")):
            changed = dict(self.collection); changed[field] = value; self.path.write_bytes(fit.encode(changed))
            with self.subTest(field=field), self.assertRaises(ValueError):self.read()

    def test_output_tree_cannot_hide_uncommitted_or_symlink_entries(self):
        (self.out / "uncommitted.tmp").write_bytes(b"unfinished")
        with self.assertRaises(ValueError):self.read()
        (self.out / "uncommitted.tmp").unlink()
        (self.out / "hidden_link").symlink_to(self.input_path)
        with self.assertRaises(ValueError):self.read()

    def test_input_changes_and_conflicting_bindings_are_rejected(self):
        values={str(self.input_path):fit.sha(self.input_path)}
        with self.assertRaises(ValueError):fit.merge_inputs(values,{str(self.input_path):"0"*64})
        self.input_path.write_bytes(b"different")
        with self.assertRaises(ValueError):fit.verify_inputs(values)

    def test_default_has_no_execution_or_numpy_requirement(self):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):self.assertEqual(fit.main([]),0)
        self.assertFalse(json.loads(stream.getvalue())["scientific_execution"])

    def test_release_binds_actual_driver_and_original_cohort_members(self):
        inputs, paths = {}, {}
        for key in fit.RELEASE_PATHS:
            path=self.root/(key+'.json');path.write_bytes(fit.encode({'synthetic':key}))
            paths[key]=str(path);inputs[str(path)]=fit.sha(path)
        models=[];members=[]
        for seed in (0,1,2):
            path=self.root/f'mix{seed}.pt';path.write_bytes(f'synthetic checkpoint {seed}'.encode());pin=fit.sha(path);inputs[str(path)]=pin
            models.append(dict(seed=seed,checkpoint_path=str(path),checkpoint_sha256=pin,run_config_sha256='b'*64))
            for arm in ('base','mix'):
                members.append(dict(seed=seed,arm=arm,objective='faithful',completed_steps=100000,
                    checkpoint_sha256=pin if arm=='mix' else hashlib.sha256(f'base{seed}'.encode()).hexdigest()))
        cohort=dict(schema='adaptgns_goop_graph_support_final_cohort_v1',status='frozen_for_final_evaluation',
            dataset='Goop',updates=100000,issued_by='root',models=members,
            protocol_sha256=inputs[paths['original_training_protocol_sha256']],
            training_admission_sha256=inputs[paths['training_admission_sha256']],
            cohort_audit_sha256=inputs[paths['cohort_audit_sha256']])
        Path(paths['cohort_sha256']).write_bytes(fit.encode(cohort));inputs[paths['cohort_sha256']]=fit.sha(paths['cohort_sha256'])
        release=dict(lineage_paths=paths,models=models,driver_sha256=inputs[paths['driver_sha256']])
        fit.bind_release_lineage(release,inputs)
        missing=dict(inputs);del missing[paths['driver_sha256']]
        with self.assertRaises(ValueError):fit.bind_release_lineage(release,missing)
        missing=dict(inputs);del missing[models[0]['checkpoint_path']]
        with self.assertRaises(ValueError):fit.bind_release_lineage(release,missing)
        changed=copy.deepcopy(release);changed['models'][0]['seed']=False
        with self.assertRaises(ValueError):fit.bind_release_lineage(changed,inputs)

    def test_final_publication_does_not_serialize_again(self):
        raw=fit.encode({'status':'synthetic_pass'});path=self.root/'final.json'
        with mock.patch.object(fit,'encode',side_effect=AssertionError('Second serialization')):
            pin=fit.publish_bytes(path,raw)
        self.assertEqual(path.read_bytes(),raw)
        self.assertEqual(pin,hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    unittest.main()
