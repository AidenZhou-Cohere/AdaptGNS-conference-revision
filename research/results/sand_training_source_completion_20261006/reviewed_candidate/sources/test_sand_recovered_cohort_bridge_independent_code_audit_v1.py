"""Independent scalar-only bridge checks. No model/data payload or remote access."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sand_recovered_bridge_independent", HERE / "prepare_sand_recovered_final_cohort_v1.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
H = B.helper()


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def fixture():
    # Completely synthetic scalar evidence; the production source's exact
    # release digest is represented by a fake inventory row, never read.
    release = {"schema": "adaptgns_sand_runtime_migration_recovery_release_v1",
               "status": "approved_for_sand_migration_training", "issued_by": "root", "cohort_id": B.COHORT_ID,
               "host": "synthetic-host", "owner_output_directory": "/synthetic/owner",
               "training_stop_utc": "2026-10-06T22:44:00+00:00", "python_environment": {"lexical_path": "/synthetic/venv/bin/python"},
               "adapter_path": "/synthetic/source/train_sand_runtime_migration_recovery_v1.py", "jobs": []}
    for plan in H.schedule("A"):
        name = plan["id"]
        checkpoints = {f"checkpoint-{step:09d}.pt": digest(name + "original" + str(step))
                       for step in range(0, 50001, 10000)}
        release["jobs"].append({"id": name, "arm": plan["arm"], "seed": plan["seed"], "gpu": plan["gpu"],
            "origin_directory": "/synthetic/original/" + name, "output_directory": "/synthetic/recovered/" + name,
            "original_runtime": {"deterministic_algorithms": True, "cublas_workspace_config": ":4096:8"},
            "actual_runtime": {"gpu_uuid": "synthetic-gpu-" + str(plan["gpu"])},
            "origin_checkpoint_inventory": checkpoints, "origin_checkpoint_sha256": checkpoints["checkpoint-000050000.pt"],
            "origin_protocol_sha256": digest(name + "protocol"), "origin_pointer_sha256": digest(name + "pointer"),
            "origin_lock_sha256": digest(name + "lock"), "origin_stdout_path": "/synthetic/original_logs/" + name,
            "origin_stdout_sha256": digest(name + "stdout"), "replay_receipt_path": "/synthetic/replay/" + name + ".json",
            "replay_receipt_sha256": digest(name + "replay")})
    release_path = "/synthetic/recovery_release.json"
    owner = Path(release["owner_output_directory"])
    files = {release_path: {"sha256": B.RECOVERY_RELEASE_SHA, "bytes": 1}}

    def add(path, value=None):
        value = value or digest(str(path))
        files[str(path)] = {"sha256": value, "bytes": 1}
        return value

    started = {"schema": B.OWNER_SCHEMA, "mode": "train", "owner_pid": 100,
               "owner_release_sha256": B.RECOVERY_RELEASE_SHA}
    wave = {"schema": B.OWNER_SCHEMA, "mode": "train", "state": "verified_recovery_endpoints",
            "error": None, "unreaped_owned_children": [], "never_started": [], "final_gpu_inventory": [],
            "whole_six_model_cohort_admitted": False, "scientific_promotion_from_replay": False,
            "owner_identity": {"pid": 100}, "ended_utc": "2026-10-06T22:00:00+00:00",
            "audit_ended_utc": "2026-10-06T22:10:00+00:00", "verified_jobs": [], "all_children": [], "pairing": []}
    terminal = {"schema": B.OWNER_SCHEMA, "mode": "train", "status": "complete_stopped_and_reaped",
                "owner_release_sha256": B.RECOVERY_RELEASE_SHA, "terminal_utc": "2026-10-06T22:11:00+00:00"}
    for name in ("owner_started.json", "wave_A.json", "owner_terminal.json"):
        add(owner / name)
    terminal["wave_sha256"] = files[str(owner / "wave_A.json")]["sha256"]
    process = {"schema": B.PROCESS_SCHEMA, "status": "verified_completed_recovered_A", "issued_by": "root",
               "hostname": release["host"], "owner_pid": 100, "owner_release_sha256": B.RECOVERY_RELEASE_SHA,
               "owner_exited": True, "all_children_reaped": True, "matching_training_processes": [],
               "scoped_gpu_inventory": [], "checked_utc": "2026-10-06T22:12:00+00:00",
               "files_sha256": {release_path: B.RECOVERY_RELEASE_SHA, **{
                   str(owner / n): files[str(owner / n)]["sha256"] for n in
                   ("owner_started.json", "wave_A.json", "owner_terminal.json")}}}
    jobs = {}
    for number, original in enumerate(release["jobs"]):
        job = copy.deepcopy(original)
        origin, recovered = Path(job["origin_directory"]), Path(job["output_directory"])
        config = {"schema": H.TRAIN_SCHEMA, "arm": job["arm"], "seed": job["seed"], "updates": 100000,
                  "research_protocol_sha256": H.PINS["protocol"], "checkpoint_every": 10000,
                  "log_every": 100, "runtime": original["original_runtime"]}
        config_sha = H.canonical(config)
        for name, value in job["origin_checkpoint_inventory"].items():
            add(origin / name, value)
        recovered_inventory = {f"checkpoint-{step:09d}.pt": digest(job["id"] + str(step))
                               for step in range(50000, 100001, 10000)}
        recovered_inventory["checkpoint-000050000.pt"] = job["origin_checkpoint_sha256"]
        for name, value in recovered_inventory.items():
            add(recovered / name, value)
        for path, value in ((origin / "protocol.json", job["origin_protocol_sha256"]),
                            (origin / "latest.json", job["origin_pointer_sha256"]),
                            (origin / "run.lock", job["origin_lock_sha256"]),
                            (Path(job["origin_stdout_path"]), job["origin_stdout_sha256"]),
                            (Path(job["replay_receipt_path"]), job["replay_receipt_sha256"])):
            add(path, value)
        artifacts = {name: add(recovered / name) for name in
                     ("protocol.json", "status.json", "history.json", "latest.json", "runtime_migration.json", "migration_receipt.json")}
        artifacts["protocol.json"] = add(recovered / "protocol.json", job["origin_protocol_sha256"])
        initial = {"path": "checkpoint-000000000.pt", "completed_steps": 0, "run_config_sha256": config_sha,
                   "sha256": job["origin_checkpoint_inventory"]["checkpoint-000000000.pt"]}
        final = {"path": "checkpoint-000100000.pt", "completed_steps": 100000, "run_config_sha256": config_sha,
                 "sha256": recovered_inventory["checkpoint-000100000.pt"]}
        command = [release["python_environment"]["lexical_path"], "-B", release["adapter_path"], "--execute",
                   "--mode", "train", "--release", release_path, "--release-sha256", B.RECOVERY_RELEASE_SHA,
                   "--job", job["id"]]
        external = {"exit_code": 0, "signals": [], "command": command, "pid": 200 + number,
                    "identity": {"argv": command, "pid": 200 + number, "ppid": 100},
                    "owner_release_sha256": B.RECOVERY_RELEASE_SHA}
        for stream in ("stdout", "stderr"):
            path = str(owner / "logs" / (job["id"] + "." + stream))
            external[stream + "_file"] = path
            external[stream + "_sha256"] = add(path)
        job.update(schema=B.AUDIT_SCHEMA, status="verified_recovered_scientific100k_endpoint", directory=str(recovered),
                   recovery_release_path=release_path, recovery_release_sha256=B.RECOVERY_RELEASE_SHA,
                   parent_completed_steps=50000, new_completed_updates=50000, parent_history_prefix_exact=True,
                   original_lock_retained=True, historical_runtime_is_original_lineage=True,
                   cross_physical_device_bitwise_equivalence_claim=False, recovery_is_fresh_from_scratch=False,
                   original_initialization_from_scratch_verified=True, config_sha256=config_sha, runtime=config["runtime"],
                   final_pointer=final, initial_pointer=initial, history={"graph_updates": 100000, "logged_rows": 1001},
                   combined_scientific_checkpoint_steps=list(range(0, 100001, 10000)),
                   recovered_checkpoint_inventory=recovered_inventory, artifact_sha256=artifacts, external=external)
        for key, step in (("initial_checkpoint", 0), ("parent_checkpoint", 50000), ("endpoint", 100000)):
            job[key] = {"completed_steps": step, "finite_model_tensors": 3, "adam_parameter_states": 3 if step else 0,
                        "cpu_cuda_rng_serialized": True, "checkpoint_schema_verified": True}
        status = {"schema": H.TRAIN_SCHEMA, "state": "complete", "error": None, "completed_steps": 100000,
                  "committed_steps": 100000, "requested_steps": 100000, "run_config_sha256": config_sha,
                  "latest_checkpoint": final}
        jobs[job["id"]] = {"config": config, "status": status, "pointer": final}
        wave["verified_jobs"].append(job)
        wave["all_children"].append({"job": original, "exit_code": 0, "signals": []})
    for seed in (1, 2):
        wave["pairing"].append({"seed": seed, "initial_model_adam_cpu_cuda_rng_exact": True,
            "all100k_frame_noise_native_graph_rng_exact": True, "all_logged_saved_lr_exact": True,
            "original50k_prefix_preserved_in_each_arm": True, "cross_physical_device_bitwise_equivalence_claim": False,
            "initial_audit_ordering": "verified after recovery training; no before-step1 barrier claimed"})
    return {"release": release, "release_path": release_path, "started": started, "wave": wave,
            "terminal": terminal, "process": process, "jobs": jobs}, files


class ScalarBridgeReview(unittest.TestCase):
    def test_all_four_recovered_jobs_accept_exact_split_lineage(self):
        receipts, files = fixture()
        jobs, pairs = B.check_a(receipts, files, H)
        self.assertEqual({(j["arm"], j["seed"]) for j in jobs}, {(a, s) for a in ("base", "mix") for s in (1, 2)})
        self.assertEqual({p["seed"] for p in pairs}, {1, 2})

    def test_scalar_mutations_reject_incomplete_or_mislabelled_recovery(self):
        changes = {
            "missing_endpoint": lambda r, f: r["wave"]["verified_jobs"].pop(),
            "unreaped": lambda r, f: r["wave"]["unreaped_owned_children"].append(200),
            "foreign_gpu_process": lambda r, f: r["process"]["scoped_gpu_inventory"].append({"pid": 999}),
            "wrong_owner": lambda r, f: r["process"].update(owner_pid=101),
            "wrong_native_digest": lambda r, f: r["process"]["files_sha256"].update({r["release_path"]: "0" * 64}),
            "late_audit": lambda r, f: r["wave"].update(audit_ended_utc="2026-10-07T00:00:00+00:00"),
            "fresh_training_claim": lambda r, f: r["wave"]["verified_jobs"][0].update(recovery_is_fresh_from_scratch=True),
            "bitwise_migration_claim": lambda r, f: r["wave"]["verified_jobs"][0].update(cross_physical_device_bitwise_equivalence_claim=True),
            "missing_original_history": lambda r, f: r["wave"]["verified_jobs"][0].update(parent_history_prefix_exact=False),
            "short_graph_history": lambda r, f: r["wave"]["verified_jobs"][0]["history"].update(graph_updates=50000),
            "wrong_endpoint_adam": lambda r, f: r["wave"]["verified_jobs"][0]["endpoint"].update(adam_parameter_states=0),
            "nonzero_child": lambda r, f: r["wave"]["all_children"][0].update(exit_code=1),
            "short_pairing": lambda r, f: r["wave"]["pairing"].pop(),
            "unverified_noise_pairing": lambda r, f: r["wave"]["pairing"][0].update(all100k_frame_noise_native_graph_rng_exact=False),
            "wrong_copied_parent": lambda r, f: r["wave"]["verified_jobs"][0]["recovered_checkpoint_inventory"].update({"checkpoint-000050000.pt": "0" * 64}),
            "changed_original_lock": lambda r, f: f[str(Path(r["release"]["jobs"][0]["origin_directory"]) / "run.lock")].update(sha256="0" * 64),
            "changed_replay_receipt": lambda r, f: f[r["release"]["jobs"][0]["replay_receipt_path"]].update(sha256="0" * 64),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                receipts, files = fixture()
                change(receipts, files)
                with self.assertRaises(ValueError):
                    B.check_a(receipts, files, H)

    def test_incomplete_gate_precedes_checkpoint_tree_hash(self):
        receipts, files = fixture()
        owner = Path(receipts["release"]["owner_output_directory"])
        mapping = {receipts["release_path"]: receipts["release"], "/synthetic/process.json": receipts["process"],
                   **{str(owner / n): receipts[k] for k, n in (("started", "owner_started.json"),
                                                              ("wave", "wave_A.json"), ("terminal", "owner_terminal.json"))}}
        receipts["process"]["owner_exited"] = False
        def fake_sha(path):
            return files[str(path)]["sha256"]
        with patch.object(B, "helper", return_value=H), patch.object(B, "read", side_effect=lambda p: mapping[str(p)]), \
             patch.object(B, "sha", side_effect=fake_sha), patch.object(B, "tree_inventory") as tree:
            with self.assertRaises(ValueError):
                B.inventory_a(Path(receipts["release_path"]), Path("/synthetic/process.json"))
            tree.assert_not_called()

    def test_build_preserves_six_models_and_honest_provenance_for_new_test_gate(self):
        receipts, files = fixture()
        a = {"schema": B.INVENTORY_SCHEMA, "status": "completed_recovered_A_bytes_verified", "issued_by": "root",
             "source_sha256": B.sha(B.__file__), "source_pins": B.SOURCE_PINS, "cohort_id": B.COHORT_ID,
             "host_role": "A", "receipts": receipts, "files": files}
        b_jobs = []
        for j in receipts["wave"]["verified_jobs"][:2]:
            j = copy.deepcopy(j)
            j.update(seed=0, directory="/synthetic/original_B/" + j["arm"])
            j["final_pointer"]["sha256"] = digest(j["arm"] + "B-final")
            b_jobs.append(j)
        b = {"schema": H.INVENTORY_SCHEMA, "status": "completed_host_bytes_verified", "issued_by": "root",
             "host_role": "B", "cohort_id": B.COHORT_ID, "source_sha256": B.SOURCE_PINS["prepare_sand_final_cohort_scoped_v1.py"],
             "receipts": {"summary": {"jobs": b_jobs, "pairing": [{"seed": 0}]}}, "files": {}}
        paths = {k: Path("/synthetic/final_sources") / k for k in H.FINAL_PINS}
        inventory_paths = {"A": Path("/synthetic/inventory_A.json"), "B": Path("/synthetic/inventory_B.json")}
        bindings = {str(v): H.FINAL_PINS[k] for k, v in paths.items()}
        bindings.update({str(p): digest(role + "-inventory") for role, p in inventory_paths.items()})
        bindings[str(Path(B.__file__))] = B.sha(B.__file__)
        release = {"schema": B.RELEASE_SCHEMA, "status": "approved_complete_recovery_cohort_bridge", "issued_by": "root",
                   "adapter_sha256": bindings[str(Path(B.__file__))], "cohort_id": B.COHORT_ID,
                   "recovery_release_sha256": B.RECOVERY_RELEASE_SHA, "review_rationale": "synthetic test only",
                   "original_B_adapter_sha256": B.SOURCE_PINS["prepare_sand_final_cohort_scoped_v1.py"],
                   "final_source_sha256": H.FINAL_PINS,
                   "inventory_sha256": {role: bindings[str(path)] for role, path in inventory_paths.items()}}
        # Original B's receipt validator is separately frozen and previously
        # tested; this stub isolates only the new bridge's interface mapping.
        with patch.object(B, "helper", return_value=H), patch.object(H, "check_receipts", return_value="B") as old_b, \
             patch.object(B, "read", side_effect=lambda p: a if p == inventory_paths["A"] else b), \
             patch.object(B, "sha", side_effect=lambda p: bindings[str(p)]):
            audit, cohort = B.build(inventory_paths["A"], inventory_paths["B"], release, paths)
        old_b.assert_called_once_with(b["receipts"], b["files"])
        self.assertEqual(len(cohort["models"]), 6)
        self.assertEqual(len(audit["paired_seeds"]), 3)
        self.assertEqual(audit["provenance"]["A"]["kind"], "original50k_plus_reviewed_recovered50k")
        self.assertEqual(audit["provenance"]["B"]["kind"], "original_uninterrupted100k")
        self.assertFalse(audit["cross_physical_device_bitwise_equivalence_claim"])
        self.assertFalse(audit["fresh_tensor_deserialization_by_adapter"])
        self.assertEqual({m["runtime_migration"] for m in audit["models"] if m["host_role"] == "A"}, {True})
        spec = importlib.util.spec_from_file_location("new_sand_reserved_gate", HERE / "prepare_sand_reserved_test_recovered_v1.py")
        new = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(new)
        new.cohort_gate(cohort, audit)
        short = copy.deepcopy(cohort)
        short["models"].pop()
        with self.assertRaises(ValueError):
            new.cohort_gate(short, audit)

    def test_old_reserved_test_gate_does_not_accept_new_adapter_identity(self):
        # Regression documents the downstream integration blocker without
        # opening a split, deserializing tensors or changing either source.
        spec = importlib.util.spec_from_file_location("old_sand_reserved_gate", HERE / "prepare_sand_reserved_test_scoped_v1.py")
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        cohort = {"schema": "adaptgns_sand_graph_support_final_cohort_v1", "status": "frozen_for_final_evaluation",
                  "issued_by": "root", "dataset": "Sand", "updates": 100000,
                  "protocol_sha256": old.PROTOCOL_SHA, "training_admission_sha256": old.TRAIN_ADMISSION_SHA,
                  "trainer_source_sha256": old.TRAINER_SHA, "adapter_sha256": B.sha(B.__file__),
                  "operational_amendment_sha256": old.AMENDMENT_SHA, "schedule_plan_sha256": old.PLAN_SHA}
        with self.assertRaisesRegex(ValueError, "Complete reviewed Sand scoped cohort"):
            old.cohort_gate(cohort, {})


if __name__ == "__main__":
    unittest.main()
