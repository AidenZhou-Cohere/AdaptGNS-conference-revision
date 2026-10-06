#!/usr/bin/env python3
"""Independent inert queue probes; never create an operating-system process."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "supervise_goop_action_gate_scoped_v1.py"
EXPECTED_SOURCE = "4cda31a408c097944e5f15b3502d05bcfab0d3db5b8ee25dcc5fdc1968fd61df"
TESTS = HERE / "test_supervise_goop_action_gate_scoped_v1.py"
EXPECTED_TESTS = "801e4f86160a46111b75fcb51d0f65bdf7d7ed408b6e747d9deba1d9be33ae87"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert digest(SOURCE) == EXPECTED_SOURCE
    assert digest(TESTS) == EXPECTED_TESTS
    spec = importlib.util.spec_from_file_location("_independent_gate_supervisor_fixtures", TESTS)
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    m = t.M
    output = HERE / "goop_action_gate_supervisor_retired_pid_final_fixture_code_audit_v1"
    output.mkdir(exist_ok=False)
    (output / "reviewed_source.py").write_bytes(SOURCE.read_bytes())
    cases = [
        ("retired_reused_gpu2", "retired", [2], "changed", True),
        ("retired_reused_gpu3", "retired", [3], "changed", True),
        ("retired_reused_both_unassigned", "retired", [2, 3], "changed", True),
        ("retired_absent_gpu2", "retired", [2], "absent", True),
        ("retired_same_identity_gpu2", "retired", [2], "same", False),
        ("live_owned_wrong_gpu2", "live", [2], "same", False),
        ("retired_reused_owned_gpu0", "retired", [0], "changed", False),
        ("foreign_owned_gpu1", "foreign", [1], "changed", False),
        ("retired_reused_mixed_owned_unassigned", "retired", [2, 0], "changed", False),
    ]
    outcomes = []
    for name, who, devices, identity_mode, expected_complete in cases:
        fixture = output / name
        fixture.mkdir()
        args, ctx = t.fixture(fixture)
        rt = t.Runtime(ctx)
        original_gpu_identity, original_stop = rt.gpu_identity, rt.stop
        state = {"injected": False, "target_pid": None, "retired_pid": None, "signals": []}

        def gpu_identity(pid):
            if state["injected"] and pid == state["target_pid"]:
                if identity_mode == "absent":
                    raise FileNotFoundError("inert retired PID no longer exists")
                if who == "foreign":
                    return {"pid": pid, "ppid": 1, "start_ticks": 90001, "argv": ["unrelated_fixture"]}
                value = original_gpu_identity(pid)
                if identity_mode == "changed":
                    return {**value, "ppid": 1, "start_ticks": value["start_ticks"] + 10000,
                            "argv": ["unrelated_fixture"]}
                return value
            return original_gpu_identity(pid)

        def gpu_result():
            if state["injected"] or len(rt.created) < 3:
                return None
            retired = rt.created[0]["process"]
            assert retired.returncode is not None
            live = rt.created[1]["process"]
            assert live.returncode is None
            state.update(injected=True, retired_pid=retired.pid,
                         target_pid=retired.pid if who == "retired" else live.pid if who == "live" else 88888)
            return ([{"pid": state["target_pid"], "gpu_uuid": ctx.release["gpu_uuids"][device]}
                     for device in devices], None)

        def stop(child, sig):
            state["signals"].append(child["process"].pid)
            return original_stop(child, sig)

        rt.gpu_identity, rt.gpu_result, rt.stop = gpu_identity, gpu_result, stop
        code = m.execute_queue(args, ctx, rt)
        ledger = m.snapshot(args.output_dir / "phase_ledger.json")[0]
        assert state["injected"]
        assert code == (0 if expected_complete else 1)
        assert ledger["state"] == ("complete_fixed_phase" if expected_complete else "stopped_requires_review")
        assert all(p["process"].returncode is not None for p in rt.created)
        assert state["retired_pid"] not in state["signals"] and 88888 not in state["signals"]
        if expected_complete:
            assert not state["signals"]
            observation = m.snapshot(args.output_dir / "logs/gpu_000001.json")[0]
            assert len(observation["retired_pid_observations"]) == 1
            assert all(v["classification"] == "unowned_device_observed_without_control"
                       for v in observation["checked"])
        outcomes.append({"case": name, "passed": True, "exit_code": code, "state": ledger["state"],
                         "abort_reason": ledger["abort_reason"], "unassigned_or_retired_pid_signaled": False,
                         "fixture": str(args.output_dir.relative_to(HERE))})
    assert digest(SOURCE) == EXPECTED_SOURCE and digest(TESTS) == EXPECTED_TESTS
    receipt = {"schema": "adaptgns_goop_action_gate_supervisor_retired_pid_independent_code_audit_v1",
               "status": "passed", "source_sha256": EXPECTED_SOURCE, "tests_sha256": EXPECTED_TESTS,
               "probe_source_sha256": digest(Path(__file__)), "inert_processes_only": True,
               "official_arrays_or_checkpoints_accessed": False, "operating_system_processes_launched": False,
               "passed": len(outcomes), "cases": outcomes}
    target = HERE / "goop_action_gate_supervisor_retired_pid_independent_code_audit_v1.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"path": str(target), "sha256": digest(target), "passed": len(outcomes)}, indent=2))


if __name__ == "__main__":
    main()
