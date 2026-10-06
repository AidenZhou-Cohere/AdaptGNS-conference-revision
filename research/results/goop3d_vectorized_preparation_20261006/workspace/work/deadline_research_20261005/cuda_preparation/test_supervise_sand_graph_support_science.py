"""Scalar dictionaries, tiny CPU tensors and mocked processes only; no launches."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
PATH = HERE / "supervise_sand_graph_support_science.py"
spec = importlib.util.spec_from_file_location("scientific_supervisor_test", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture(autouse=True)
def configured():
    m.configure(SimpleNamespace(capacity_source=HERE / "measure_sand_graph_support_capacity.py",
        lifecycle_source=HERE / "measure_sand_cuda_capacity_v2.py", trainer=HERE / "train_sand_graph_support_cuda.py"))


def test_default_does_not_import_scientific_packages_or_run():
    code = """import builtins,runpy,sys
original=builtins.__import__
def restricted(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','numpy','scipy','gns','research'): raise AssertionError(name)
 return original(name,*args,**kwargs)
builtins.__import__=restricted
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(PATH)], capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)
    assert value["status"] == "description_only" and value["updates"] == 100000


def test_fixed_scientific_command_and_pinned_lifecycle():
    args = SimpleNamespace(**{k: Path("/" + k) for k in ("python", "trainer", "repo", "train_manifest", "admission", "structural_report", "protocol", "output_dir")})
    assert m.sha(m.B.__file__) == m.LIFECYCLE_SHA
    assert Path(m.B.cleanup_owned.__code__.co_filename).name == "measure_sand_cuda_capacity_v2.py"
    assert m.B.verify_job is m.record_reaped_job
    for job in m.C.SCHEDULE:
        command = m.fixed_command(args, job)
        for flag, value in (("--updates", "100000"), ("--checkpoint-every", "10000"), ("--log-every", "100"),
                            ("--objective", "faithful"), ("--arm", job["arm"]), ("--threads", "2")):
            assert command[command.index(flag) + 1] == value
        assert not any(flag in command for flag in ("--resume", "--stop-after", "--clear-stale-lock"))
    for args in (["--execute"], ["--resume"], ["--host-role", "C"]):
        with pytest.raises(SystemExit):
            m.parse_args(args)


def forecast_fixture():
    costs = {"schema": "adaptgns_sand_graph_support_cost_estimate_v1", "status": "complete_workload_estimated",
             "policy_count": 6, "rollout_outcomes": 1080, "horizon": 314, "diagnostic_forward_calls": 124704,
             "full_rollout_seconds": 1000., "diagnostics_execution_seconds": 2000., "ledger_total_reserve_seconds": 1000.,
             "ledger_bound_rationale": "synthetic", "timing_evidence_sha256": ["a" * 64]}
    jobs = [{**job, "status": "verified_512_graph_timing_probe", "steady_wall_seconds_per_update": .1,
             "nonnegative_external_minus_all_guarded_seconds": 10.} for job in m.C.SCHEDULE]
    f = m.C.forecast(jobs, costs, datetime(2026, 10, 6, tzinfo=timezone.utc))
    return costs, {"schema": m.C.SCHEMA, "status": "all_six_verified", "scientific_training_admitted": False,
                   "cost_estimate_sha256": "c" * 64, "jobs": jobs, "forecast": f}


def release_fixture():
    costs, forecast = forecast_fixture()
    now = datetime(2026, 10, 6, 1, tzinfo=timezone.utc)
    budget = m.forecast_contract(forecast, costs, "c" * 64)
    uuids = [f"00000000-0000-0000-0000-{i:012x}" for i in range(4)]
    hashes = {"costs": "c" * 64}
    process = {"schema": "adaptgns_sand_scientific_process_check_v1", "checked_utc": now.isoformat(),
               "host": m.socket.gethostname(), "gpu_uuids": uuids, "matching_training_processes": [], "gpu_processes": []}
    clock = {"schema": "adaptgns_sand_scientific_clock_check_v1", "checked_utc": now.isoformat(), "issued_by": "root",
             "host": m.socket.gethostname(), "root_host_samples_reviewed": True, "clock_error_bound_seconds": 1.}
    release = {"schema": m.RELEASE_SCHEMA, "status": "admitted_for_scientific_training", "issued_by": "root",
               "scientific_training_admitted": True, "files_sha256": hashes, "schedule": m.C.SCHEDULE, "environment": m.C.ENVIRONMENT,
               "host_role": "A", "host": m.socket.gethostname(), "cohort_id": "synthetic", "review_rationale": "synthetic only",
               "gpu_uuids": uuids, "process_identity_checked_utc": process["checked_utc"], "clock_checked_utc": clock["checked_utc"],
               "clock_error_bound_seconds": 1., "latest_start_utc": (budget["latest_start_utc"] - timedelta(seconds=1 + m.B.CLEANUP_SECONDS)).isoformat(),
               "training_stop_utc": budget["training_stop_utc"].isoformat(), "compute_analysis_deadline_utc": m.DEADLINE.isoformat()}
    return release, hashes, SimpleNamespace(host_role="A"), process, clock, forecast, costs, now


def test_complete_forecast_and_fresh_root_release_fit():
    values = release_fixture()
    result = m.validate_release(*values[:-1], now=values[-1])
    assert result["training_stop_utc"] < m.DEADLINE
    assert result["latest_start_utc"] < result["training_stop_utc"]
    assert result["cleanup_trigger_utc"] == result["training_stop_utc"] - timedelta(seconds=1 + m.B.CLEANUP_SECONDS)


def test_zero_timing_residual_is_valid():
    costs, forecast = forecast_fixture()
    for job in forecast["jobs"]:
        job["nonnegative_external_minus_all_guarded_seconds"] = 0.
    forecast["forecast"] = m.C.forecast(forecast["jobs"], costs, datetime(2026, 10, 6, tzinfo=timezone.utc))
    assert m.forecast_contract(forecast, costs, "c" * 64)["training_seconds"] > 0


@pytest.mark.parametrize("change", [
    lambda v: v[0].update(scientific_training_admitted=False),
    lambda v: v[0].update(host_role="B"),
    lambda v: v[0].update(latest_start_utc=m.DEADLINE.isoformat()),
    lambda v: v[0].update(training_stop_utc=v[-1].isoformat()),
    lambda v: v[0].update(latest_start_utc=(m.stamp(v[0]["training_stop_utc"]) - timedelta(seconds=m.forecast_contract(v[5], v[6], "c" * 64)["training_seconds"])).isoformat()),
    lambda v: v[3].update(gpu_processes=[{"pid": 123}]),
    lambda v: v[3].update(checked_utc="2026-10-05T00:00:00+00:00"),
    lambda v: v[4].update(root_host_samples_reviewed=False),
    lambda v: v[5]["forecast"].update(total_compute_analysis_seconds=None),
    lambda v: v[5]["jobs"].pop(),
    lambda v: v[5].update(cost_estimate_sha256="f" * 64),
    lambda v: v[5]["forecast"].update(total_compute_analysis_seconds=1.),
])
def test_incomplete_mismatched_or_unfit_release_never_passes(change):
    values = list(release_fixture())
    change(values)
    with pytest.raises((ValueError, TypeError)):
        m.validate_release(*values[:-1], now=values[-1])


def payload_fixture(torch, completed):
    config = {"arm": "base", "seed": 0, "updates": 100000, "objective": "faithful"}
    history = {"training": [], "graph_updates": [None] * completed, "elapsed_seconds": 1.}
    simulator = {"particle_dimensions": 2, "nnode_in": 30, "nedge_in": 3, "latent_dim": 128, "nmessage_passing_steps": 10,
                 "nmlp_layers": 2, "mlp_hidden_dim": 128, "connectivity_radius": .015, "nparticle_types": 9,
                 "particle_type_embedding_size": 16, "uncertainty_parameterization": "variance", "variance_floor": 1e-6,
                 "max_num_neighbors": 128, "detach_variance_features": True, "radius_backend": "scipy_host",
                 "boundaries": [[.1, .9], [.1, .9]], "boundary_clamp_limit": 1.,
                 "normalization_stats": {name: {"mean": torch.tensor([0., 0.]), "std": torch.tensor([.1, .2])}
                                         for name in ("acceleration", "velocity")}}
    group = {"params": [0], "betas": (.9, .999), "eps": 1e-8, "weight_decay": 0., "amsgrad": False,
             "maximize": False, "foreach": False, "capturable": False, "differentiable": False, "fused": False,
             "lr": 1e-4 * (.1 ** (max(completed - 1, 0) / 99999))}
    moments = {} if completed == 0 else {0: {"step": torch.tensor(float(completed)),
        "exp_avg": torch.tensor([.1, .2]), "exp_avg_sq": torch.tensor([.01, .04])}}
    payload = {"format_version": 2, "cuda_sand_graph_support_schema": m.T.SCHEMA, "completed_steps": completed,
               "run_config": config, "run_config_sha256": m.B.canonical_hash(config), "history": history,
               "training_config": {"loss": "faithful", "cuda_sand_graph_support_run": config, "completed_optimizer_updates": completed},
               "simulator_config": simulator, "state_dict": {"fake.weight": torch.tensor([1., 2.])},
               "optimizer_state": {"state": moments, "param_groups": [group]},
               "rng_states": {"cpu": torch.tensor([1, 2], dtype=torch.uint8), "cuda": torch.tensor([3, 4], dtype=torch.uint8)}}
    return payload, config, history


def test_cpu_payload_endpoint_and_initial_empty_adam():
    import torch
    for completed in (0, 100000):
        payload, config, history = payload_fixture(torch, completed)
        assert m.check_payload(torch, payload, config, completed, history)["completed_steps"] == completed


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(completed_steps=512),
    lambda p: p.update(cuda_sand_training_schema="old"),
    lambda p: p["state_dict"]["fake.weight"].fill_(float("nan")),
    lambda p: p["optimizer_state"]["state"][0]["step"].fill_(512),
    lambda p: p["optimizer_state"]["state"][0]["exp_avg_sq"].fill_(-1),
    lambda p: p["optimizer_state"]["param_groups"][0].update(foreach=True),
    lambda p: p["rng_states"].pop("cuda"),
    lambda p: p["simulator_config"]["normalization_stats"]["velocity"]["std"].fill_(0),
    lambda p: p["simulator_config"].update(boundaries=[[0., 1.], [0., 1.]]),
])
def test_partial_nonfinite_or_wrong_lineage_checkpoint_rejected(mutation):
    import torch
    payload, config, history = payload_fixture(torch, 100000)
    mutation(payload)
    with pytest.raises(ValueError):
        m.check_payload(torch, payload, config, 100000, history)


def host_args(tmp_path):
    for name in ("jobs", "logs"):
        (tmp_path / name).mkdir()
    return SimpleNamespace(host_role="B", output_dir=tmp_path,
        **{key: Path("/" + key) for key in ("python", "trainer", "repo", "train_manifest", "admission", "structural_report", "protocol")})


def test_child_registered_before_failed_identity_and_owned_cleanup_called(tmp_path, monkeypatch):
    args = host_args(tmp_path)
    now = datetime(2026, 10, 6, tzinfo=timezone.utc)
    monkeypatch.setattr(m.B, "now", lambda: now)
    monkeypatch.setattr(m.B, "gpu_processes", lambda: [])
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=123, returncode=None))
    monkeypatch.setattr(m.B, "process_identity", lambda pid: {"argv": ["wrong"], "ppid": m.os.getpid()})
    called = []
    def cleanup(args, children, record, verified, launch, manifest):
        called.append(len(children))
        for child in children:
            for handle in child["handles"]:
                handle.close()
        record["unreaped_owned_children"] = []
    monkeypatch.setattr(m.B, "cleanup_owned", cleanup)
    launch = {"latest_start_utc": (now + timedelta(hours=1)).isoformat(), "clock_error_bound_seconds": 1.}
    with pytest.raises(ValueError, match="identity"):
        m.run_host(args, launch, {}, now + timedelta(hours=2))
    assert called == [1]
    assert json.loads((tmp_path / "wave_B.json").read_text())["state"] == "failed"


def test_expired_fit_prevents_even_first_mock_child(tmp_path, monkeypatch):
    args = host_args(tmp_path)
    now = datetime(2026, 10, 6, tzinfo=timezone.utc)
    monkeypatch.setattr(m.B, "now", lambda: now)
    monkeypatch.setattr(m.B, "gpu_processes", lambda: [])
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: pytest.fail("No process may launch"))
    monkeypatch.setattr(m.B, "cleanup_owned", lambda *a: None)
    launch = {"latest_start_utc": now.isoformat(), "clock_error_bound_seconds": 1.}
    with pytest.raises(ValueError, match="launch-fit"):
        m.run_host(args, launch, {}, now + timedelta(hours=2))


def test_reaping_is_light_and_full_verification_waits_for_every_child(tmp_path, monkeypatch):
    args = host_args(tmp_path)
    now = datetime(2026, 10, 6, tzinfo=timezone.utc)
    processes, endpoints = [], []
    monkeypatch.setattr(m.B, "now", lambda: now)
    monkeypatch.setattr(m.B, "gpu_processes", lambda: [])
    def popen(command, **kwargs):
        child = SimpleNamespace(pid=100 + len(processes), returncode=None, command=command)
        processes.append(child)
        return child
    monkeypatch.setattr(m.subprocess, "Popen", popen)
    monkeypatch.setattr(m.B, "process_identity", lambda pid: {"argv": next(p.command for p in processes if p.pid == pid), "ppid": m.os.getpid()})
    monkeypatch.setattr(m.B, "capture_initial", lambda *args: None)
    def reap(args, child, record, retained, launch, manifest):
        directory = args.output_dir / "jobs" / child["job"]["id"]
        directory.mkdir()
        (directory / "status.json").write_text(json.dumps({"state": "complete", "error": None,
            "completed_steps": 100000, "committed_steps": 100000, "requested_steps": 100000}))
        external = {"exit_code": 0, "signals": [], "nominal_poll_interval_seconds": .1}
        retained.append(m.B.verify_job(directory, child["job"], external, manifest, "p", "g"))
        assert not endpoints
        assert retained[-1]["status"] == "reaped_pending_endpoint_verification"
        assert retained[-1]["external"]["nominal_poll_interval_seconds"] == .2
        child["process"].returncode = 0
        for handle in child["handles"]:
            handle.close()
    monkeypatch.setattr(m.B, "reap_child", reap)
    def endpoint(directory, job, *args):
        assert len(processes) == 2 and all(p.returncode == 0 for p in processes)
        endpoints.append(job["id"])
        return job
    monkeypatch.setattr(m, "verify_job", endpoint)
    monkeypatch.setattr(m, "verify_pairing", lambda jobs: [{"seed": 2}])
    monkeypatch.setattr(m.time, "sleep", lambda *args: None)
    launch = {"latest_start_utc": (now + timedelta(hours=1)).isoformat(), "clock_error_bound_seconds": 1.,
              "files_sha256": {"protocol": "p"}, "gpu_uuids": ["g"] * 4}
    verified, pairing = m.run_host(args, launch, {}, now + timedelta(hours=2))
    assert len(verified) == 2 and pairing == [{"seed": 2}]
    assert json.loads((tmp_path / "wave_B.json").read_text())["state"] == "verified"


def test_changed_snapshot_or_release_is_detected(tmp_path, monkeypatch):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "source.py").write_text("source")
    (tmp_path / "protocol.md").write_text("protocol")
    (tmp_path / "inputs/protocol.md").write_text("protocol")
    (tmp_path / "release.json").write_text('{"status": "synthetic"}')
    (tmp_path / "inputs/release.json").write_bytes((tmp_path / "release.json").read_bytes())
    monkeypatch.setattr(m.T, "SOURCE_PINS", {"source.py": m.sha(tmp_path / "source.py")})
    args = SimpleNamespace(repo=tmp_path, output_dir=tmp_path, release=tmp_path / "release.json")
    values = (args, {"protocol": tmp_path / "protocol.md"}, {"protocol": m.sha(tmp_path / "protocol.md")},
              {"protocol": "inputs/protocol.md"}, {"release_sha256": m.sha(args.release)}, m.read(args.release))
    m.verify_bound_inputs(*values)
    (tmp_path / "inputs/protocol.md").write_text("changed")
    with pytest.raises(ValueError, match="copied"):
        m.verify_bound_inputs(*values)
    (tmp_path / "inputs/protocol.md").write_text("protocol")
    (tmp_path / "inputs/release.json").write_text("{}")
    with pytest.raises(ValueError, match="release"):
        m.verify_bound_inputs(*values)
