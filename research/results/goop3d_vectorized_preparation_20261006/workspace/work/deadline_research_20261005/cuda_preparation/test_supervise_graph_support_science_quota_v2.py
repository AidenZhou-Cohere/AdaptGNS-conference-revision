"""Synthetic scalar allocations, tiny CPU tensors and mocked dispatch only."""
import ast
import builtins
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "supervise_graph_support_science_quota_v2.py"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


m = module(SOURCE, "science_quota_subject")
payload_helpers = module(HERE / "test_supervise_sand_graph_support_science.py", "private_test_payload_helpers")
goop_helpers = module(HERE / "test_measure_goop_graph_support_capacity.py", "private_test_goop_helpers")


@pytest.fixture(params=["Sand", "Goop"])
def configured(request):
    dataset = request.param
    m.configure(SimpleNamespace(dataset=dataset, v1_source=HERE / "supervise_sand_graph_support_science.py",
        capacity_source=HERE / f"measure_{dataset.lower()}_graph_support_capacity.py",
        lifecycle_source=HERE / "measure_sand_cuda_capacity_v2.py", trainer=HERE / f"train_{dataset.lower()}_graph_support_cuda.py"))
    payload_helpers.m = m
    goop_helpers.m = m.C
    return dataset


def allocation():
    return dict(schema=f"adaptgns_{m.DATASET.lower()}_graph_support_execution_allocation_v2", status="fixed_allocation_declared",
        issued_by="root", dataset=m.DATASET, cost_basis=m.COST_BASIS, stage_quotas_seconds=m.STAGE_QUOTAS,
        evaluation_quota_supervisor_sha256=m.EVALUATION_QUOTA_SHA,
        cleanup_seconds_per_invocation=15, evaluation_stream_allocation_seconds=11760,
        cohort_verification_reserve_seconds=2700, analysis_reserve_seconds=3600,
        all_required_evaluation_outcomes_promised=False, v1_timing_gate_reinterpreted=False,
        ledger_total_reserve_seconds=1000., ledger_bound_rationale="synthetic only", ledger_evidence_sha256=["a" * 64])


def capacity():
    jobs = [{**job, "status": "verified_512_graph_timing_probe", "steady_wall_seconds_per_update": .1,
             "nonnegative_external_minus_all_guarded_seconds": 10.} for job in m.C.SCHEDULE]
    return dict(schema=m.C.SCHEMA, status="all_six_verified", scientific_training_admitted=False, jobs=jobs,
                forecast=m.C.forecast(jobs, None, datetime(2026, 10, 6, tzinfo=timezone.utc)))


def release_fixture():
    costs, forecast = allocation(), capacity()
    now = datetime(2026, 10, 6, 1, tzinfo=timezone.utc)
    budget = m.forecast_contract(forecast, costs, "c" * 64)
    uuids = [f"00000000-0000-0000-0000-{i:012x}" for i in range(4)]
    hashes = dict(costs="c" * 64, v1_source=m.V1_SHA, supervisor=m.sha(SOURCE))
    process = dict(schema=f"adaptgns_{m.DATASET.lower()}_scientific_process_check_v1", checked_utc=now.isoformat(),
                   host=m.socket.gethostname(), gpu_uuids=uuids, matching_training_processes=[], gpu_processes=[])
    clock = dict(schema=f"adaptgns_{m.DATASET.lower()}_scientific_clock_check_v1", checked_utc=now.isoformat(), issued_by="root",
                 host=m.socket.gethostname(), root_host_samples_reviewed=True, clock_error_bound_seconds=1.)
    release = dict(schema=m.RELEASE_SCHEMA, status="admitted_for_scientific_training", issued_by="root", dataset=m.DATASET,
        cost_basis=m.COST_BASIS, all_required_evaluation_outcomes_promised=False, v1_timing_gate_reinterpreted=False,
        scientific_training_admitted=True, files_sha256=hashes, schedule=m.C.SCHEDULE, environment=m.C.ENVIRONMENT,
        host_role="A", host=m.socket.gethostname(), cohort_id="synthetic", review_rationale="synthetic only",
        gpu_uuids=uuids, process_identity_checked_utc=process["checked_utc"], clock_checked_utc=clock["checked_utc"],
        clock_error_bound_seconds=1., latest_start_utc=(budget["latest_start_utc"] - timedelta(seconds=16)).isoformat(),
        training_stop_utc=budget["training_stop_utc"].isoformat(), compute_analysis_deadline_utc=m.DEADLINE.isoformat())
    return release, hashes, SimpleNamespace(host_role="A"), process, clock, forecast, costs, now


def test_private_v1_loop_pairing_and_source_identities_are_preserved(configured):
    assert m.sha(m.V.__file__) == m.V1_SHA and m.V.__file__ != str(SOURCE)
    assert m.sha(m.C.__file__) == m.PINS[configured]["capacity"]
    assert m.sha(m.T.__file__) == m.PINS[configured]["trainer"]
    assert Path(m.V.run_host.__code__.co_filename).name == "supervise_sand_graph_support_science.py"
    assert Path(m.V.verify_pairing.__code__.co_filename).name == "supervise_sand_graph_support_science.py"
    assert m.V.verify_job is m.verify_job and m.V.fixed_command is m.fixed_command
    assert m.B.verify_job is m.V.record_reaped_job and (m.V.C, m.V.B, m.V.T) == (m.C, m.B, m.T)


def test_fixed_scientific_commands(configured):
    args = SimpleNamespace(**{k: Path("/" + k) for k in ("python", "trainer", "repo", "train_manifest", "admission",
        "structural_report", "protocol", "output_dir", "acquisition_report", "context_semantics", "auxiliary_report")})
    for job in m.C.SCHEDULE:
        command = m.fixed_command(args, job)
        for flag, value in (("--updates", "100000"), ("--checkpoint-every", "10000"), ("--log-every", "100"),
                            ("--objective", "faithful"), ("--arm", job["arm"]), ("--threads", "2")):
            assert command[command.index(flag) + 1] == value
        assert not any(x in command for x in ("--resume", "--stop-after", "--clear-stale-lock"))
        assert ("--context-semantics" in command) == (configured == "Goop")


def test_exact_budget_preserves_unadmitted_v1_forecast(configured):
    values = release_fixture()
    before = copy.deepcopy(values[5])
    assert before["forecast"]["fits_before_writing_reserve"] is None
    budget = m.validate_release(*values[:-1], now=values[-1])
    assert budget["training_seconds"] == 1.35 * (100000 * .1 + 12 * 10) + 1000
    assert budget["remaining_evaluation_verification_seconds"] == 11760 + 2700 + 3600
    assert budget["cleanup_trigger_utc"] == budget["training_stop_utc"] - timedelta(seconds=16)
    assert values[5] == before and budget["all_required_evaluation_outcomes_promised"] is False


@pytest.mark.parametrize("change", [
    lambda v: v[0].update(cost_basis="complete_workload_estimated"),
    lambda v: v[0].update(dataset="Wrong"),
    lambda v: v[0].update(v1_timing_gate_reinterpreted=True),
    lambda v: v[0].update(all_required_evaluation_outcomes_promised=True),
    lambda v: v[0].update(latest_start_utc=m.DEADLINE.isoformat()),
    lambda v: v[3].update(schema="wrong_dataset_process_check"),
    lambda v: v[4].update(clock_error_bound_seconds=6),
    lambda v: v[5]["jobs"].pop(),
    lambda v: v[5]["jobs"][0].update(steady_wall_seconds_per_update=float("nan")),
    lambda v: v[5]["forecast"].update(q6=.000001),
    lambda v: v[6].update(evaluation_stream_allocation_seconds=11700),
    lambda v: v[6].update(evaluation_quota_supervisor_sha256="0" * 64),
    lambda v: v[6].update(cohort_verification_reserve_seconds=0),
    lambda v: v[6].update(ledger_total_reserve_seconds=0),
    lambda v: v[6].update(ledger_evidence_sha256=[]),
])
def test_wrong_or_unfit_scientific_release_refused(configured, change):
    values = list(release_fixture())
    change(values)
    with pytest.raises(ValueError):
        m.validate_release(*values[:-1], now=values[-1])


def synthetic_payload(torch, completed):
    payload, config, history = payload_helpers.payload_fixture(torch, completed)
    if m.DATASET == "Goop":
        payload["cuda_goop_graph_support_schema"] = payload.pop("cuda_sand_graph_support_schema")
        payload["training_config"]["cuda_goop_graph_support_run"] = payload["training_config"].pop("cuda_sand_graph_support_run")
    return payload, config, history


def test_tiny_cpu_payload_initial_and_exact100k(configured):
    import torch
    for completed in (0, 100000):
        payload, config, history = synthetic_payload(torch, completed)
        assert m.check_payload(torch, payload, config, completed, history)["completed_steps"] == completed


@pytest.mark.parametrize("change", [
    lambda p: p.update(completed_steps=512),
    lambda p: p.update(**{"cuda_sand_graph_support_schema" if m.DATASET == "Goop" else "cuda_goop_graph_support_schema": "wrong"}),
    lambda p: p["state_dict"]["fake.weight"].fill_(float("nan")),
    lambda p: p["optimizer_state"]["state"][0]["step"].fill_(512),
    lambda p: p["rng_states"].pop("cuda"),
])
def test_wrong_payload_lineage_or_numerics_refused(configured, change):
    import torch
    payload, config, history = synthetic_payload(torch, 100000)
    change(payload)
    with pytest.raises(ValueError):
        m.check_payload(torch, payload, config, 100000, history)


def test_exact_dataset_frame_range(configured, monkeypatch):
    monkeypatch.setattr(m, "UPDATES", 1)
    monkeypatch.setattr(m.T, "validate_graph_history", lambda *args: None)
    target = m.FRAMES - 1
    ids = [f"train:000000:{target}"]
    row = dict(completed_steps=1, loss=.1, lr=1e-4, frame_ids=ids, particles=2, guarded_update_seconds=.1, elapsed_seconds=.2)
    history = dict(training=[row], graph_updates=[dict(frame_ids=ids, examples=[dict(n_particles=2)])])
    manifest = dict(records=[dict(id="train:000000", positions=dict(shape=[m.FRAMES, 2, 2]))])
    assert m.check_history(history, dict(arm="base", seed=0), manifest)["logged_rows"] == 1
    ids[0] = f"train:000000:{m.FRAMES}"
    with pytest.raises(ValueError, match="frame/particle"):
        m.check_history(history, dict(arm="base", seed=0), manifest)


class SandSpecialization(ast.NodeTransformer):
    def visit_If(self, node):
        if ast.unparse(node.test) == "DATASET == 'Goop'":
            return None
        return self.generic_visit(node)

    def visit_Name(self, node):
        values = dict(DATASET="Sand", FRAMES=320, PARTICLE_TYPE=6)
        return ast.copy_location(ast.Constant(values[node.id]), node) if node.id in values else node

    def visit_JoinedStr(self, node):
        if not any(isinstance(child, ast.Name) and child.id == "DATASET" for child in ast.walk(node)):
            return node
        return ast.copy_location(ast.Constant(eval(compile(ast.Expression(node), "<synthetic specialization>", "eval"), {"DATASET": "Sand"})), node)


def test_sand_endpoint_and_history_match_frozen_v1_ast(configured):
    for function in ("check_history", "verify_job"):
        tree = ast.parse(inspect.getsource(getattr(m, function)))
        normalized = SandSpecialization().visit(tree)
        original = ast.parse(inspect.getsource(getattr(m.V, function) if function != "verify_job" else module(HERE / "supervise_sand_graph_support_science.py", "comparison_v1").verify_job))
        assert ast.dump(normalized, include_attributes=False) == ast.dump(original, include_attributes=False)
    # Numerical checkpoint/Adam/RNG checks after the schema assertion are exact.
    assert ast.dump(ast.Module(body=ast.parse(inspect.getsource(m.check_payload)).body[0].body[1:], type_ignores=[])) == ast.dump(
        ast.Module(body=ast.parse(inspect.getsource(m.V.check_payload)).body[0].body[1:], type_ignores=[]))


def test_goop_context_lineage_rejected_before_checkpoint_open(tmp_path, configured, monkeypatch):
    if configured != "Goop":
        return
    import torch
    job = m.C.SCHEDULE[0]
    config = goop_helpers.config(job)
    config["log_every"] = 100
    config["data"]["auxiliary_report_sha256"] = "0" * 64
    monkeypatch.setattr(m, "read", lambda path: config if Path(path).name == "protocol.json" else {})
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Wrong context must fail before checkpoint opening"))
    with pytest.raises(ValueError, match="Goop context/source"):
        m.verify_job(tmp_path, job, dict(exit_code=0, signals=[]), {}, "protocol", goop_helpers.UUIDS[0])


@pytest.mark.parametrize("dataset", ["Sand", "Goop"])
def test_default_has_no_scientific_import_or_dispatch(dataset, monkeypatch, capsys):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "numpy", "scipy", "gns", "research"):
            pytest.fail("Scientific import not allowed in default description")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(m, "configure", lambda args: pytest.fail("Default cannot configure execution"))
    assert m.main(["--dataset", dataset]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "description_only"
