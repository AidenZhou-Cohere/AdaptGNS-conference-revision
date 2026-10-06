"""Tiny synthetic CPU networks/mock CUDA RNG only; no real model/data/launch."""
import copy
import contextlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent
PATH = ROOT / "train_goop3d_graph_support_cuda_v2.py"
spec = importlib.util.spec_from_file_location("_goop3d_graph_support_test", PATH)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def test_default_mode_has_no_scientific_imports():
    code = """import builtins, runpy, sys
original = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'numpy', 'scipy', 'gns', 'research'):
        raise AssertionError('Scientific import in description: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = restricted
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    process = subprocess.run([sys.executable, "-I", "-c", code, str(PATH)], text=True, capture_output=True, check=True)
    result = json.loads(process.stdout)
    assert result["schema"] == "adaptgns_goop3d_graph_support_cuda_training_v2"
    assert result["endpoint_selected"] is False


def argv(endpoint=37):
    return ["--execute", "--repo", "r", "--train-manifest", "t", "--admission", "a", "--structural-report", "s",
            "--acquisition-report", "q", "--context-semantics", "c", "--auxiliary-report", "u", "--protocol", "p", "--output-dir", "o", "--arm", "base", "--seed", "0", "--updates", str(endpoint)]


def test_explicit_endpoint_is_required_and_not_hardcoded():
    assert entry.parse_args(argv(37)).updates == 37
    assert entry.parse_args(argv(50000)).updates == 50000
    assert entry.parse_args(argv(100000)).updates == 100000
    for extra in (["--objective", "nll"], ["--seed", "3"], ["--updates", "1"], ["--arm", "expanded25"], ["--stop-after", "2"], ["--clear-stale-lock"]):
        with pytest.raises(SystemExit):
            entry.parse_args(argv() + extra)
    assert entry.graph_exposure_config("mix")["mixture_probability_per_example"] == .5
    assert entry.graph_exposure_config("base")["mixture_probability_per_example"] == 0


def test_partial_json_artifact_is_preserved_instead_of_overwritten(tmp_path):
    path = tmp_path / "status.json"
    partial = tmp_path / "status.json.tmp"
    partial.write_bytes(b"retained partial evidence")
    with pytest.raises(FileExistsError):
        entry.atomic_json(path, {"state": "new"})
    assert partial.read_bytes() == b"retained partial evidence" and not path.exists()


@pytest.fixture
def synthetic(monkeypatch):
    repo = ROOT.parents[2] / "outputs" / "AdaptGNS"
    helpers, support = entry.load_helpers(repo)
    np, torch = helpers.np, helpers.torch
    torch.set_num_threads(1)
    from gns.learned_simulator import LearnedSimulator
    def make(seed=0):
        torch.manual_seed(seed)
        stats = {name: {"mean": torch.tensor([.01, -.02, .03]), "std": torch.tensor([.3, .4, .5])}
                 for name in ("velocity", "acceleration")}
        model = LearnedSimulator(particle_dimensions=3, nnode_in=37, nedge_in=4, latent_dim=8,
            nmessage_passing_steps=1, nmlp_layers=1, mlp_hidden_dim=8, connectivity_radius=.025,
            boundaries=[[.1, .9]] * 3, normalization_stats=stats, nparticle_types=9,
            particle_type_embedding_size=16, max_num_neighbors=128, device="cpu",
            uncertainty_parameterization="variance", variance_floor=1e-6, detach_variance_features=True,
            radius_backend="scipy_host")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                     weight_decay=0., foreach=False, fused=False)
        return model, optimizer
    points = np.asarray([[.2+.022*x, .2+.022*y, .2+.022*z] for x in range(2) for y in range(2) for z in range(2)], dtype=np.float32)
    positions = np.broadcast_to(points, (7, 8, 3)).copy()
    positions[:, :, 0] += np.arange(7, dtype=np.float32)[:, None]*.0001
    one = ((positions[:6].transpose(1, 0, 2), np.full(8, 7, dtype=np.int64), 8), positions[6])
    batch = entry.unpack_batch(helpers, [one, one])
    generator = torch.Generator(device="cpu").manual_seed(4000)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: generator.get_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value, device: generator.set_state(value))
    return SimpleNamespace(repo=repo, helpers=helpers, support=support, torch=torch, np=np, make=make, batch=batch)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_native_base_output_target_loss_gradients_and_first_adam_step_are_exact(synthetic, seed):
    s = synthetic
    left, left_opt = s.make(seed)
    right, right_opt = s.make(seed)
    noise = s.helpers.host_noise(s.batch[0].shape, s.batch[1], seed, 0)
    original = s.helpers.forward_batch(left, s.batch, noise, "cpu")
    exposed = s.support.forward_batch(right, s.batch, noise, "cpu", seed, 0, "base")
    assert all(s.torch.equal(a, b) for a, b in zip(original, exposed[:3]))
    mask = s.batch[1] != 3
    losses = []
    for model, optimizer, values in ((left, left_opt, original), (right, right_opt, exposed)):
        loss = entry.guarded_update(s.helpers, model, optimizer, values[0], values[1], values[2], mask, "faithful", 1, 37)
        losses.append(loss)
    assert s.torch.equal(*losses)
    assert entry.tree_equal(s.torch, left.state_dict(), right.state_dict())
    assert entry.tree_equal(s.torch, left_opt.state_dict(), right_opt.state_dict())
    for a, b in zip(left.parameters(), right.parameters()):
        assert s.torch.equal(a.grad, b.grad)
    assert all(record["selected_optional_pairs"] == 0 for record in exposed[3])


@pytest.mark.parametrize("step", [0, 1, 64, 99999])
def test_graph_rng_does_not_change_host_noise_or_sample_stream(synthetic, step):
    s = synthetic
    samples = s.helpers.sample_indices(2, step, 395000, 2)
    noise = s.helpers.host_noise(s.batch[0].shape, s.batch[1], 2, step)
    before = s.torch.get_rng_state().clone()
    coin, rng, cm, pm = s.support.graph_rng(2, step, 0)
    first = rng.permutation(20)
    other_coin, other_rng, _, _ = s.support.graph_rng(2, step, 0)
    assert coin == other_coin and s.np.array_equal(first, other_rng.permutation(20))
    assert cm == [20261005, 2, step, 0, 4409] and pm == [20261005, 2, step, 0, 5501]
    assert s.torch.equal(before, s.torch.get_rng_state())
    assert samples == s.helpers.sample_indices(2, step, 395000, 2)
    assert s.torch.equal(noise, s.helpers.host_noise(s.batch[0].shape, s.batch[1], 2, step))


def test_mix_uses_exact_annulus_budget_and_preserves_native_prefix_and_partition(synthetic):
    s = synthetic
    model, _ = s.make()
    step = next(index for index in range(100) if any(s.support.graph_rng(0, index, slot)[0] for slot in (0, 1)))
    sequence, types, counts, _ = s.batch
    _, native, _ = model._encoder_preprocessor(sequence, counts, types, None)
    edges, ledger = s.support.append_optional_edges(sequence, counts, native, .025, 0, step, "mix")
    assert s.torch.equal(edges[:, :native.shape[1]], native)
    assert any(row["expanded"] and row["selected_optional_pairs"] > 0 for row in ledger)
    assert all(row["selected_optional_pairs"] == (row["optional_budget_if_exposed"] if row["expanded"] else 0) for row in ledger)
    assert edges.shape[1] == native.shape[1] + 2*sum(row["selected_optional_pairs"] for row in ledger)
    membership = s.torch.repeat_interleave(s.torch.arange(2), counts)
    assert s.torch.equal(membership[edges[0]], membership[edges[1]])
    assert int((edges[0] == edges[1]).sum()) == int((native[0] == native[1]).sum())


def test_cap_active_zero_budget_returns_original_native_tensor(synthetic):
    s = synthetic
    model, _ = s.make()
    positions = s.torch.zeros(132, 6, 3)
    types, counts = s.torch.full((132,), 7), s.torch.tensor([132])
    _, native, _ = model._encoder_preprocessor(positions, counts, types, None)
    edges, ledger = s.support.append_optional_edges(positions, counts, native, .025, 0, 0, "mix")
    assert edges is native and ledger[0]["receivers_above_native_cap"] == 132
    assert ledger[0]["annulus_pairs"] == ledger[0]["selected_optional_pairs"] == 0
    assert int((edges[0] == edges[1]).sum()) < 132


def test_target_change_cannot_change_graph_or_network_output(synthetic):
    s = synthetic
    model, _ = s.make()
    noise = s.torch.zeros_like(s.batch[0])
    first = s.support.forward_batch(model, s.batch, noise, "cpu", 0, 0, "mix")
    changed = (*s.batch[:3], s.batch[3] + .01)
    second = s.support.forward_batch(model, changed, noise, "cpu", 0, 0, "mix")
    assert s.torch.equal(first[0], second[0]) and s.torch.equal(first[1], second[1])
    assert first[3] == second[3] and not s.torch.equal(first[2], second[2])


def step_with_history(s, model, optimizer, step, arm):
    noise = s.helpers.host_noise(s.batch[0].shape, s.batch[1], 0, step)
    output = s.support.forward_batch(model, s.batch, noise, "cpu", 0, step, arm)
    optimizer.param_groups[0]["lr"] = entry.learning_rate(step, 37)
    optimizer.zero_grad(set_to_none=True)
    entry.guarded_update(s.helpers, model, optimizer, *output[:3], s.batch[1] != 3, "faithful", step + 1, 37)
    return {"completed_steps": step + 1, "absolute_schedule_step": step,
            "frame_ids": ["synthetic:000000:6", "synthetic:000000:6"],
            "noise_sha256": s.support.state_hash(noise.numpy()), "examples": output[3]}


def test_new_schema_refuses_old_lineages_and_opposite_arm_and_restores_exactly(synthetic):
    s = synthetic
    model, optimizer = s.make()
    record = step_with_history(s, model, optimizer, 0, "mix")
    config = {"objective": "faithful", "arm": "mix", "seed": 0, "updates": 37,
              "graph_exposure": entry.graph_exposure_config("mix")}
    history = {"training": [], "graph_updates": [record], "elapsed_seconds": 1.0}
    payload = entry.checkpoint_payload(s.helpers, model, optimizer, config, 1, history, "mock-cuda")
    assert payload["cuda_goop3d_graph_support_schema"] == entry.SCHEMA
    assert "cuda_goop_training_schema" not in payload and "graph_support_schema" not in payload
    restored, restored_opt = s.make()
    completed, _ = entry.restore_payload(s.helpers, payload, restored, restored_opt, config, "mock-cuda")
    assert completed == 1 and entry.tree_equal(s.torch, model.state_dict(), restored.state_dict())
    assert entry.tree_equal(s.torch, optimizer.state_dict(), restored_opt.state_dict())
    for mutation in (lambda p: p.update(cuda_goop_training_schema="old"),
                     lambda p: p.update(cuda_sand_graph_support_schema="old"),
                     lambda p: p.update(graph_support_schema=1),
                     lambda p: p["history"]["graph_updates"][0].update(absolute_schedule_step=1)):
        modified = copy.deepcopy(payload)
        mutation(modified)
        with pytest.raises(ValueError):
            entry.restore_payload(s.helpers, modified, restored, restored_opt, config, "mock-cuda")
    opposite = {**config, "arm": "base", "graph_exposure": entry.graph_exposure_config("base")}
    with pytest.raises(ValueError, match="lineage"):
        entry.restore_payload(s.helpers, payload, restored, restored_opt, opposite, "mock-cuda")


def test_malformed_graph_counts_hashes_and_rng_material_are_rejected(synthetic):
    s = synthetic
    model, optimizer = s.make()
    record = step_with_history(s, model, optimizer, 0, "mix")
    history = {"training": [], "graph_updates": [record], "elapsed_seconds": 1.0}
    entry.validate_graph_history(history, 1, "mix", 0)
    for mutation in (lambda r: r["examples"][0].update(selected_optional_pairs=999),
                     lambda r: r["examples"][0].update(native_edge_sha256="bad"),
                     lambda r: r["examples"][0].update(coin_seed_material=[0]),
                     lambda r: r.update(noise_sha256="bad")):
        changed = copy.deepcopy(history)
        mutation(changed["graph_updates"][0])
        with pytest.raises(ValueError):
            entry.validate_graph_history(changed, 1, "mix", 0)


def test_tiny_mock_interrupted_run_resumes_only_its_own_declared_endpoint(synthetic, tmp_path, monkeypatch):
    s = synthetic
    args = SimpleNamespace(repo=s.repo, train_manifest=tmp_path / "data" / "train.json",
        admission=tmp_path / "admission.json", structural_report=tmp_path / "structural.json",
        protocol=tmp_path / "protocol.md", output_dir=tmp_path / "output", objective="faithful",
        arm="mix", seed=0, updates=3, checkpoint_every=1, log_every=1, resume=False)
    args.output_dir.mkdir()
    for path in (args.train_manifest, args.admission, args.structural_report, args.protocol):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)
    info = {"trajectory_ids": ["synthetic:000000"], "manifest_sha256": entry.sha(args.train_manifest),
            "admission_sha256": entry.sha(args.admission), "structural_report_sha256": entry.sha(args.structural_report)}
    monkeypatch.setattr(entry, "load_admitted_dataset", lambda args, helpers: ([None, None], {}, info))
    monkeypatch.setattr(entry, "reverify_goop3d_evidence_and_auxiliaries", lambda args, helpers: None)
    def indices(seed, step, total, count):
        if step == 2:
            raise KeyboardInterrupt("synthetic interruption")
        return [0, 1]
    monkeypatch.setattr(s.helpers, "sample_indices", indices)
    monkeypatch.setattr(entry, "unpack_batch", lambda helpers, examples: s.batch)
    monkeypatch.setattr(s.helpers, "frame_identity", lambda train, ids, index: f"synthetic:000000:{6+index}")
    monkeypatch.setattr(s.helpers, "build_simulator", lambda *args, **kwargs: s.make()[0])
    monkeypatch.setattr(s.helpers.data_loader, "load_manifest_data", lambda *args, **kwargs: [])
    monkeypatch.setattr(s.torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(s.torch.cuda, "manual_seed", lambda seed: None)
    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        entry.run_training(args, s.helpers, s.support, {"pid": 123}, "cpu", {"scope": "synthetic_mock"})
    first_pointer = json.loads((args.output_dir / "latest.json").read_text())
    assert first_pointer["completed_steps"] == 2
    initial = s.torch.load(args.output_dir / "checkpoint-000000000.pt", weights_only=True)
    assert initial["completed_steps"] == 0 and initial["optimizer_state"]["state"] == {}
    assert initial["history"]["graph_updates"] == []
    assert initial["run_config"]["optimizer"]["decay_updates"] == 3
    args.resume = True
    monkeypatch.setattr(s.helpers, "sample_indices", lambda seed, step, total, count: [0, 1])
    entry.run_training(args, s.helpers, s.support, {"pid": 124}, "cpu", {"scope": "synthetic_mock"})
    final = s.torch.load(args.output_dir / "checkpoint-000000003.pt", weights_only=True)
    assert final["completed_steps"] == len(final["history"]["graph_updates"]) == 3
    assert [row["absolute_schedule_step"] for row in final["history"]["graph_updates"]] == [0, 1, 2]
    assert final["optimizer_state"]["param_groups"][0]["lr"] == 1e-5
    assert (args.output_dir / first_pointer["path"]).exists()
    assert len(list((args.output_dir / "attempts").glob("*/status.json"))) == 2
    status = json.loads((args.output_dir / "status.json").read_text())
    assert status["state"] == "complete" and status["committed_steps"] == status["requested_steps"] == 3
    args.updates = 4
    with pytest.raises(ValueError, match="exact matching"):
        entry.run_training(args, s.helpers, s.support, {"pid": 125}, "cpu", {"scope": "synthetic_mock"})


def synthetic_contract(count=3, particle_types=(2, 5)):
    import hashlib
    digest = lambda value: hashlib.sha256(str(value).encode()).hexdigest()
    admission = {'schema': entry.ADMISSION, 'status': 'admitted', 'issued_by': 'root', 'dataset': 'Goop-3D',
        'manifest_sha256': digest('manifest'), 'source_sha256': digest('source'), 'metadata_sha256': entry.METADATA_SHA,
        'converter_sha256': entry.CONVERTER_SHA, 'reader_sha256': entry.READER_SHA,
        'acquisition_report_sha256': digest('acquisition'), 'context_semantics_sha256': entry.CONTEXT_SEMANTICS_SHA,
        'auxiliary_validator_sha256': entry.AUXILIARY_VALIDATOR_SHA,
        'structural_report_sha256': digest('structural'), 'auxiliary_report_sha256': digest('auxiliary'),
        'frames_per_trajectory': 301, 'dimension': 3, 'record_count': count, 'particle_type_ids': list(particle_types),
        'eligible_six_frame_histories': count * 295, 'position_dtype': '<f4', 'particle_type_dtype': '<i8',
        'auxiliary_policy': {'name': 'step_context', 'handling': 'preserved_and_excluded_from_model_inputs',
                            'reason': entry.OMISSION_REASON, 'official_commit': entry.OFFICIAL_COMMIT}}
    records, offset = [], 0
    for index in range(count):
        p = {'path': f'train/position_{index:06d}.npy', 'shape': [301, 2, 3], 'dtype': '<f4', 'size_bytes': 7352, 'sha256': digest(('p', index))}
        t = {'path': f'train/type_{index:06d}.npy', 'shape': [2], 'dtype': '<i8', 'size_bytes': 144, 'sha256': digest(('t', index))}
        length = 1000 if index < count - 1 else entry.OFFICIAL_TRAIN['size_bytes'] - offset - 16
        record = {'id': f'train:{index:06d}', 'source_index': index, 'source_key': [123 + index],
                  'source_offset_bytes': offset, 'record_payload_bytes': length, 'record_payload_sha256': digest(('payload', index)),
                  'positions': p, 'particle_types': t, 'trajectory_content_sha256': digest(p['sha256'] + ':' + t['sha256'])}
        if index % 2 == 0:
            record['step_context'] = {'path': f'train/step_context_{index:06d}.npy', 'shape': [301, 2], 'dtype': '<f4',
                'size_bytes': 2536, 'sha256': digest(('aux', index)), 'use': entry.AUXILIARY_USE}
        records.append(record)
        offset += length + 16
    manifest = {'format': 'gns-trajectory-manifest', 'version': 1, 'split': 'train', 'dataset': 'Goop-3D',
        'source': {**entry.OFFICIAL_TRAIN, 'sha256': admission['source_sha256'], 'record_count': count,
                   'acquisition_report_sha256': admission['acquisition_report_sha256']},
        'metadata': {'dim': 3, 'sequence_length': 300, 'default_connectivity_radius': .025},
        'metadata_sha256': entry.METADATA_SHA, 'record_count': count, 'records': records,
        'converter_sha256': entry.CONVERTER_SHA, 'reader_sha256': entry.READER_SHA}
    return manifest, admission


@pytest.mark.parametrize('count,types', [(1, [7]), (3, [2, 5]), (8, [1, 3, 6, 8])])
def test_actual_count_types_source_keys_and_optional_auxiliary_are_not_inferred(count, types):
    manifest, admission = synthetic_contract(count, types)
    entry.validate_manifest_contract(manifest, admission)
    assert manifest['source']['record_count'] == count and admission['particle_type_ids'] == types
    assert manifest['records'][0]['source_key'] == [123]


@pytest.mark.parametrize('mutation', [
    lambda m,a: m.update(dataset='Goop'),
    lambda m,a: a.update(record_count=1000),
    lambda m,a: a.update(eligible_six_frame_histories=1000),
    lambda m,a: a.update(particle_type_ids=[3]),
    lambda m,a: a.update(particle_type_ids=[9]),
    lambda m,a: a.update(frames_per_trajectory=401),
    lambda m,a: a.update(auxiliary_validator_sha256='0'*64),
    lambda m,a: a['auxiliary_policy'].update(reason='all_values_nan'),
    lambda m,a: m['metadata'].update(context_mean=[0.]),
    lambda m,a: m['records'][0]['positions'].update(shape=[301, 2, 2]),
    lambda m,a: m['records'][0]['particle_types'].update(shape=[]),
    lambda m,a: m['records'][0]['step_context'].update(shape=[300, 2]),
    lambda m,a: m['records'][0]['positions'].update(path='../escape.npy'),
    lambda m,a: m['records'][1].update(source_offset_bytes=0),
    lambda m,a: m['records'][-1].update(record_payload_bytes=1000),
    lambda m,a: m['source'].update(CRC_verified=False),
])
def test_wrong_or_partial3d_source_contract_is_refused(mutation):
    manifest, admission = synthetic_contract()
    mutation(manifest, admission)
    with pytest.raises(ValueError):
        entry.validate_manifest_contract(manifest, admission)


def endpoint_fixture(tmp_path, endpoint=37):
    _, admission = synthetic_contract()
    protocol = tmp_path / 'protocol.md'
    protocol.write_text('Synthetic prospective endpoint only')
    args = SimpleNamespace(updates=endpoint, checkpoint_every=10, log_every=2, protocol=protocol)
    admission.update(prospective_endpoint_updates=endpoint, endpoint_selection_basis=entry.ENDPOINT_BASIS,
        paired_arm_seed_schedule=entry.PAIRED_SCHEDULE, checkpoint_every=10, log_every=2,
        trainer_sha256=entry.sha(PATH), graph_adapter_sha256=entry.GRAPH_SHA, protocol_sha256=entry.sha(protocol),
        selection_evidence_sha256=['e'*64], initialization='fresh_no_probe_or_parent_checkpoint')
    return args, admission


@pytest.mark.parametrize('endpoint', [2, 37, 15000, 50000, 100000])
def test_endpoint_and_lr_are_prospective_cohort_parameters(tmp_path, endpoint):
    args, admission = endpoint_fixture(tmp_path, endpoint)
    entry.validate_prospective_endpoint(args, admission)
    assert entry.learning_rate(0, endpoint) == 1e-4
    assert entry.learning_rate(endpoint - 1, endpoint) == 1e-5
    with pytest.raises(ValueError):
        entry.learning_rate(endpoint, endpoint)


@pytest.mark.parametrize('change', [
    lambda a: a.update(prospective_endpoint_updates=100000),
    lambda a: a.update(endpoint_selection_basis='best_validation_loss'),
    lambda a: a.update(paired_arm_seed_schedule=[{'arm':'base','seed':0}]),
    lambda a: a.update(protocol_sha256='0'*64),
    lambda a: a.update(graph_adapter_sha256='0'*64),
    lambda a: a.update(trainer_sha256='0'*64),
    lambda a: a.update(initialization='first_record_probe'),
    lambda a: a.update(selection_evidence_sha256=[]),
    lambda a: a.update(checkpoint_every=1),
])
def test_endpoint_selection_and_probe_lineage_cannot_be_changed(tmp_path, change):
    args, admission = endpoint_fixture(tmp_path)
    change(admission)
    with pytest.raises(ValueError, match='Prospective'):
        entry.validate_prospective_endpoint(args, admission)


def test_auxiliary_recheck_handles_present_absent_and_noncanonical_nan_bits(tmp_path):
    import numpy as np
    numeric = tmp_path / 'train'
    numeric.mkdir()
    args = SimpleNamespace(train_manifest=tmp_path / 'train.json')
    values = np.zeros((301, 2), dtype='<f4')
    values.view('<u4')[0] = [0x7fc00001, 0x80000000]
    path = numeric / 'step_context_000000.npy'
    np.save(path, values)
    descriptor = dict(path='train/step_context_000000.npy', shape=[301, 2], dtype='<f4', size_bytes=path.stat().st_size,
                      sha256=entry.sha(path), use=entry.AUXILIARY_USE)
    records = [dict(id='train:000000', source_index=0, positions=dict(shape=[301, 2, 3]), step_context=descriptor),
               dict(id='train:000001', source_index=1, positions=dict(shape=[301, 2, 3]))]
    spec = importlib.util.spec_from_file_location('synthetic_3d_census', ROOT / 'audit_goop3d_auxiliary.py')
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    rows = [audit.census_record(tmp_path, 'train', record, np)[0] for record in records]
    entry.verify_goop3d_auxiliary_arrays(args, SimpleNamespace(np=np), {'records': records}, rows)
    assert rows[1] == dict(id='train:000001', source_index=1, present=False)
    assert '7fc00001' in rows[0]['unique_float32_bits_hex'] and '80000000' in rows[0]['unique_float32_bits_hex']
    values.view('<u4')[0, 0] = 0x7fc00002
    np.save(path, values)
    descriptor.update(sha256=entry.sha(path), size_bytes=path.stat().st_size)
    with pytest.raises(ValueError, match='rawbit'):
        entry.verify_goop3d_auxiliary_arrays(args, SimpleNamespace(np=np), {'records': records}, rows)


def test_trustworthy_lock_rng_history_pointer_and_cuda_checks_unchanged_ast():
    import ast
    import hashlib
    base = ROOT / 'train_goop_graph_support_cuda.py'
    assert hashlib.sha256(base.read_bytes()).hexdigest() == entry.BASE_TRAINER_SHA
    old = {n.name:n for n in ast.parse(base.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    new = {n.name:n for n in ast.parse(PATH.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for name in ('RunLock','atomic_json','graph_exposure_config','validate_graph_history','capture_rng','restore_rng','tree_equal','read_pointer','configure_cuda'):
        assert ast.dump(old[name]) == ast.dump(new[name]), name
