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
PATH = ROOT / "train_goop_graph_support_cuda.py"
spec = importlib.util.spec_from_file_location("_goop_graph_support_test", PATH)
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
    assert result["schema"] == "adaptgns_goop_graph_support_cuda_training_v1"
    assert result["endpoint_selected"] is False


def test_only_fixed_faithful_100k_three_seed_base_mix_cohort_is_accepted():
    args = ["--execute", "--repo", "r", "--train-manifest", "t", "--admission", "a", "--structural-report", "s",
            "--acquisition-report", "q", "--context-semantics", "c", "--auxiliary-report", "u", "--protocol", "p", "--output-dir", "o", "--arm", "base", "--seed", "0", "--updates", "100000"]
    parsed = entry.parse_args(args)
    assert parsed.objective == "faithful" and parsed.checkpoint_every == 10000
    for extra in (["--objective", "nll"], ["--seed", "3"], ["--updates", "512"], ["--arm", "risk"], ["--clear-stale-lock"]):
        with pytest.raises(SystemExit):
            entry.parse_args(args + extra)
    assert entry.parse_args(args + ["--stop-after", "512"]).updates == 100000
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
        stats = {name: {"mean": torch.tensor([.01, -.02]), "std": torch.tensor([.3, .4])}
                 for name in ("velocity", "acceleration")}
        model = LearnedSimulator(particle_dimensions=2, nnode_in=16, nedge_in=3, latent_dim=8,
            nmessage_passing_steps=1, nmlp_layers=1, mlp_hidden_dim=8, connectivity_radius=.015,
            boundaries=[[.1, .9], [.1, .9]], normalization_stats=stats, nparticle_types=9,
            particle_type_embedding_size=2, max_num_neighbors=128, device="cpu",
            uncertainty_parameterization="variance", variance_floor=1e-6, detach_variance_features=True,
            radius_backend="scipy_host")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                     weight_decay=0., foreach=False, fused=False)
        return model, optimizer
    points = np.asarray([[.2+.013*x, .2+.013*y] for x in range(3) for y in range(3)], dtype=np.float32)
    positions = np.broadcast_to(points, (7, 9, 2)).copy()
    positions[:, :, 0] += np.arange(7, dtype=np.float32)[:, None]*.0001
    one = ((positions[:6].transpose(1, 0, 2), np.full(9, 7, dtype=np.int64), 9), positions[6])
    batch = helpers.unpack_batch([one, one])
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
        loss = entry.guarded_update(s.helpers, model, optimizer, values[0], values[1], values[2], mask, "faithful", 1)
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
    edges, ledger = s.support.append_optional_edges(sequence, counts, native, .015, 0, step, "mix")
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
    positions = s.torch.zeros(132, 6, 2)
    types, counts = s.torch.full((132,), 7), s.torch.tensor([132])
    _, native, _ = model._encoder_preprocessor(positions, counts, types, None)
    edges, ledger = s.support.append_optional_edges(positions, counts, native, .015, 0, 0, "mix")
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
    optimizer.param_groups[0]["lr"] = s.helpers.learning_rate(step)
    optimizer.zero_grad(set_to_none=True)
    entry.guarded_update(s.helpers, model, optimizer, *output[:3], s.batch[1] != 3, "faithful", step + 1)
    return {"completed_steps": step + 1, "absolute_schedule_step": step,
            "frame_ids": ["synthetic:000000:6", "synthetic:000000:6"],
            "noise_sha256": s.support.bridge.full.state_hash(noise.numpy()), "examples": output[3]}


def test_new_schema_refuses_old_lineages_and_opposite_arm_and_restores_exactly(synthetic):
    s = synthetic
    model, optimizer = s.make()
    record = step_with_history(s, model, optimizer, 0, "mix")
    config = {"objective": "faithful", "arm": "mix", "seed": 0, "updates": 100000,
              "graph_exposure": entry.graph_exposure_config("mix")}
    history = {"training": [], "graph_updates": [record], "elapsed_seconds": 1.0}
    payload = entry.checkpoint_payload(s.helpers, model, optimizer, config, 1, history, "mock-cuda")
    assert payload["cuda_goop_graph_support_schema"] == entry.SCHEMA
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


def test_tiny_mock_run_starts_at_zero_and_resumes_only_its_own_graph_history(synthetic, tmp_path, monkeypatch):
    s = synthetic
    args = SimpleNamespace(repo=s.repo, train_manifest=tmp_path / "data" / "train.json",
        admission=tmp_path / "admission.json", structural_report=tmp_path / "structural.json",
        protocol=tmp_path / "protocol.md", output_dir=tmp_path / "output", objective="faithful",
        arm="mix", seed=0, updates=100000, checkpoint_every=10000, log_every=1, resume=False, stop_after=2)
    args.output_dir.mkdir()
    for path in (args.train_manifest, args.admission, args.structural_report, args.protocol):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)
    info = {"trajectory_ids": ["synthetic:000000"], "manifest_sha256": entry.sha(args.train_manifest),
            "admission_sha256": entry.sha(args.admission), "structural_report_sha256": entry.sha(args.structural_report)}
    monkeypatch.setattr(entry, "load_admitted_dataset", lambda args, helpers: ([None, None], {}, info))
    monkeypatch.setattr(entry, "reverify_goop_evidence_and_auxiliaries", lambda args, helpers: None)
    monkeypatch.setattr(s.helpers, "sample_indices", lambda seed, step, total, count: [0, 1])
    monkeypatch.setattr(s.helpers, "unpack_batch", lambda examples: s.batch)
    monkeypatch.setattr(s.helpers, "frame_identity", lambda train, ids, index: f"synthetic:000000:{6+index}")
    monkeypatch.setattr(s.helpers, "build_simulator", lambda *args, **kwargs: s.make()[0])
    monkeypatch.setattr(s.helpers.data_loader, "load_manifest_data", lambda *args, **kwargs: [])
    monkeypatch.setattr(s.torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(s.torch.cuda, "manual_seed", lambda seed: None)
    entry.run_training(args, s.helpers, s.support, {"pid": 123}, "cpu", {"scope": "synthetic_mock"})
    first_pointer = json.loads((args.output_dir / "latest.json").read_text())
    assert first_pointer["completed_steps"] == 2
    initial = s.torch.load(args.output_dir / "checkpoint-000000000.pt", weights_only=True)
    assert initial["completed_steps"] == 0 and initial["optimizer_state"]["state"] == {}
    assert initial["history"]["graph_updates"] == []
    assert initial["run_config"]["graph_exposure"] == entry.graph_exposure_config("mix")
    args.resume, args.stop_after = True, 3
    entry.run_training(args, s.helpers, s.support, {"pid": 124}, "cpu", {"scope": "synthetic_mock"})
    final = s.torch.load(args.output_dir / "checkpoint-000000003.pt", weights_only=True)
    assert final["completed_steps"] == len(final["history"]["graph_updates"]) == 3
    assert [row["absolute_schedule_step"] for row in final["history"]["graph_updates"]] == [0, 1, 2]
    assert (args.output_dir / first_pointer["path"]).exists()
    assert len(list((args.output_dir / "attempts").glob("*/status.json"))) == 2
    status = json.loads((args.output_dir / "status.json").read_text())
    assert status["state"] == "planned_stop_incomplete" and status["committed_steps"] == 3


def synthetic_contract():
    import hashlib
    digest = lambda value: hashlib.sha256(str(value).encode()).hexdigest()
    records, offset = [], 0
    for index in range(1000):
        p = {'path': f'train/position_{index:06d}.npy', 'shape': [401, 2, 2],
             'dtype': '<f4', 'size_bytes': 6544, 'sha256': digest(('p', index))}
        t = {'path': f'train/type_{index:06d}.npy', 'shape': [2],
             'dtype': '<i8', 'size_bytes': 144, 'sha256': digest(('t', index))}
        a = {'path': f'train/step_context_{index:06d}.npy', 'shape': [401, 1],
             'dtype': '<f4', 'size_bytes': 1732, 'sha256': digest('aux'), 'use': entry.AUXILIARY_USE}
        length = 1000 if index < 999 else entry.OFFICIAL_TRAIN_SOURCE['size_bytes'] - offset - 16
        records.append({'id': f'train:{index:06d}', 'source_index': index, 'source_key': [index],
                        'source_offset_bytes': offset, 'record_payload_bytes': length,
                        'record_payload_sha256': digest(('payload', index)), 'positions': p,
                        'particle_types': t, 'step_context': a,
                        'trajectory_content_sha256': digest(p['sha256']+':'+t['sha256'])})
        offset += length + 16
    manifest = {'format': 'gns-trajectory-manifest', 'version': 1, 'split': 'train', 'dataset': 'Goop',
                'source': copy.deepcopy(entry.OFFICIAL_TRAIN_SOURCE), 'metadata': {'dim': 2, 'sequence_length': 400,
                    'default_connectivity_radius': .015, 'dt': .0025, 'bounds': [[.1, .9], [.1, .9]]},
                'metadata_sha256': entry.METADATA_SHA, 'record_count': 1000, 'records': records,
                'converter_sha256': entry.CONVERTER_SHA, 'reader_sha256': entry.READER_SHA}
    admission = {'schema': entry.ADMISSION, 'status': 'admitted', 'issued_by': 'root', 'dataset': 'Goop',
                 'manifest_sha256': entry.MANIFEST_SHA, 'metadata_sha256': entry.METADATA_SHA,
                 'converter_sha256': entry.CONVERTER_SHA, 'reader_sha256': entry.READER_SHA,
                 'acquisition_report_sha256': entry.ACQUISITION_SHA, 'context_semantics_sha256': entry.CONTEXT_SEMANTICS_SHA,
                 'structural_report_sha256': digest('structural'), 'auxiliary_report_sha256': entry.AUXILIARY_REPORT_SHA,
                 'frames_per_trajectory': 401, 'record_count': 1000, 'particle_type_ids': [7],
                 'position_dtype': '<f4', 'particle_type_dtype': '<i8',
                 'auxiliary_policy': {'name': 'step_context', 'handling': 'preserved_and_excluded_from_model_inputs',
                                     'reason': entry.OMISSION_REASON, 'official_commit': entry.OFFICIAL_COMMIT}}
    return manifest, admission


def test_exact_goop_admission_and_preserved_full_type_vectors():
    manifest, admission = synthetic_contract()
    entry.validate_manifest_contract(manifest, admission)
    assert manifest['records'][0]['particle_types']['shape'] == [2]
    for mutation in (lambda m, a: m.update(dataset='Sand'),
                     lambda m, a: a.update(schema='adaptgns_sand_training_admission_v1'),
                     lambda m, a: a.update(particle_type_ids=[6]),
                     lambda m, a: a.update(frames_per_trajectory=320),
                     lambda m, a: a.update(record_count=999),
                     lambda m, a: m['records'][0]['particle_types'].update(shape=[]),
                     lambda m, a: m['records'][0]['particle_types'].update(dtype='<i4'),
                     lambda m, a: m['records'][0]['positions'].update(shape=[400, 2, 2]),
                     lambda m, a: m['records'][0]['positions'].update(dtype='<f8')):
        changed, changed_admission = copy.deepcopy(manifest), copy.deepcopy(admission)
        mutation(changed, changed_admission)
        with pytest.raises(ValueError):
            entry.validate_manifest_contract(changed, changed_admission)


def test_context_omission_requires_official_semantics_not_nan_inference():
    manifest, admission = synthetic_contract()
    for mutation in (lambda m, a: a.pop('auxiliary_policy'),
                     lambda m, a: a['auxiliary_policy'].update(reason='all_values_nan'),
                     lambda m, a: a.update(context_semantics_sha256='0'*64),
                     lambda m, a: m['metadata'].update(context_mean=[0.]),
                     lambda m, a: m['metadata'].update(context_std=[1.]),
                     lambda m, a: m['records'][0].pop('step_context'),
                     lambda m, a: m['records'][0]['step_context'].update(shape=[400, 1]),
                     lambda m, a: m['records'][0].update(material_property=0.5)):
        changed, changed_admission = copy.deepcopy(manifest), copy.deepcopy(admission)
        mutation(changed, changed_admission)
        with pytest.raises(ValueError):
            entry.validate_manifest_contract(changed, changed_admission)


def test_source_crc_reader_offsets_keys_hashes_and_paths_cannot_be_relaxed():
    manifest, admission = synthetic_contract()
    for mutation in (lambda m: m['source'].update(family='designsafe_published_npz'),
                     lambda m: m['source'].update(CRC_verified=False),
                     lambda m: m['source'].update(generation='changed'),
                     lambda m: m.update(reader_sha256='0'*64),
                     lambda m: m['records'][1].update(source_offset_bytes=0),
                     lambda m: m['records'][0].update(source_key=[1]),
                     lambda m: m['records'][-1].update(record_payload_bytes=1000),
                     lambda m: m['records'][0]['positions'].update(path='../escape.npy'),
                     lambda m: m['records'][0].update(logical_content_sha256='0'*64),
                     lambda m: m['records'][0].update(trajectory_content_sha256='0'*64)):
        changed = copy.deepcopy(manifest)
        mutation(changed)
        with pytest.raises(ValueError):
            entry.validate_manifest_contract(changed, admission)


def test_auxiliary_census_requires_all_rows_and_exact_preserved_bits(tmp_path, monkeypatch):
    import numpy as np
    manifest, admission = synthetic_contract()
    args = SimpleNamespace(**{key: tmp_path/(key+'.json') for key in (
        'train_manifest', 'structural_report', 'acquisition_report', 'context_semantics', 'auxiliary_report')})
    structure = {'schema': 'official_goop_numeric_preparation_v1', 'status': 'complete_structural_only',
                 'test_accessed': False, 'dataset': 'Goop', 'source_family': 'official_gns_tfrecord',
                 'metadata_sha256': entry.METADATA_SHA, 'reader_sha256': entry.READER_SHA,
                 'wrapper_sha256': entry.CONVERTER_SHA, 'acquisition_report_sha256': entry.ACQUISITION_SHA,
                 'splits': {'train': {'manifest_sha256': entry.MANIFEST_SHA, 'record_count': 1000,
                    'frame_lengths': [401], 'particle_type_ids': [7], 'forecast_horizons_after_six_frames': [395],
                    **{key: True for key in ('source_EOF_SHA256_verified', 'whole_object_crc32c_verified',
                       'TFRecord_length_and_payload_CRC32C_verified', 'all_match_official_parser_frame_count',
                       'all_source_array_bytes_preserved_exact')}}}}
    rows = [{'id': r['id'], 'source_index': r['source_index'], 'sha256': r['step_context']['sha256'],
             'shape': [401, 1], 'dtype': '<f4', 'elements': 401, 'nan_count': 1,
             'finite_count': 400, 'positive_infinity_count': 0, 'negative_infinity_count': 0,
             'unique_float32_bits_hex': ['00000000', '7fc00000']} for r in manifest['records']]
    census = {'schema': 'adaptgns_goop_auxiliary_census_v1', 'status': 'all_preserved_auxiliary_bytes_verified',
              'test_accessed': False, 'splits': {'train': {'manifest_sha256': entry.MANIFEST_SHA, 'record_count': 1000,
                  'context_mean_absent': True, 'context_std_absent': True,
                  'all_context_descriptors_and_bytes_verified': True, 'all_context_values_nan': False, 'records': rows}}}
    args.structural_report.write_text(json.dumps(structure)); args.auxiliary_report.write_text(json.dumps(census))
    pins = {args.train_manifest: entry.MANIFEST_SHA, args.structural_report: admission['structural_report_sha256'],
            args.acquisition_report: entry.ACQUISITION_SHA, args.context_semantics: entry.CONTEXT_SEMANTICS_SHA,
            args.auxiliary_report: admission['auxiliary_report_sha256']}
    pins.update({tmp_path/'goop_context_semantics_sources'/name: value for name, value in entry.CONTEXT_SOURCE_PINS.items()})
    actual_sha = entry.sha
    monkeypatch.setattr(entry, 'sha', lambda path: pins[Path(path)])
    assert len(entry.verify_goop_evidence(args, manifest, admission)) == 1000
    census['splits']['train']['records'][0]['finite_count'] = 1
    args.auxiliary_report.write_text(json.dumps(census))
    with pytest.raises(ValueError, match='census row'):
        entry.verify_goop_evidence(args, manifest, admission)
    monkeypatch.setattr(entry, 'sha', actual_sha)
    path = tmp_path/'context.npy'; values = np.zeros((401, 1), dtype='<f4'); values[0, 0] = np.nan; np.save(path, values)
    descriptor = {'shape': [401, 1], 'dtype': '<f4', 'size_bytes': path.stat().st_size, 'sha256': actual_sha(path), 'path': path.name}
    helpers = SimpleNamespace(np=np, data_loader=SimpleNamespace(_manifest_array_path=lambda base, d: base/d['path']))
    entry.verify_goop_auxiliary_arrays(args, helpers, {'records': [{'step_context': descriptor}]}, [rows[0]])
    values.view('<u4')[0, 0] = 0x7fc00001; np.save(path, values)
    descriptor.update(size_bytes=path.stat().st_size, sha256=actual_sha(path))
    with pytest.raises(ValueError, match='bits differ'):
        entry.verify_goop_auxiliary_arrays(args, helpers, {'records': [{'step_context': descriptor}]}, [rows[0]])


def test_numerical_training_functions_are_unchanged_from_pinned_sand_source():
    import ast
    import hashlib
    sand_path = ROOT/'train_sand_graph_support_cuda.py'
    assert hashlib.sha256(sand_path.read_bytes()).hexdigest() == 'fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124'
    old = {n.name: n for n in ast.parse(sand_path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    new = {n.name: n for n in ast.parse(PATH.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for name in ('RunLock', 'load_helpers', 'graph_exposure_config', 'validate_graph_history', 'capture_rng',
                 'restore_rng', 'tree_equal', 'assert_adam', 'guarded_update', 'read_pointer', 'configure_cuda'):
        assert ast.dump(old[name]) == ast.dump(new[name]), name
    old_run = ast.get_source_segment(sand_path.read_text(), old['run_training']).replace('sand', 'goop').replace('Sand', 'Goop')
    new_run = ast.get_source_segment(PATH.read_text(), new['run_training'])
    new_run = new_run.replace('        reverify_goop_evidence_and_auxiliaries(args, helpers)\n', '')
    assert ast.dump(ast.parse(old_run)) == ast.dump(ast.parse(new_run))
