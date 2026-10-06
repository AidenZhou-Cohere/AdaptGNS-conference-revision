#!/usr/bin/env python3
"""Bounded root-dispatched Sand operational helper; imports no scientific code."""
import base64, ctypes, datetime, hashlib, json, os, pathlib, socket, subprocess, sys, time
R='/root/repos/AdaptGNS-cuda-20261006';RP=R+'/cuda_preparation';A=R+'/sand_final_analysis_20261006_v1'
EVALUATIONS = {'A': {'boot_id': 'cbe68c69-6a30-47f9-a9f8-b42a80785a1f', 'gpu_uuids': ['GPU-b6685200-7eaf-b46c-89b4-53e8715b8e21', 'GPU-38f0a7dd-8a4a-a461-2de3-747ebc7a73c6', 'GPU-af93e07a-ed08-8d07-25c4-4a51969065e0', 'GPU-8e9d199c-b92d-57f0-57af-ffea9caeace7'], 'historical_pids': [4823, 4826, 4827, 4828, 4829, 16106, 16108, 16109, 16110, 16625, 16626, 16627, 17698, 17699, 17700, 17886, 17887, 17888, 18011, 18012, 18022, 18201, 18202, 18203, 18204, 18205, 18206, 18438, 18439, 18440, 18441, 18442, 18443], 'hostname': 'aidenzhou-yellow-worm-77-78fff65d5-62zgv', 'initial_children': [{'argv': ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/base_seed1/checkpoint-000100000.pt', '--checkpoint-sha256', '68cec3c0e02281c630c5033d51d80450c2ed50e6ec5fbbbc5e17117bbee5d7ec', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/base_seed1/full_rollout_test', '--mode', 'full-rollout', '--split', 'test', '--objective', 'faithful', '--arm', 'base', '--seed', '1', '--cuda-index', '0', '--threads', '2', '--max-seconds', '7200'], 'executable': '/usr/bin/python3.12', 'pgid': 24164, 'pid': 24164, 'ppid': 24151, 'sid': 24164, 'start_ticks': 120241301}, {'argv': ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/mix_seed1/checkpoint-000100000.pt', '--checkpoint-sha256', '9b237f5297fd47ff5e214b7ff26dc69a708f023644b2dd94797b69b4c765c6cf', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/mix_seed1/full_rollout_test', '--mode', 'full-rollout', '--split', 'test', '--objective', 'faithful', '--arm', 'mix', '--seed', '1', '--cuda-index', '1', '--threads', '2', '--max-seconds', '7200'], 'executable': '/usr/bin/python3.12', 'pgid': 24165, 'pid': 24165, 'ppid': 24151, 'sid': 24165, 'start_ticks': 120241302}, {'argv': ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/base_seed2/checkpoint-000100000.pt', '--checkpoint-sha256', '84cded9870d2eb51d1ebe0df89c3155d1c88de7cd56ccece95acdd6702bd16a7', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/base_seed2/full_rollout_test', '--mode', 'full-rollout', '--split', 'test', '--objective', 'faithful', '--arm', 'base', '--seed', '2', '--cuda-index', '2', '--threads', '2', '--max-seconds', '7200'], 'executable': '/usr/bin/python3.12', 'pgid': 24166, 'pid': 24166, 'ppid': 24151, 'sid': 24166, 'start_ticks': 120241303}, {'argv': ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_runtime_migration_A_20261006_v2/training/mix_seed2/checkpoint-000100000.pt', '--checkpoint-sha256', '15b29c13596d3398f0935703f42d018a520d769dd634581046a1098013865679', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1/jobs/mix_seed2/full_rollout_test', '--mode', 'full-rollout', '--split', 'test', '--objective', 'faithful', '--arm', 'mix', '--seed', '2', '--cuda-index', '3', '--threads', '2', '--max-seconds', '7200'], 'executable': '/usr/bin/python3.12', 'pgid': 24167, 'pid': 24167, 'ppid': 24151, 'sid': 24167, 'start_ticks': 120241303}], 'owned_gpu_indices': [0, 1, 2, 3], 'owner': {'argv': ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_sand_final_evaluation_scoped_v1.py', '--execute', '--release', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_controls_20261006_v1/A.evaluation_release.json', '--lifecycle-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/measure_sand_cuda_capacity_v2.py', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1'], 'executable': '/usr/bin/python3.12', 'pgid': 24151, 'pid': 24151, 'ppid': 40, 'sid': 24151, 'start_ticks': 120240947}, 'queue_root': '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_A_20261006_v1', 'release_path': '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_controls_20261006_v1/A.evaluation_release.json', 'release_sha256': '37cf446c894cb0e46567d060365716bcfc9bae685f1a7bb44a2d473b58127af4', 'session_id': 51595}, 'B': {'boot_id': '3d254b4e-ede4-492f-839a-5f16b6f3d1b3', 'gpu_uuids': ['GPU-37c3dfdb-9027-e3cb-c899-b0a12f130b52', 'GPU-3b22d55b-8f80-cc77-5a6f-f9181f3286c4', 'GPU-89b83aa8-f20b-dd13-811c-5f54165eb865', 'GPU-28475341-1f76-1575-5aaf-bb8616bc1052'], 'historical_pids': [31391, 31394, 31395, 66090, 66092, 66093, 66094, 73206, 73207, 73222, 73223, 73224, 73225], 'hostname': 'aidenzhou-aquamarine-toad-75-6d8b45c98d-mgjlq', 'initial_children': [{'argv': ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_scoped_science_B_20261006_v1/jobs/base_seed0/checkpoint-000100000.pt', '--checkpoint-sha256', 'f4cbea33853233228e569cc7363851d2f0b396d87fc20f9c80825a7023d8d829', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_B_20261006_v1/jobs/base_seed0/full_rollout_test', '--mode', 'full-rollout', '--split', 'test', '--objective', 'faithful', '--arm', 'base', '--seed', '0', '--cuda-index', '2', '--threads', '2', '--max-seconds', '7200'], 'executable': '/usr/bin/python3.12', 'pgid': 74097, 'pid': 74097, 'ppid': 74093, 'sid': 74097, 'start_ticks': 536613864}, {'argv': ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py', '--execute', '--repo', '/root/repos/AdaptGNS-cuda-20261006', '--benchmark-helper', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/benchmark_sand_graph_support_rollout.py', '--benchmark-sha256', '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13', '--cohort', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort.json', '--cohort-audit', '/root/repos/AdaptGNS-cuda-20261006/sand_post_training_A_20261006_v1/cohort/cohort_audit.json', '--manifest', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/test.json', '--admission', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/controls/test.root_admission.json', '--structural-report', '/root/repos/AdaptGNS-cuda-20261006/sand_reserved_preparation_20261006_v1/numeric/structural_report.json', '--train-admission', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_train_admission.json', '--trainer-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/train_sand_graph_support_cuda.py', '--protocol', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_graph_support_100k_protocol_v1.md', '--checkpoint', '/root/repos/AdaptGNS-cuda-20261006/sand_scoped_science_B_20261006_v1/jobs/mix_seed0/checkpoint-000100000.pt', '--checkpoint-sha256', 'f7abcf804bacf7d4b21f7557a4823a15c85effd013188548f0b98a0f306beccb', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_B_20261006_v1/jobs/mix_seed0/full_rollout_test', '--mode', 'full-rollout', '--split', 'test', '--objective', 'faithful', '--arm', 'mix', '--seed', '0', '--cuda-index', '3', '--threads', '2', '--max-seconds', '7200'], 'executable': '/usr/bin/python3.12', 'pgid': 74098, 'pid': 74098, 'ppid': 74093, 'sid': 74098, 'start_ticks': 536613865}], 'owned_gpu_indices': [2, 3], 'owner': {'argv': ['/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_sand_final_evaluation_scoped_v1.py', '--execute', '--release', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_controls_20261006_v1/B.evaluation_release.json', '--lifecycle-source', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/measure_sand_cuda_capacity_v2.py', '--output-dir', '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_B_20261006_v1'], 'executable': '/usr/bin/python3.12', 'pgid': 74093, 'pid': 74093, 'ppid': 41, 'sid': 74093, 'start_ticks': 536613652}, 'queue_root': '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_B_20261006_v1', 'release_path': '/root/repos/AdaptGNS-cuda-20261006/sand_final_evaluation_controls_20261006_v1/B.evaluation_release.json', 'release_sha256': '8fa068d3ea1768da53bf62297c0bfe2220c8aefc44b7c334acfc5f0c9e64a466', 'session_id': 12559}}
SOURCE_PINS = {'/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/evaluate_sand_graph_support_final.py': '952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/launch_stopped_analysis_cpu_v1.py': '294cfaebb8da60e5364b76fa2f1f889a2f06fdb07dc1ff3253f7bb44d8a2c492', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/qualify_sand_native_timer_owner_v1.py': '306c44b7c693f8ce60b4363c1f6891b0537ac177d554d87e7d2bd483e0da1782', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_saved_array_audit_v2/audit_sand_paired_arrays_v1.py': '541bdc67a99e44ccb263b1bbe439fa37132ef95ab597e1dd440742c0bb4a7a2d', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_saved_array_audit_v2/audit_sand_saved_arrays_v1.py': 'a7e47c8e4740cc8c1767dc778ef4fa5291453f4142a0753cc62a0bf82bea7740', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_saved_array_audit_v2/sand_saved_diagnostic_audit_v1.py': '700695b69691b9da45e31ce7a9821900a12ff882be5b4a09927ff8b6dd0f8cdd', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/sand_scoped_operational_amendment_v1.md': '411547c3355859906f59792b3b493841030b0afd2e105e17e466114df6743738', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/stopped_analysis_operations_v1.json': '7e5e09e6025916c7309b103cf715db3fe9cfbb56be5df5eda7c372a8c6cf8b9b', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/summarize_goop_graph_support_quota_v2.py': 'da85058ea2ce0fc0f1b67e6cad442369835dfe1e8926d8f26c595d143f1ec33c', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/summarize_sand_graph_support_scoped_v1.py': '1e70f1689a75c6f68eb970048aab7837153c19f70caf30ec19143cf66ab5d272', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_goop_evaluation_gpu_scoped_v3.py': 'a13ca1bc30161f89cf79b6c0b632c65a272913302ef0cc37dcfd7ed762c6aa21', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_sand_final_evaluation_scoped_v1.py': 'efebe762ea60b1b711925fb9bb3bc91461a16b1440491b2fa554c99171773829', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_sand_post_completion_cpu_v2.py': '43708e9d5c62f8eb20ab2bb19777946485909383c0bf6df91de0f448fdf97d8a', '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_stopped_analysis_cpu_v1.py': '18b5f6cff4f212c7152810f8ff26e9159416ae322aabd666c37e8a60117d594e'}
OPS=('sand_collect_A','sand_collect_B','sand_saved_A','sand_saved_B','sand_summarize','sand_paired')
def need(ok,message):
    if not ok:raise ValueError(message)
def exact(a,b):return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
def dt(value):
    result=datetime.datetime.fromisoformat(value);need(result.tzinfo is not None,'Aware UTC required');return result
def utc():return datetime.datetime.now(datetime.timezone.utc)
def ensure_empty_owner_parent(check):
    check();root=pathlib.Path(A);need(root.is_dir() and not root.is_symlink() and root.resolve()==root,'Ordinary canonical analysis root required')
    parent=root/'owners';created=not parent.exists()
    if created:parent.mkdir(mode=0o700,exist_ok=False)
    need(parent.is_dir() and not parent.is_symlink() and not list(parent.iterdir()),'Empty ordinary owners parent required');check()
    return {'path':str(parent),'created':created,'empty':True}

def validate_evaluation_closure(role,release,snapshot,status,ledger,process,observations,metadata):
    """Pure accounting of all originally registered children, including failures."""
    cfg=EVALUATIONS[role];owner=cfg['owner'];Q=cfg['queue_root']
    need(exact(release,snapshot),'Original queue release snapshot differs from exact issued release')
    schema='adaptgns_sand_evaluation_gpu_scoped_v1'
    need(release['host_role']==role and release['host']==cfg['hostname'] and release['dataset']=='Sand','Original release role differs')
    need(status['schema']==ledger['schema']==process['schema']==schema and ledger['dataset']=='Sand','Stopped Sand metadata schema differs')
    need(status['state'] in ('allocation_finished','stopped_requires_review') and status['coverage_ledger_sha256']==metadata[Q+'/coverage_ledger.json'],'Exact stopped queue status required')
    need(status['unreaped_owned_children']==ledger['unreaped_owned_children']==process['unreaped_owned_children']==[],'Unreaped original child remains')
    need(exact(process['gpu_scope'],release['gpu_scope']),'Original GPU scope changed')
    if observations is None:
        need(process['gpu_observations_sha256'] is None and process['all_children']==[] and all(row['outcomes']==[] for row in process['streams']) and bool(process['abort_reason']),'GPU observations absent after an original child')
    else:need(process['gpu_observations_sha256']==metadata[Q+'/gpu_observations.json'],'Original GPU observations bytes differ')
    names=('full_rollout_test','same_state_valid','same_state_test','clean_validation')
    commands={(s['id'],name):s['commands'][i] for s in release['streams'] for i,name in enumerate(names)}
    need(len(commands)==(16 if role=='A' else 8),'Original full stage allocation differs')
    reverse={tuple(command):key for key,command in commands.items()};need(len(reverse)==len(commands),'Duplicate original command')
    rows=ledger['stages'];need(len(rows)==len(commands) and {(r['stream'],r['stage']) for r in rows}==set(commands),'Missing/duplicate original ledger stage')
    outcomes={};streams={s['id'] for s in release['streams']}
    need(len(process['streams'])==len(streams) and {s['id'] for s in process['streams']}==streams,'Original process stream grid differs')
    for stream in process['streams']:
        for outcome in stream['outcomes']:
            key=(stream['id'],outcome['stage']);need(key in commands and key not in outcomes,'Duplicate or undeclared original outcome');outcomes[key]=outcome
    for row in rows:
        key=(row['stream'],row['stage']);need(exact(row['outcome'],outcomes.get(key,{'stage':row['stage'],'state':'never_started'})),'Ledger/original process outcome differs')
    identities={};registered_keys=set()
    for child in process['all_children']:
        key=reverse.get(tuple(child['command']));need(key in commands and key not in registered_keys and child['stage']==key[1],'Unknown/duplicate registered original child command');registered_keys.add(key)
        identity=child['identity'];need(isinstance(identity,dict) and exact(identity['argv'],commands[key]),'Missing original registered native child identity')
        need(all(type(identity[k]) is int and identity[k]>0 for k in ('pid','ppid','start_ticks')) and identity['ppid']==owner['pid'],'Original child native parent/start differs')
        need(identity['pid'] not in identities,'Original child PID reused/duplicated')
        outcome=outcomes.get(key);need(outcome and type(outcome['pid']) is int and outcome['pid']==identity['pid'] and type(outcome['exit_code']) is int and outcome['state'] in ('exited','quota_expired','aborted_by_supervisor'),'Every started original child needs actual reaped exit')
        need(child['output_dir']==outcome['output_dir']==commands[key][commands[key].index('--output-dir')+1] and exact(child['signals'],outcome['signals']) and exact(child['quota_expired'],outcome['quota_expired']),'Original output/signal/quota history differs')
        identities[identity['pid']]=identity
    need(registered_keys==set(outcomes),'Unaccounted original registered/outcome child')
    for initial in cfg['initial_children']:
        need(initial['pid'] in identities and all(exact(identities[initial['pid']][key],initial[key]) for key in ('pid','ppid','start_ticks','argv','executable')),'Original observed startup child missing or changed')
    return {'registered_children':{str(pid):value for pid,value in identities.items()},'original_stage_outcomes':rows,'all_children_reaped':True,'complete_registered_child_ledger_checked':True,'original_queue_status':status,'original_abort_reason':process.get('abort_reason')}

def main():
    need(not sys.flags.optimize,'Nonoptimized helper required')
    p=json.load(sys.stdin);role=p['role'];need(role in EVALUATIONS,'Exact original host role required');cfg=EVALUATIONS[role];Q=cfg['queue_root']
    stop=dt(p['stop_utc']);origin=utc();mono_stop=time.monotonic()+(stop-origin).total_seconds()
    def check():need(utc()+datetime.timedelta(seconds=5)<stop and time.monotonic()+5<mono_stop,'Original operational/analysis stop reached')
    check();need(socket.gethostname()==cfg['hostname'],'Exact original hostname required')
    boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip();need(boot==cfg['boot_id']==p['boot_id'],'Original qualified boot changed')
    hashes={};trees={}
    def pin(path,h):
        path=str(path);need(path not in hashes or hashes[path]==h,'Read file changed: '+path);hashes[path]=h;return h
    def ordinary(path,kind='file'):
        x=pathlib.Path(path);need(not x.is_symlink() and x.resolve()==x and (x.is_file() if kind=='file' else x.is_dir()),'Canonical ordinary '+kind+' required: '+str(x));return x
    def digest(path):
        check();x=pathlib.Path(path)
        if str(x)==R+'/.venv/bin/python':need(str(x.resolve())=='/usr/bin/python3.12','Qualified Python alias changed')
        else:ordinary(x)
        h=hashlib.sha256()
        with x.open('rb') as f:
            while True:
                check();block=f.read(1<<20)
                if not block:break
                h.update(block)
        return pin(x,h.hexdigest())
    def doc(path):
        check();raw=ordinary(path).read_bytes();check();pin(path,hashlib.sha256(raw).hexdigest());return json.loads(raw)
    def blob(path):
        check();raw=ordinary(path).read_bytes();check();return {'path':str(path),'sha256':pin(path,hashlib.sha256(raw).hexdigest()),'bytes':len(raw),'base64':base64.b64encode(raw).decode()}
    def absent(pids):
        need(all(type(pid) is int and pid>0 for pid in pids),'Exact positive native PIDs required');value={str(pid):not pathlib.Path('/proc',str(pid)).exists() for pid in sorted(set(pids))};need(value and all(value.values()),'Historical native process remains');return value
    def gpu():
        check();raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True,timeout=5);check();mapping={int(a.strip()):b.strip() for a,b in (row.split(',') for row in raw.splitlines() if row.strip())};need(mapping==dict(enumerate(cfg['gpu_uuids'])),'Original GPU UUID mapping changed')
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=5);check();apps=[]
        for row in raw.splitlines():
            if not row.strip():continue
            pid,uuid=row.split(',');pid=int(pid.strip());uuid=uuid.strip();need(pid>0 and uuid in cfg['gpu_uuids'] and cfg['gpu_uuids'].index(uuid) not in cfg['owned_gpu_indices'],'Assigned device still has a compute process');apps.append({'pid':pid,'gpu_uuid':uuid})
        return {'gpu_uuids':cfg['gpu_uuids'],'observed_unassigned_gpu_processes':apps,'assigned_devices_empty':True}
    def tree(root):
        need(root==Q,'Only exact original role queue may be inventoried');q=ordinary(root,'directory');entries=[];files={}
        for x in sorted(q.rglob('*')):
            check();need(not x.is_symlink() and (x.is_file() or x.is_dir()),'Special stopped queue entry');rel=str(x.relative_to(q));entries.append([rel,'file' if x.is_file() else 'directory'])
            if x.is_file():files[rel]={'sha256':digest(x),'bytes':x.stat().st_size}
        inv={'entries':entries,'files':files};trees[root]=inv;return inv
    def stage(rows,transfer=False):
        for row in rows:
            check();x=pathlib.Path(row['path']);raw=base64.b64decode(row['base64']);need(hashlib.sha256(raw).hexdigest()==row['sha256'],'Staging bytes differ')
            need(x.is_absolute() and not x.is_symlink() and x.resolve()==x and type(row['allow_identical_existing']) is bool,'Canonical ordinary stage target required')
            if transfer:need(role=='A' and str(x) in (A+'/B.stopped_collection.json',A+'/B.saved_array_audit.json'),'Only exact B scalar products transfer to A')
            else:
                controls={A+'/controls/'+n for n in ('analysis_phase.json','A.evaluation_closure.json','B.evaluation_closure.json','A.collection_release.json','B.collection_release.json','summary_release.json')}|{A+'/controls/'+op+'.cpu_release.json' for op in OPS}
                need(str(x) in SOURCE_PINS or str(x) in controls,'Undeclared stage target')
                if str(x) in SOURCE_PINS:need(row['sha256']==SOURCE_PINS[str(x)],'Frozen stage source differs')
            if str(x) not in SOURCE_PINS:need(row['allow_identical_existing'] is False,'Controls/products must be fresh')
            if x.exists():need(row['allow_identical_existing'] and digest(x)==row['sha256'],'Existing attempt/output cannot be replaced')
            else:
                x.parent.mkdir(parents=True,exist_ok=True);ordinary(x.parent,'directory')
                with x.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            need(digest(x)==row['sha256'],'Published stage bytes differ')
    result={'action':p['action'],'role':role,'hostname':cfg['hostname'],'host_boot_id':boot}
    if p['action']=='evaluation_closure':
        need(p['original_exit_observed'] is True and p['original_tool_session']==cfg['session_id'],'Original tool exit required first')
        absent(cfg['historical_pids']+[cfg['owner']['pid']]+[v['pid'] for v in cfg['initial_children']])
        need(digest(cfg['release_path'])==cfg['release_sha256'],'Original root release bytes changed');release=doc(cfg['release_path'])
        docs={name:doc(Q+'/'+name) for name in ('release_snapshot.json','queue_status.json','coverage_ledger.json','process_outcomes.json')}
        observations=doc(Q+'/gpu_observations.json') if pathlib.Path(Q+'/gpu_observations.json').exists() else None
        metadata={path:h for path,h in hashes.items() if path.startswith(Q+'/')}
        checked=validate_evaluation_closure(role,release,docs['release_snapshot.json'],docs['queue_status.json'],docs['coverage_ledger.json'],docs['process_outcomes.json'],observations,metadata)
        pids=cfg['historical_pids']+[cfg['owner']['pid']]+[int(pid) for pid in checked['registered_children']]
        result.update(checked,closure_metadata_sha256=metadata,queue_release_sha256=metadata[Q+'/release_snapshot.json'],original_release_sha256=cfg['release_sha256'],native_absent=absent(pids),gpu=gpu())
    elif p['action']=='stage_runtime':
        stage(p['stage_files']);ordinary(R,'directory')
        need(sys.platform.startswith('linux') and os.uname().machine=='aarch64' and ctypes.sizeof(ctypes.c_long)==ctypes.sizeof(ctypes.c_void_p)==8,'Qualified Linux aarch64 LP64 required')
        glibc=os.confstr('CS_GNU_LIBC_VERSION');need(glibc and glibc.startswith('glibc '),'Qualified glibc required')
        actual={path:digest(path) for path in p['expected_sha256']};need(actual==p['expected_sha256'],'Frozen runtime/source bytes differ')
        for path in p.get('fresh_paths',[]):
            x=pathlib.Path(path);need(x.is_absolute() and x.resolve()==x and not x.exists() and not x.is_symlink(),'Fresh canonical output required: '+path)
        if p.get('prepare_empty_owner_parent') is True:result['prepared_owner_parent']=ensure_empty_owner_parent(check)
        result.update(runtime_sha256=actual,glibc=glibc,machine=os.uname().machine,timeout_version=subprocess.check_output(['/usr/bin/timeout','--version'],text=True,timeout=5).splitlines()[0],native_absent=absent(p['native_pids']),gpu=gpu())
    elif p['action']=='snapshot':
        result.update(native_absent=absent(p['native_pids']),gpu=gpu());inputs={}
        for path,h in p['inputs_sha256'].items():
            actual=digest(path);need(h is None or h==actual,'Operation input differs');inputs[path]=actual
        invs={root:tree(root) for root in p['tree_roots']}
        for root,inv in invs.items():
            for rel,row in inv['files'].items():inputs[root+'/'+rel]=row['sha256']
        entries={root:inv['entries'] for root,inv in invs.items()}
        if p.get('expected_tree_entries') is not None:need(exact(entries,p['expected_tree_entries']),'Stopped tree membership changed')
        documents=[blob(path) for path in p.get('documents_required',[])];missing=[]
        for path in p.get('documents_optional',[]):
            if pathlib.Path(path).exists():documents.append(blob(path))
            else:missing.append(path)
        result.update(inputs_sha256=inputs,protected_tree_entries=entries,queue_inventories=invs,documents=documents,optional_documents_absent=missing)
    elif p['action']=='transfer':
        stage(p['stage_files'],True);result.update(transferred_sha256={row['path']:row['sha256'] for row in p['stage_files']},native_absent=absent(p['native_pids']),gpu=gpu())
    elif p['action']=='capture':
        directory=ordinary(p['directory'],'directory');need(str(directory) in {A+'/owners/'+op for op in OPS},'Exact operation owner directory required');terminal=doc(directory/'owner_terminal.json')
        need(terminal.get('child') and terminal['child'].get('identity'),'Complete original worker identity required')
        pids=[terminal['outer_timeout_identity']['pid'],terminal['owner_identity']['pid'],terminal['child']['pid']]
        need(all(x.is_file() and not x.is_symlink() for x in directory.iterdir()),'Owner outputs must be ordinary files')
        paths=[str(x) for x in sorted(directory.iterdir())]+p['outputs'];result.update(files=[blob(path) for path in paths],native_absent=absent(pids+p['historical_pids']),gpu=gpu())
        for field in ('input_sha256','evidence_sha256','output_sha256'):
            for path,h in terminal[field].items():need(digest(path)==h,'Final owner binding changed')
        for root,wanted in p['tree_entries'].items():
            inv=tree(root);need(exact(inv['entries'],wanted),'Original stopped queue membership changed after operation');need(all(terminal['input_sha256'][root+'/'+rel]==row['sha256'] for rel,row in inv['files'].items()),'Original stopped queue bytes changed after operation')
        result['final_input_and_output_hashes_verified']=True
    else:raise ValueError('Unknown operational action')
    check();encoded=json.dumps(result,sort_keys=True,allow_nan=False);check()
    for root,inv in list(trees.items()):need(exact(tree(root),inv),'Stopped queue membership/bytes changed before publication')
    for path,h in list(hashes.items()):need(digest(path)==h,'Observed source/input bytes changed before publication')
    if 'prepared_owner_parent' in result:
        parent=ordinary(A+'/owners','directory');need(not list(parent.iterdir()),'Owner parent changed before readiness publication')
    need(pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'Original boot changed')
    need(absent([int(pid) for pid in result['native_absent']])==result['native_absent'],'Native process appeared before publication')
    final_gpu=gpu();need(final_gpu['gpu_uuids']==result['gpu']['gpu_uuids'] and final_gpu['assigned_devices_empty'] is True,'Assigned GPU state changed')
    check();clock={'host_utc':utc().isoformat(),'host_monotonic_seconds':time.monotonic(),'host_boot_id':boot};suffix=json.dumps({'clock':clock,'checked_utc':clock['host_utc']},sort_keys=True,allow_nan=False);check();print(encoded[:-1]+','+suffix[1:])
if __name__=='__main__':main()
