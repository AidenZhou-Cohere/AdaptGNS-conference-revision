"""Read-only local scalar/source audit; no remote calls, arrays, models or CUDA."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import math
import re
import numpy as np

ROOT = Path(__file__).resolve().parent
checks = 0


def check(value, message):
    global checks
    checks += 1
    if not value:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def constants(name):
    result = {}
    for node in ast.parse((ROOT / name).read_text()).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = value
    return result


report_name = 'goop3d_vectorized_numerical_v1_report.json'
release_name = 'goop3d_vectorized_numerical_release_v1.json'
r, release = read(report_name), read(release_name)
v = constants('validate_goop3d_vectorized_cuda_v1.py')
t = constants('train_goop3d_graph_support_cuda_v2.py')
inputs = {ROOT / report_name: 'c5367c9e1c3dbd11a9ef280bd8dac81f7bd8e8b35674ab94dcc2e7d4c105f552',
    ROOT / release_name: 'ce2fd036b49c0e2310a0e639131c3ef1eb62cbf7564cffd1bf04a420128117c7',
    ROOT / 'validate_goop3d_vectorized_cuda_v1.py': '984a358a07e6b3e7bc80d7e0c64a2b23d4076fdf1a278c0f5d03121616fd3dd1',
    ROOT / 'train_goop3d_graph_support_cuda_v2.py': v['TRAINER_SHA'],
    ROOT / 'goop3d_graph_support.py': v['ORIGINAL_SHA'],
    ROOT / 'goop3d_graph_support_vectorized_v1.py': v['CANDIDATE_SHA']}
data_files = {'train_manifest':'goop3d_structural_metadata_v1/train.json',
    'structural_report':'goop3d_structural_metadata_v1/structural_report.json',
    'acquisition_report':'goop3d_acquisition_v1_report.json',
    'context_semantics':'goop3d_context_semantics_review.json',
    'auxiliary_report':'goop3d_auxiliary_v1_report.json',
    'data_review':'goop3d_data_compatibility_root_review_v1.json',
    'cpu_proof':'goop3d_vectorized_cpu_comparison_v1_report.json'}
inputs.update({ROOT / data_files[k]: value for k, value in v['DATA_PINS'].items()})
repo = ROOT.parents[2] / 'outputs' / 'AdaptGNS'
inputs.update({repo / key: value for key, value in t['SOURCE_PINS'].items()})
for path, digest in inputs.items():
    check(sha(path) == digest, 'Input hash differs: ' + str(path))
check(r['schema'] == v['SCHEMA'] and r['status'] == 'implementation_passed', 'Wrong completed report')
for flag in ('scientific_training_admitted','scientific_endpoint_selected','test_accessed'):
    check(r.get(flag) is False, 'Unexpected scope: ' + flag)
check(r['arms'] == ['base','mix'] and r['seed'] == 0 and r['steps_per_branch'] == 2, 'Numerical schedule')
check(r['max_optimizer_updates'] == r['completed_optimizer_calls'] == 26, 'Optimizer call count')
check(r['probe_lr_horizon'] == 100000 and r['attempted_update'] is None, 'Probe horizon/current attempt')
check(r['all_inputs_reverified'] is True and r['prospective_batches_saved_before_cuda'] is True, 'Evidence gates')
check(not any(k in r for k in ('error','error_type','unsuccessful_state','state_preservation_error')), 'Failure fields')
check(math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds'] > 0, 'Elapsed time')
check(release['schema'] == v['RELEASE_SCHEMA'] and release['issued_by'] == 'root'
      and release['status'] == 'admitted_for_bounded_validation'
      and release['scientific_training_admitted'] is False, 'Bounded root release')
for key in ('source_sha256','trainer_sha256','original_graph_sha256','candidate_graph_sha256'):
    check(r[key] == release[key], 'Report/release source binding: ' + key)
check(r['release_sha256'] == inputs[ROOT/release_name], 'Report release digest')
check(release['data_files_sha256'] == v['DATA_PINS'] == r['training_data']['source_evidence_sha256'], 'Data evidence pins')
check(release['max_optimizer_updates'] == 26, 'Released call limit')
contract = release['data_contract']
check(contract['scope'] == 'bounded_implementation_data_only' and contract['scientific_training_admitted'] is False
      and not any(k in contract for k in ('prospective_endpoint_updates','endpoint_selection_basis','selection_evidence_sha256')), 'Source-only contract')
manifest = read(data_files['train_manifest'])
records = manifest['records']
check(manifest['record_count'] == len(records) == 1000, 'Full train source')
order = sorted(range(1000), key=lambda i:(records[i]['positions']['shape'][1],i))
schedule=[]
for name,rank in [('small',0),('median',500),('large',998)]:
    ix=order[rank:rank+2]
    schedule.append({'case':name,'trajectory_indices':ix,'target_frames':[6,150],
        'trajectory_ids':[records[i]['id'] for i in ix], 'particle_counts':[records[i]['positions']['shape'][1] for i in ix],
        'dataset_indices':[ix[0]*295,ix[1]*295+144]})
check(schedule == r['schedule'] == release['schedule'], 'Independently reconstructed count-ranked schedule')
check(r['training_data']['trajectory_ids'] == [row['id'] for row in records]
      and r['training_data']['n_trajectories'] == 1000 and r['training_data']['eligible_frames'] == 295000
      and r['training_data']['manifest_sha256'] == v['DATA_PINS']['train_manifest']
      and r['training_data']['source'] == manifest['source'], 'Training provenance')
inventory_name='third_vm_pre_numerical_inventory_v1.json'
inventory=read(inventory_name);inputs[ROOT/inventory_name]=sha(ROOT/inventory_name)
check(inventory['utc'] == release['process_identity_checked_utc'] and inventory['gpu_apps'] == '', 'Released prelaunch idle inventory')
runtime=r['runtime']
check(release['cuda_index'] == 0 and runtime['device'] == 'cuda:0'
      and runtime['uuid'].lower().removeprefix('gpu-') == release['gpu_uuid'].lower().removeprefix('gpu-')
      and f"0, {release['gpu_uuid']}, NVIDIA GB200" in inventory['gpu_inventory'], 'Exact released GPU identity')
for key,value in {'torch':'2.13.0+cu129','cuda':'12.9','name':'NVIDIA GB200','threads':2,
    'deterministic_algorithms':True,'deterministic_warn_only':False,'float32_matmul_precision':'highest',
    'tf32':False,'amp':False,'compile':False,'ddp':False,'cublas_workspace_config':':4096:8'}.items():
    check(runtime[key] == value, 'Runtime: ' + key)

artifacts={}
def artifact(row, name):
    check(row['file'] == name and re.fullmatch('[0-9a-f]{64}',row['sha256']) is not None, 'Artifact descriptor: ' + name)
    check(name not in artifacts, 'Duplicate artifact filename')
    artifacts[name]=row['sha256']

check(len(r['saved_batches'])==3,'Three saved batches')
for i,(item,saved) in enumerate(zip(schedule,r['saved_batches'])):
    check(all(saved[k]==value for k,value in item.items()) and saved['noise_steps']==[2*i,2*i+1], 'Prospective saved batch metadata')
    artifact(saved,item['case']+'_prospective_batch.npz')
artifact(r['initial_state'],'initial_seed0.pt')
components={'before','graph','ledger','gradients','state_dict','optimizer_state','rng_states'}
check(len(r['cases'])==6 and [(c['case'],c['arm']) for c in r['cases']]==[(c['case'],a) for c in schedule for a in ('base','mix')], 'Six ordered case/arm cells')
case_summary=[];gates=0;component_count=0;unique_exposed=0
for i,item in enumerate(schedule):
    base,mix=r['cases'][2*i:2*i+2]
    for row in (base,mix):
        name,arm=row['case'],row['arm'];check([s['completed_steps'] for s in row['steps']]==[1,2],'Two steps')
        for step in row['steps']:
            checks_map=step['bytewise_component_checks']
            check(step['passed'] is True and set(checks_map)==components and all(x is True for x in checks_map.values()),'Complete exact components')
            component_count+=len(checks_map)
            for branch in ('original','candidate'):
                artifact(step[branch],f"{name}_{arm}_{branch}_step{step['completed_steps']}.pt")
            check(step['original']['sha256']==step['candidate']['sha256'],'Recorded serialized original/candidate bytes differ')
        gs=row['graph_gates'];check(len(gs)==(5 if i==0 else 4),'One graph gate per optimizer call')
        check(gs[0]==gs[2] and gs[1]==gs[3] and (i!=0 or gs[1]==gs[4]),'Original/candidate/replay graph identity')
        for j,gate in enumerate(gs):
            gates+=1;absolute=2*i+([0,1,0,1,1][j])
            check(all(gate[k] is True for k in ('passed','edges_exact','ledger_exact','rng_unchanged')),'Graph gates')
            check(re.fullmatch('[0-9a-f]{64}',gate['ordered_edge_sha256']) is not None,'Graph digest')
            check(len(gate['graph_ledger'])==2,'Two graph examples')
            for slot,e in enumerate(gate['graph_ledger']):
                coin_mat=[20261005,0,absolute,slot,4409]
                coin=bool(np.random.default_rng(np.random.SeedSequence(coin_mat)).random()<.5)
                check(e['example_slot']==slot and e['n_particles']==item['particle_counts'][slot],'Example identity/count')
                check(e['coin_seed_material']==coin_mat and e['pair_seed_material']==[20261005,0,absolute,slot,5501]
                      and e['exposure_coin'] is coin,'Independent graph coin reconstruction')
                check(e['expanded'] is (arm=='mix' and coin),'Exposure rule')
                check(e['optional_budget_if_exposed']==e['annulus_pairs']//4
                      and e['selected_optional_pairs']==(e['annulus_pairs']//4 if e['expanded'] else 0),'Exact optional budget')
                check(0<=e['native_self_edges']<=e['n_particles'] and 0<=e['native_directed_edges']<=128*e['n_particles']
                      and 0<=e['receivers_above_native_cap']<=e['n_particles'],'Native count ranges')
                for field in ('native_edge_sha256','optional_pair_sha256','noisy_current_sha256'):
                    check(re.fullmatch('[0-9a-f]{64}',e[field]) is not None,'Example digest')
                unique_exposed+=int(arm=='mix' and j<2 and e['selected_optional_pairs']>0)
        case_summary.append({'case':name,'arm':arm,'numerical_steps':2,'graph_gates':len(gs),
            'selected_pairs_per_unique_batch':[sum(e['selected_optional_pairs'] for e in g['graph_ledger']) for g in gs[:2]]})
    for bg,mg in zip(base['graph_gates'],mix['graph_gates']):
        for b,m in zip(bg['graph_ledger'],mg['graph_ledger']):
            check({k:v for k,v in b.items() if k not in ('expanded','selected_optional_pairs','optional_pair_sha256')}
                  =={k:v for k,v in m.items() if k not in ('expanded','selected_optional_pairs','optional_pair_sha256')}, 'Paired noisy/native/annulus state')
check(gates==26 and unique_exposed>0,'Complete nonvacuous gates')
check(set(r['replay'])=={'base','mix'},'Both replays')
for arm,replay in r['replay'].items():
    check(all(replay[k] is True for k in ('passed','checkpoint_serialization_exact','final_payload_exact')),'Serialized replay gates')
    check(set(replay['bytewise_component_checks'])==components and all(x is True for x in replay['bytewise_component_checks'].values()),'Replay components')
    component_count+=len(components)
    artifact(replay['checkpoint'],f'{arm}_bounded_replay_checkpoint_step1.pt')
    artifact(replay['resumed_artifact'],f'{arm}_bounded_replay_step2.pt')
    row=next(c for c in r['cases'] if c['case']=='small' and c['arm']==arm)
    check(replay['resumed_artifact']['sha256']==row['steps'][1]['candidate']['sha256'],'Resumed/uninterrupted serialized bytes differ')
check(len(artifacts)==32,'Full32 artifact inventory')
check(component_count==98,'Twelve step pairs plus two replay comparisons times seven components')
for name in ('goop3d_vectorized_numerical_v1.stderr.log','goop3d_vectorized_numerical_v1.stdout.log'):
    inputs[ROOT/name]=sha(ROOT/name)
log=(ROOT/'goop3d_vectorized_numerical_v1.stderr.log').read_text()
check(re.search(r'^Traceback|^[\w.]+(?:Error|Exception):',log,re.M) is None,'Unexpected failure in stderr')
for path,digest in inputs.items():
    check(sha(path)==digest,'Terminal input hash differs')
output={'schema':'adaptgns_goop3d_bounded_numerical_report_methods_audit_v1',
    'status':'passed_bounded_same_cuda_report_consistency','reviewer':'conference_critique_methods',
    'reviewed_utc':datetime.now(timezone.utc).isoformat(),'checks_passed':checks,
    'scope':'Local scalar report, release, metadata and frozen-source consistency; no remote access, model deserialization, official numeric arrays or CUDA execution.',
    'files_sha256':{str(p):h for p,h in inputs.items()},'auditor_sha256':sha(__file__),
    'optimizer_calls':26,'case_arm_cells':6,'exact_original_candidate_step_pairs':12,
    'exact_checkpoint_replays':2,'component_checks':98,'graph_gates':gates,
    'exposed_unique_mix_examples':unique_exposed,'reported_artifact_sha256':artifacts,
    'cases':case_summary,'runtime':runtime,'observed_validator_elapsed_seconds':r['elapsed_seconds'],
    'scientific_training_admitted':False,'scientific_endpoint_selected':False,'test_accessed':False,
    'remaining_findings':[],
    'interpretation':'The pinned bounded validator reports bytewise exact original/vectorized same-CUDA inputs, outputs, gradients, Adam, RNG and serialized replay across the prospective size cases. It supports the subsequent root-gated capacity prerequisite only.',
    'limits':['The32 .pt/.npz payload files remain on the execution host; this audit verifies their report descriptors and reported byte equality, not independent reads of those remote bytes.',
        'Actual host launch/reaping and raw-artifact retention remain root-owned execution evidence. The prelaunch inventory and released/runtime physical device agree.',
        'Passing26 calls is bounded implementation evidence, not proof over all batches/seeds, CPU/CUDA parity, scientific efficacy, capacity throughput or selected training endpoint.']}
destination=ROOT/'goop3d_vectorized_numerical_v1_independent_review_methods.json'
with destination.open('x') as f:
    json.dump(output,f,indent=2);f.write('\n')
print(json.dumps({'audit':str(destination),'sha256':sha(destination),'checks':checks,
                  'graph_gates':gates,'component_checks':component_count,'exposed_unique_mix_examples':unique_exposed}))
