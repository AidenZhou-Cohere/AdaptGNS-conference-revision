#!/usr/bin/env python3
"""Saved-scalar-only independent review of the root's Sand CUDA graph check."""
import hashlib,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
REPORT=P/'sand_graph_support_validation_v1_report.json'
EXPECTED='72a7eaaf28aa642c9f523515742c6b8f07cb0e0a3727a8cb5d14032ce91c3390'
checks=0;comparisons=0;finite_scalars=0;exposure=[]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def check(value,message):
 global checks
 checks+=1
 if not value:raise ValueError(message)
def digest(value):return isinstance(value,str) and len(value)==64 and set(value)<=set('0123456789abcdef')
def walk(value,path='root'):
 global comparisons,finite_scalars
 if isinstance(value,dict):
  if 'passed' in value:check(value['passed'] is True,'Failed saved check: '+path)
  if 'tensor_count' in value:check(value['tensor_count']==len(value['tensors']) and value['names_equal'] is True,'Tensor map mismatch: '+path)
  if 'tolerance' in value:
   comparisons+=1
   check(value['tolerance']=='exact' and value['atol']==value['rtol']==0.,'Tolerance altered: '+path)
   check(value['bitwise_equal'] is True and value['finite'] is True and value['reference_shape']==value['candidate_shape'],'Exact tensor comparison mismatch: '+path)
   check(all(value[k]==0 for k in ('max_absolute_error','max_relative_error_floor1e-12','relative_l2_error_floor1e-12','maximum_tolerance_ratio','outside_tolerance_elements')),'Nonzero exact difference: '+path)
  for k,v in value.items():walk(v,path+'/'+k)
 elif isinstance(value,list):
  for i,v in enumerate(value):walk(v,path+'/'+str(i))
 elif type(value) in (int,float):finite_scalars+=1;check(math.isfinite(value),'Nonfinite scalar: '+path)

def graph_check(graph,arm,step,label):
 check(graph['passed'] is True and all(graph[k] is True for k in ('native_prefix_exact','ordered_edges_and_budget_exact','partition_preserved','ledger_exact','graph_torch_rng_unchanged','zero_addition_tensor_identity')),'Graph booleans differ')
 examples=graph['examples'];check([r['example_slot'] for r in examples]==[0,1],'Two ordered examples required')
 for e in examples:
  slot=e['example_slot'];material=[20261005,0,step,slot,4409]
  expected_coin=bool(np.random.default_rng(np.random.SeedSequence(material)).random()<.5)
  check(e['coin_seed_material']==material and e['pair_seed_material']==[20261005,0,step,slot,5501],'Graph RNG material mismatch')
  check(e['exposure_coin'] is expected_coin and e['expanded'] is (arm=='mix' and expected_coin),'Exposure differs from fixed coin')
  check(e['optional_budget_if_exposed']==e['annulus_pairs']//4 and e['selected_optional_pairs']==(e['optional_budget_if_exposed'] if e['expanded'] else 0),'Exact25% optional budget mismatch')
  check(e['n_particles']>0 and 0<=e['native_self_edges']<=e['native_directed_edges'] and 0<=e['receivers_above_native_cap']<=e['n_particles'],'Impossible graph counts')
  check(all(digest(e[k]) for k in ('native_edge_sha256','optional_pair_sha256','noisy_current_sha256')),'Malformed graph hash')
 check(graph['native_directed_edges']==sum(e['native_directed_edges'] for e in examples),'Native directed count mismatch')
 check(graph['actual_directed_edges']==graph['native_directed_edges']+2*sum(e['selected_optional_pairs'] for e in examples),'Symmetric optional suffix mismatch')
 exposure.append({'scope':label,'arm':arm,'graph_step':step,'selected_pairs_by_example':[e['selected_optional_pairs'] for e in examples]})

def main():
 check(sha(REPORT)==EXPECTED,'Report hash mismatch');v=json.loads(REPORT.read_text())
 check(v['schema']=='adaptgns_sand_graph_support_cuda_validation_v1' and v['status']=='implementation_passed','Unexpected validation identity')
 check(v['scientific_training_admitted'] is False and v['prior_cpu_cuda_validation_status']=='validation_failed','Old failure or admission status changed')
 check(v['seed']==0 and v['objective']=='faithful' and v['shared_trainer_functions_exact'] is True and v['all_inputs_reverified'] is True,'Incomplete required gate')
 check(v['max_optimizer_updates']==15 and v['elapsed_seconds']>0,'Bounded execution metadata differs')
 check(sha(P/'validate_sand_graph_support_cuda.py')==v['source_sha256'],'Validator source mismatch')
 for name,d in v['source_pins'].items():check(sha(P/name)==d,'Pinned source differs: '+name)
 previous=json.loads((P/'sand_actual_data_validation_v2_report.json').read_text())
 check(sha(P/'sand_actual_data_validation_v2_report.json')==v['saved_report_sha256'] and previous['status']=='validation_failed','Prior strict failure changed')
 check([c['case'] for c in v['cases']]==['small','median','large'],'Complete fixed cases required')
 for step,(c,old) in enumerate(zip(v['cases'],previous['cases'])):
  check(c['batch_sha256']==old['batch_sha256'] and c['particle_counts']==old['particle_counts'] and c['graph_schedule_step']==step,'Saved training batch identity differs')
  check(set(c['branches'])=={'old_native','base','mix'},'Missing branch')
  for arm,b in c['branches'].items():check(b['passed'] is True and b['finite'] is True and b['forward_rng_unchanged'] is True and b['completed_steps']==1,'First update failed')
  check(c['branches']['old_native']['evidence']['sha256']==c['branches']['base']['evidence']['sha256'],'Reported old/base saved evidence hashes differ')
  check(c['old_native_vs_base']['optimizer_metadata_exact'] is True and c['mix_target_exact']['bitwise_equal'] is True,'Adam metadata or target mismatch')
  for arm in ('base','mix'):graph_check(c['branches'][arm]['graph'],arm,step,c['case'])
 check(set(v['replay'])=={'base','mix'},'Both replay arms required')
 for arm,r in v['replay'].items():
  check(r['passed'] is True and r['serialized_payload_exact'] is True and all(r['restoration'].values()) and r['final_payload_exact'] is True and r['graph_ledger_exact'] is True and r['checkpoint_snapshot_history_length']==1,'Replay restoration differs')
  for stage,step in [('first',0),('uninterrupted',1),('resumed',1)]:
   check(r['steps'][stage]['completed_steps']==step+1,'Replay step differs');graph_check(r['steps'][stage]['graph'],arm,step,'replay/'+stage)
 runtime=v['runtime'];check(runtime['torch']=='2.13.0+cu129' and runtime['cuda']=='12.9' and runtime['name']=='NVIDIA GB200' and runtime['threads']==2 and runtime['deterministic_algorithms'] is True and runtime['deterministic_warn_only'] is False and runtime['float32_matmul_precision']=='highest' and runtime['cublas_workspace_config']==':4096:8' and all(runtime[k] is False for k in ('tf32','amp','compile','ddp')),'Runtime contract differs')
 walk(v)
 out={'schema':'adaptgns_sand_graph_support_validation_independent_scalar_audit_v1','status':'passed','report_sha256':EXPECTED,'audit_source_sha256':sha(Path(__file__)),'scalar_checks':checks,'finite_numeric_scalars':finite_scalars,'exact_tensor_comparisons':comparisons,'elapsed_seconds':v['elapsed_seconds'],'exposure':exposure,'supports_root_timing_only_gate':True,'scientific_training_admitted':False,'old_cpu_cuda_validation_status':'validation_failed',
  'findings':['All three old-native/new-base comparisons pass exact tensor and optimizer metadata checks; their reported whole saved-evidence hashes are identical case by case.','Mix graph/target/finite-first-update checks pass: active selected pairs are small63+0, median0+0, large1315+1195. The median mix case is an unexpanded control.','Both arms pass new-schema checkpoint serialization/restoration, CPU+CUDA RNG probes and resumed/uninterrupted second update/final payload checks.'],
  'limits':['Scalar/source audit only: raw PT/NPZ evidence bytes and GPU execution are not independently re-executed or rehashed here.','For seed0 graph step1 both exposure coins are false: mix replay restores a state after63 optional pairs atstep0 and verifies a subsequent unexpanded step. It does not directly test replay of an actively expanded post-resume step.','Only first updates on three saved batches and two-update replay are covered; no100k trajectory equivalence, second-host reproduction or scientific checkpoint admission follows.','Prior CPU/CUDA gradient/Adam discrepancies remain failed and unchanged.'],
  'interpretation':'The bounded same-device implementation evidence supports a distinct root-reviewed infrastructure timing gate for this unchanged graph trainer. Scientific release still needs complete measured capacity/evaluation costs and final protocol review.'}
 path=P/'sand_graph_support_validation_v1_independent_scalar_audit.json';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'path':str(path),'sha256':sha(path),'checks':checks,'exact_tensor_comparisons':comparisons,'finite_numeric_scalars':finite_scalars}))
if __name__=='__main__':main()
