#!/usr/bin/env python3
"""Independent scalar-only audit of completed base-graph capacityv2 summary."""
import hashlib,json,math,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'sand_capacity_v2_summary.json'
MANIFEST=HERE/'train.json'
EXPECTED='de5ca0bc4d08c7e5d789ed2e542ce03314d19a3144bdf3460933c225bbccc8fa'
checks=0
lr_rounding=[]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def check(value,message):
 global checks
 checks+=1
 if not value:raise ValueError(message)
def close(a,b):return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)

def main():
 check(sha(SOURCE)==EXPECTED,'Summary byte identity changed')
 check(sha(MANIFEST)=='f133a629a0d94b67875267245ad7bda67464731caf83e2411743a5a85a71bb1f','Training scalar manifest changed')
 v=json.loads(SOURCE.read_text());manifest=json.loads(MANIFEST.read_text());counts={r['id']:r['positions']['shape'][1] for r in manifest['records']}
 check(v['schema']=='adaptgns_sand_cuda_capacity_v2' and v['status']=='all_six_verified' and v['scientific_training_admitted'] is False,'Unexpected study/status')
 jobs={j['id']:j for j in v['jobs']}
 check(len(jobs)==len(v['jobs'])==6 and set(jobs)=={f'{o}_seed{s}' for o in ('faithful','nll') for s in range(3)},'Exact six jobs required')
 values=[]
 for name,j in jobs.items():
  rows=j['pairing_rows'];check(len(rows)==len(j['all512_timings'])==512,'Complete per-update timing required')
  check(j['seed'] in (0,1,2) and name==f"{j['objective']}_seed{j['seed']}" and j['wave']==('B' if j['seed']==2 else 'A'),'Schedule identity')
  previous=0.
  for step,(row,timing) in enumerate(zip(rows,j['all512_timings']),1):
   check(row['completed_steps']==step and timing=={k:row[k] for k in ('completed_steps','guarded_update_seconds','elapsed_seconds')},'Timing/row mismatch')
   check(all(math.isfinite(row[k]) for k in ('loss','lr','guarded_update_seconds','elapsed_seconds')) and row['guarded_update_seconds']>0 and row['elapsed_seconds']>previous,'Finite/ordered row required');previous=row['elapsed_seconds']
   expected_lr=1e-4*(1e-5/1e-4)**((step-1)/99999)
   check(abs(row['lr']-expected_lr)<=math.ulp(expected_lr),'Learning-rate schedule differs beyond1float64ULP')
   if row['lr']!=expected_lr:lr_rounding.append({'id':name,'step':step,'saved':row['lr'],'local_recomputed':expected_lr,'ulps':abs(row['lr']-expected_lr)/math.ulp(expected_lr)})
   n=0
   for identity in row['frame_ids']:
    trajectory,target=identity.rsplit(':',1);check(trajectory in counts and 6<=int(target)<320,'Frame outside admitted source');n+=counts[trajectory]
   check(len(row['frame_ids'])==2 and row['particles']==n,'Frame-particle count mismatch')
  guarded=[r['guarded_update_seconds'] for r in rows];steady=guarded[64:];ordered=sorted(steady);x=.9*(len(steady)-1);lo=math.floor(x);p90=ordered[lo]+(x-lo)*(ordered[lo+1]-ordered[lo])
  q=(rows[-1]['elapsed_seconds']-rows[63]['elapsed_seconds'])/448
  residual=j['external']['elapsed_seconds']-sum(guarded)
  check(j['external']['exit_code']==0 and residual>=0,'External outcome/residual invalid')
  for actual,expected in [(j['steady_wall_seconds_per_update'],q),(j['external_minus_all_guarded_seconds'],residual),(j['nonnegative_external_minus_all_guarded_seconds'],max(0,residual)),(j['sum_all512_guarded_seconds'],sum(guarded)),(j['sum_warmup64_guarded_seconds'],sum(guarded[:64])),(j['warmup_excess_over_steady_guarded_mean_seconds'],sum(guarded[:64])-64*statistics.mean(steady))]:check(close(actual,expected),'Timing arithmetic differs')
  for key,expected in [('mean',statistics.mean(steady)),('median',statistics.median(steady)),('p90_linear',p90),('maximum',max(steady))]:check(close(j['steady_guarded_seconds'][key],expected),'Steady summary arithmetic differs')
  values.append({'id':name,'wave':j['wave'],'q_wall_seconds_per_update':q,'external_residual_seconds':residual})
 paired=0
 for seed in range(3):
  for a,b in zip(jobs[f'faithful_seed{seed}']['pairing_rows'],jobs[f'nll_seed{seed}']['pairing_rows']):
   check(all(a[k]==b[k] for k in ('completed_steps','frame_ids','particles','lr')),'Paired row mismatch');paired+=1
 q4=max(j['q_wall_seconds_per_update'] for j in values if j['wave']=='A');q2=max(j['q_wall_seconds_per_update'] for j in values if j['wave']=='B')
 r4=max(j['external_residual_seconds'] for j in values if j['wave']=='A');r2=max(j['external_residual_seconds'] for j in values if j['wave']=='B')
 training=1.35*(100000*(q4+q2)+12*(r4+r2));f=v['forecast']
 for key,expected in [('q4',q4),('q2',q2),('r4',r4),('r2',r2),('training_seconds',training)]:check(close(f[key],expected),'Forecast formula differs')
 check(all(f[k] is None for k in ('complete_evaluation_seconds','diagnostics_execution_seconds','total_compute_analysis_seconds','forecast_finish_utc','fits_before_seven_hour_writing_reserve')),'Unmeasured complete cost/fit must remain null')
 out={'schema':'adaptgns_sand_base_capacity_independent_scalar_audit_v1','status':'passed','source_sha256':EXPECTED,'audit_source_sha256':sha(Path(__file__)),
      'scalar_checks':checks,'local_LR_recomputation_discrepancies':lr_rounding,'maximum_LR_recomputation_tolerance_float64_ulps':1,'paired_rows_checked':paired,'training_rows_checked':3072,'jobs':values,'q4':q4,'q2':q2,'r4':r4,'r2':r2,'training_seconds':training,'training_hours':training/3600,
      'complete_deadline_fit':None,'scientific_training_admitted':False,
      'scope':'Independent saved-scalar/manifest/arithmetic audit only; no arrays, checkpoints, models, CUDA or process launches. Original supervisor owns checkpoint/source/UUID/wave-byte validation.',
      'limitations':['q2 and q4 differ in seeds, physical assignment, concurrency and time; their ratio does not identify scaling.','This is the original base-graph faithful/NLL two-wave infrastructure measurement, not new graph-support six-parallel capacity.','Checkpoint residual and 1.35/12 allowances are planning terms, not probabilistic bounds; unmeasured full evaluation/diagnostics leave fit null.']}
 path=HERE/'sand_capacity_v2_independent_scalar_audit.json';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'audit':str(path),'sha256':sha(path),'checks':checks,'paired_rows':paired,'training_hours':training/3600}))
if __name__=='__main__':main()
