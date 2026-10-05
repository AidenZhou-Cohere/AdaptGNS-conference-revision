from pathlib import Path
import json,hashlib,statistics
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'work/graph-convention-bridge-20261005'
REPORT=BASE/'reports/graph_convention_bridge.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
summary=json.loads(REPORT.read_text());check_count=0;digest_before=sha(REPORT);scalar_runs={}
def check(condition,message):
 global check_count
 check_count+=1
 if not condition:raise ValueError(message)
def near(a,b,message):check(abs(a-b)<=1e-12*max(1.,abs(a),abs(b)),message)
for objective in ('faithful','nll'):
 for seed in (0,1,2):
  directory=BASE/f'{objective}_seed{seed}';index=json.loads((directory/'result.json').read_text());rows=[]
  for r in index['records']:
   path=directory/r['record_file'];check(sha(path)==r['record_sha256'],'input record hash')
   row=json.loads(path.read_text());check(row['status']=='complete','every declared frame complete');rows.append(row)
  for split in ('valid','test'):
   selected=[r for r in rows if r['split']==split]; trajectories=sorted(set(r['trajectory_id'] for r in selected));check(len(selected)==(128 if split=='valid' else 297),'population')
   means={}
   for case in selected[0]['cases']:
    for metric in ('normalized_coordinate_mse','position_coordinate_mse'):
     values=[statistics.mean(r['cases'][case]['metrics'][metric] for r in selected if r['trajectory_id']==t) for t in trajectories]
     means[case,metric]=statistics.mean(values)
   scalar_runs[objective,seed,split]=means
checks={}
for objective in ('faithful','nll'):
 for split in ('valid','test'):
  g=summary['groups'][objective][split]['measures']
  for (case,metric) in scalar_runs[objective,0,split]:
   values=[scalar_runs[objective,s,split][case,metric] for s in (0,1,2)]
   reported=g[f'case__{case}__{metric}'];near(statistics.mean(values),reported['mean'],'case objective mean');near(statistics.stdev(values),reported['sample_sd'],'case sample seed SD')
   for a,b in zip(values,reported['seed_values']):near(a,b,'case seed value')
  metric='normalized_coordinate_mse'
  baseline_reductions=[];risk_random=[];dense_base=[];fresh_previous=[]
  for seed in (0,1,2):
   m=scalar_runs[objective,seed,split]
   val=lambda p,l:m[f'{p}_uncapped_loops{l}',metric]
   baseline_reductions.append(100*(val('base',0)-val('base',1))/val('base',0))
   risk_random.append(val('previous-observed-base-risk25',1)-val('random25',1))
   dense_base.append(val('dense',1)-val('base',1))
   fresh_previous.append(val('current-base-risk25',1)-val('previous-observed-base-risk25',1))
   for p,ref in [('previous-observed-base-risk25','random25'),('dense','base')]:
    effect=(val(p,1)-val(ref,1))-(val(p,0)-val(ref,0))
    near(effect,g[f'contrast__loop_interaction__{p}_minus_{ref}__{metric}']['seed_values'][seed],'primary paired interaction')
  stats=lambda v:{'seed_values':v,'mean':statistics.mean(v),'sample_sd':statistics.stdev(v)}
  checks[objective+'/'+split]={'base_mse_relative_reduction_percent':stats(baseline_reductions),'loops1_risk_minus_random':stats(risk_random),'loops1_dense_minus_base':stats(dense_base),'loops1_current_minus_previous_risk':stats(fresh_previous)}
check(sha(REPORT)==digest_before,'summary bytes unchanged')
out={'passed':True,'checks':check_count,'summary_sha256':digest_before,'source_sha256':sha(Path(__file__)),'findings':checks,'scope':'Independent scalar aggregation and primary interaction reconstruction from all2550 hashed original frame records; no inference. Full array/geometry/source audit performed by separately reviewed strict summarizer.'}
(ROOT/'work/deadline_research_20261005/graph_bridge_aggregate_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
