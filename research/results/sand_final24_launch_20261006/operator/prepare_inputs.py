"""Local scalar/provenance resolution only; never reads arrays or checkpoints."""
from pathlib import Path
import hashlib,json

PK=Path(__file__).resolve().parent;P=PK.parent;WORKSPACE=P.parents[2]
R='/root/repos/AdaptGNS-cuda-20261006';C=R+'/cuda_preparation'
BASE=P/'sand_final24_B_amendment_paths_UNADMITTED_transition_v1'
BASE_SHA='0c438762373ea714ba1f415f509c756bd58f2297c8887e8c4a230b3e066472d0'
AMEND=P/'sand_B_preflight_amendment_UNADMITTED_transition_v1'
CA=P/'sand_reserved_released_1808_root_v1';CB=P/'sand_B_preflight_amendment_released_root_v1'/'attempt2'
ROOT_COMPLETION='db7b953ccee7c8a661ebe7f2016065791f3dcdd17a99c4ea6749a0dc213bb1d8'
REVIEWS={
 'A':{'valid':'4a38f5bfda7b5894543d557baff01617b0c79e446eccdac039b17574719c5cd0','test':'b8acf9d47f4a92266735acd652a1a0a5e909628c94e9521fdecd53964e6c36db'},
 'B':{'valid':'5be588cd1c42270fccadbdca6e93ebb9cbbc2739b1dc8fe52c58d2ca9422c73c','test':'1611b9c885d9d34ebdc9569c3a30c7d080713fa6531bbda7197ddbc89331815a'}}
def need(x,s):
 if not x:raise ValueError(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def pin(p,h,records):
 p=Path(p);need(p.is_file() and not p.is_symlink() and p.suffix not in ('.npy','.npz','.pt','.pth'),'Only scalar/source evidence permitted')
 need(sha(p)==h,'Changed reviewed bytes: '+str(p));records[str(p.resolve())]=h;return read(p)
def pids(doc):
 out=set()
 if isinstance(doc,dict):
  for k,v in doc.items():
   if k=='native_absent' and isinstance(v,dict):
    need(all(x is True for x in v.values()),'Prior native closure is incomplete');out.update(int(x) for x in v)
   else:out.update(pids(v))
 elif isinstance(doc,list):
  for v in doc:out.update(pids(v))
 return out

def build():
 records={};manifest=pin(BASE/'manifest.json',BASE_SHA,records)
 for n,h in manifest['files_sha256'].items():need(sha(BASE/n)==h,'Candidate changed: '+n)
 completion=pin(P/'sand_B_amendment_root_completion_admission_v1.json',ROOT_COMPLETION,records)
 need(completion['status']=='both_B_preflights_completed_and_reviewed_inside_original_amendment' and completion['recorded_by']=='root' and completion['scientific_preflights_executed_once_each'] is True,'Actual root B completion required')
 need(completion['original_source_phase_disposition']=='seven_complete_two_unexecuted_unchanged','Preserve original incomplete phase')
 transfer=read(AMEND/'immutable_original_transfer_bindings.json');transfer=transfer.get('files_sha256',transfer)
 need(sha(AMEND/'manifest.json')=='72cfe26b2d8c13ea50d24b8d85e450ef228a4bf9ef681fb18a2de525f8640e6d','Frozen amendment bindings changed')
 for n,h in read(AMEND/'manifest.json')['files_sha256'].items():need(sha(AMEND/n)==h,'Amendment payload changed')
 census_remote=R+'/sand_reserved_preparation_20261006_v1/census/all_split_census.json'
 census=pin(CA/'collected'/census_remote.removeprefix(R+'/'),transfer[census_remote],records)
 need(census['status']=='complete_no_duplicates' and census['counts']=={'train':1000,'valid':30,'test':30} and len(census['numeric_files_sha256'])==2120,'Exact original2120-array census required')
 array_paths={root+'/'+kind+'_'+str(i).zfill(6)+'.npy' for root in (R+'/sand_numeric_train_valid_20261006_v1/valid',R+'/sand_reserved_preparation_20261006_v1/numeric/test') for kind in ('position','type') for i in range(30)}
 arrays={x:census['numeric_files_sha256'][x] for x in sorted(array_paths)};need(len(arrays)==120,'Exactly120 original evaluation arrays required')
 sources=read(BASE/'frozen_source_closure.json');profile=pin(Path(sources['profile_path']),sources['profile_sha256'],records)
 source_files={}
 for remote,h in sources['files_sha256'].items():
  local=Path(profile['files'][remote.removeprefix(R+'/')]['local_path']);need(sha(local)==h,'Frozen source changed: '+remote)
  relative='sources/'+remote.removeprefix(R+'/');dest=PK/relative;dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.open('xb') as f:f.write(local.read_bytes())
  source_files[remote]={'package_path':relative,'sha256':h}
 need(len(source_files)==23,'Exact23 frozen sources required')
 old_native={'A':set(),'B':set()}
 for op in ('acquire','convert','census','candidate_valid','candidate_test','preflight_valid','preflight_test'):
  path=P/('sand_completed_source_'+op+'_independent_review_code_audit_v1.json');v=read(path)
  need(v['status']=='passed_original_mode_outputs_nativeclosure_and_scalar_semantics','Original A source review required');records[str(path.resolve())]=sha(path);old_native['A'].update(pids(v))
 cohort=pin(P/'sand_completed_cohort_independent_review_code_audit_v1.json','2901d4e8a559ab40c6631eb1df7112e49aa491d599e0bb9176f92ee05940da3a',records)
 need(cohort['status']=='passed_completed_six_model_cohort_and_original_native_closure','Frozen six-model cohort review required');old_native['A'].update(pids(cohort))
 for role in ('A','B'):
  path=P/'sand_post_training_root_controls_1750_v2'/role/'original_closure.json';v=read(path);records[str(path.resolve())]=sha(path);old_native[role].update(pids(v))
 roles={}
 for role,local in (('A',CA),('B',CB)):
  release=read(BASE/(role+'.evaluation_release.candidate.json'));base_pins=release['files_sha256'];pins=dict(base_pins)
  for path,h in list(pins.items()):
   if h is None and path in transfer:pins[path]=transfer[path]
  for split in ('valid','test'):
   op='preflight_'+split;review_path=P/(('sand_completed_source_' if role=='A' else 'sand_completed_B_amendment_attempt2_')+op+'_independent_review_code_audit_v1.json')
   review=pin(review_path,REVIEWS[role][split],records)
   need(review['status']=='passed_original_mode_outputs_nativeclosure_and_scalar_semantics' and review['host_role']==role,'Accepted host-specific preflight required')
   for f,h in review['evidence_sha256'].items():pin(Path(f) if Path(f).is_absolute() else WORKSPACE/f,h,records)
   cpu=pin(local/(op+'.cpu_release.json'),review['operational_release_sha256'],records)
   core=pin(local/(op+'.mode_release.json'),review['mode_release_sha256'],records)
   remote=release['split_preflight'][split]['path'];report=pin(local/'collected'/remote.removeprefix(R+'/'),review['outputs_sha256'][remote],records)
   need(report['schema']=='adaptgns_sand_final_split_preflight_v1' and report['status']=='complete_split_contract_passed' and report['split']==split and report['frames']==320 and report['horizon']==314 and report['record_count']==30 and report['evaluation_executed'] is False,'Complete original split contract required')
   need(report['root_release_sha256']==review['mode_release_sha256'] and report['census_sha256']==transfer[census_remote] and report['preparation_source_sha256']=='6a40fe9196157c0666ffe556eb0e92781e9fc3b84f0c45b5ff67e09be0fce1b8','Actual preflight release/census/preparer lineage differs')
   need(all(cpu['inputs_sha256'].get(x)==h for x,h in core['files_sha256'].items()),'Reviewed core inputs differ')
   pins[remote]=review['outputs_sha256'][remote];release['split_preflight'][split]['sha256']=pins[remote];old_native[role].update(pids(review))
   if role=='B':
    pin(local/(op+'.root_original_external_exit.json'),review['original_tool_exit_receipt_sha256'],records)
    need(review['amendment_phase_sha256']==completion['phase_sha256'],'Same accepted B amendment required')
   boot=cpu['clock_sample']['host_boot_id']
   if split=='valid':role_boot=boot
   else:need(boot==role_boot,'Host boot differs across preflights')
  need(all(isinstance(h,str) and len(h)==64 for h in pins.values()),'Every scalar pin must resolve from accepted actual evidence')
  pins.update(arrays);need(len(pins)==(164 if role=='A' else 162),'Exact final immutable input count differs')
  release['files_sha256']=pins
  need(release['streams']==read(BASE/(role+'.evaluation_release.candidate.json'))['streams'],'Scientific argv changed')
  put(PK/(role+'.resolved_inputs.candidate.json'),release)
  roles[role]={'resolved_candidate':role+'.resolved_inputs.candidate.json','resolved_candidate_sha256':sha(PK/(role+'.resolved_inputs.candidate.json')),'files_sha256':pins,'expected_boot_id':role_boot,'historical_pids':sorted(old_native[role]),'hostname':release['host'],'owner_argv':read(BASE/'commands.json')['hosts'][role]['owner_argv']}
 forbidden=[C+'/'+n for n in ('supervise_sand_final_evaluation_scoped_v1.py','evaluate_sand_graph_support_final.py','train_sand_graph_support_cuda.py','supervise_sand_scoped_science_v1.py','supervise_sand_runtime_migration_recovery_v2.py','train_sand_runtime_migration_recovery_v1.py','supervise_sand_post_completion_cpu_v2.py','supervise_sand_reserved_source_cpu_v1.py','prepare_sand_reserved_test_recovered_v1.py','supervise_sand_B_preflight_amendment_cpu_v1.py','launch_sand_B_preflight_amendment_cpu_v1.py','prepare_sand_recovered_final_cohort_v1.py','prepare_sand_final_cohort_scoped_v1.py')]
 prepared={'schema':'sand_final24_prepared_inputs_v1','status':'complete_local_binding_pending_fresh_control_phase','candidate_manifest_sha256':BASE_SHA,'root_B_completion_sha256':ROOT_COMPLETION,'roles':roles,'source_files':source_files,'array_files_sha256':arrays,'original_local_evidence_sha256':records,'forbidden_program_tokens':forbidden,'phase_started':False,'scientific_execution':False,'execution_authority':False}
 put(PK/'prepared_inputs.json',prepared)
 print(json.dumps({'prepared_inputs_sha256':sha(PK/'prepared_inputs.json'),'role_input_counts':{r:len(v['files_sha256']) for r,v in roles.items()},'source_count':len(source_files),'array_count':len(arrays),'original_evidence_count':len(records)}))
if __name__=='__main__':build()
