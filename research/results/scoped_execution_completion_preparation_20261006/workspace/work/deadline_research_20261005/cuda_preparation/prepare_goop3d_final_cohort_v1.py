#!/usr/bin/env python3
"""Inventory stopped D3 science bytes and freeze its six-model cohort by root release.

No model deserialization, arrays, inference, process control or test access.
Tensor/Adam conclusions are inherited from the reviewed scientific supervisor;
this adapter independently checks source/receipt/pointer and retained file bytes.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SCHEMA='adaptgns_goop3d_final_cohort_freeze_v1'
INVENTORY_SCHEMA='adaptgns_goop3d_completed_science_byte_inventory_v1'
RELEASE_SCHEMA='adaptgns_goop3d_final_cohort_freeze_release_v1'
STOP_SCHEMA='adaptgns_goop3d_scientific_stopped_receipt_v1'
SUPERVISOR_SHA='7bf35efb06aa92bf84d3e30730412297aae9be62114a692f12e6ee80b55d107e'
EVALUATOR_SHA='9364dbbdd44d78212979f9cc6b381a545d0d875b9463d84d33155b3787c6a1de'
TRAINER_SHA='8de9b7d1de9435c1042151be49d0f6521b0a562ae9094d9e50060262ecad8fdc'
GRAPH_SHA='ca5898ec3f487d8138adf7479743156874a7796df09841f8ff7711b97536ad50'
PROTOCOL_SHA='5010f9023a3eee45f85bee80b35fc8e506ca145667daf75027b32c92658faa70'
DEADLINE=datetime(2026,10,7,1,tzinfo=timezone.utc)
PAIR_FLAGS=('initial_model_tensor_identity','initial_cpu_cuda_rng_identity','initial_empty_adam_identity',
 'all_frame_noise_lr_schedules_equal','all_graph_budgets_and_rng_material_verified')


def require(v,message):
    if not v:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def encode(obj):return (json.dumps(obj,indent=2,allow_nan=False)+'\n').encode()
def utc():return datetime.now(timezone.utc)
def stamp(value):
    x=datetime.fromisoformat(value);require(x.tzinfo is not None,'Timezone required');return x.astimezone(timezone.utc)
def write(path,obj):
    with Path(path).open('xb') as f:f.write(encode(obj))
def digest(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)


def load(name,pin,label):
    path=HERE/name;require(sha(path)==pin,'Reviewed source differs: '+name)
    spec=importlib.util.spec_from_file_location(label,path);module=importlib.util.module_from_spec(spec)
    sys.modules[label]=module;spec.loader.exec_module(module);return module


def safe(root,name):
    p=Path(name);require(not p.is_absolute() and '..' not in p.parts and p.parts,'Relative retained artifact required')
    path=Path(root)/p;require(not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve()),'Artifact leaves science tree')
    return path


def tree_inventory(root):
    root=Path(root).resolve();require(root.is_dir(),'Science tree absent')
    found={}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(),'Retained science symlink refused')
        if not p.is_file():continue
        before=p.stat();d=sha(p);after=p.stat()
        require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Artifact changed while hashing')
        found[str(p.relative_to(root))]={'sha256':d,'size_bytes':after.st_size}
    require(found,'Empty science tree');return found


def inventory(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    require(not output.is_relative_to(root) and not output.exists(),'Inventory output must be fresh and outside science tree')
    files=tree_inventory(root)
    result={'schema':INVENTORY_SCHEMA,'status':'retained_bytes_inventoried_requires_root_review','science_dir':str(root),
        'files':files,'created_utc':utc().isoformat(),'source_sha256':sha(__file__),
        'model_deserialization_performed':False,'scientific_admission_or_test_access':False}
    write(output,result);return result


def checked_receipts(root,files,S):
    def saved(name):
        p=safe(root,name);require(name in files and sha(p)==files[name]['sha256'],'Missing/changed scalar receipt '+name);return read(p)
    launch,summary,status,candidate,audit,release=[saved(n) for n in
        ('launch.json','summary.json','status.json','cohort_candidate.json','cohort_audit_candidate.json','inputs/release.json')]
    U=summary.get('endpoint_updates')
    require(type(U) is int and U>=2 and summary.get('schema')==S.SCHEMA
        and summary.get('status')=='all_six_scientific_endpoints_verified'
        and summary.get('all_inputs_and_worker_artifacts_reverified') is True
        and summary.get('whole_cohort_frozen_for_evaluation') is False and summary.get('test_accessed') is False,'Complete scientific endpoint summary required')
    require(status.get('state')=='complete_scientific_training' and status.get('summary_sha256')==files['summary.json']['sha256']
        and status.get('whole_cohort_frozen_for_evaluation') is False and status.get('test_accessed') is False,'Complete matching scientific status required')
    require(launch.get('schema')==S.SCHEMA and launch.get('schedule')==S.SCHEDULE and launch.get('endpoint_updates')==U
        and launch.get('release_sha256')==files['inputs/release.json']['sha256'] and launch.get('environment')==S.ENVIRONMENT,'Frozen launch identity differs')
    hashes=launch['files_sha256'];snapshots=launch['input_snapshots']
    T=load('train_goop3d_graph_support_cuda_v2.py',TRAINER_SHA,'_goop3d_freeze_trainer_contract')
    expected_keys=(set(S.FIELDS)-{'repo'})|{'supervisor','trainer','graph','lifecycle','validator','worksheet_source',
        'metadata','auxiliary_validator','original_graph'}|{'core:'+k for k in T.SOURCE_PINS}|{'context:'+k for k in T.CONTEXT_SOURCE_PINS}
    require(set(hashes)==expected_keys and all(hashes['core:'+k]==v for k,v in T.SOURCE_PINS.items())
        and all(hashes['context:'+k]==v for k,v in T.CONTEXT_SOURCE_PINS.items())
        and hashes['auxiliary_validator']==T.AUXILIARY_VALIDATOR_SHA,'Complete pinned scientific source/context closure required')
    require(set(snapshots)==set(hashes)-{'python'} and all(files.get(p,{}).get('sha256')==hashes[k] for k,p in snapshots.items()),'Complete frozen input snapshot hashes required')
    require(all(hashes.get(k)==v for k,v in {'supervisor':SUPERVISOR_SHA,'trainer':TRAINER_SHA,'graph':GRAPH_SHA,
        'protocol':PROTOCOL_SHA,'lifecycle':S.LIFECYCLE_SHA,'validator':S.VALIDATOR_SHA,'worksheet_source':S.WORKSHEET_SHA,
        'train_manifest':'0f0ce1802e202b9faaf86f0a6ce059c286ed454ceb53273cb926980614b0f864',
        'metadata':'727cec55bc529c142596954773284e897aaed257e907ab6439d5c367853e4a55'}.items()),'D3 source/data/protocol lineage differs')
    V=load('validate_goop3d_vectorized_cuda_v1.py',S.VALIDATOR_SHA,'_goop3d_freeze_data_contract')
    require(all(hashes.get(k)==v for k,v in V.DATA_PINS.items() if k in expected_keys)
        and hashes['original_graph']==V.ORIGINAL_SHA,'Complete pinned D3 data-support/original-graph lineage required')
    require(release.get('schema')==S.RELEASE_SCHEMA and release.get('status')=='admitted_for_scientific_training'
        and release.get('issued_by')=='root' and release.get('scientific_training_admitted') is True
        and release.get('files_sha256')==hashes and release.get('endpoint_updates')==U and release.get('output_dir')==str(root)
        and release.get('schedule')==S.SCHEDULE and release.get('hostname')==launch.get('hostname')
        and release.get('gpu_uuids')==launch.get('gpu_uuids') and release.get('cohort_id')==summary.get('cohort_id')==launch.get('cohort_id'),
        'Original concrete root scientific release differs')
    admission=saved(snapshots['admission']);plan=saved(snapshots['endpoint_plan'])
    require(admission.get('status')=='admitted' and admission.get('issued_by')=='root' and admission.get('scientific_training_admitted') is True
        and admission.get('prospective_endpoint_updates')==U and 'scope' not in admission
        and plan.get('status')=='prospectively_selected' and plan.get('issued_by')=='root' and plan.get('endpoint_updates')==U
        and plan.get('selection_used_validation_or_test_accuracy') is False,'Fresh prospectively selected scientific admission required')
    jobs=summary.get('jobs',[]);pairs=summary.get('paired_seeds',[])
    require(len(jobs)==6 and all(type(j.get('seed')) is int for j in jobs) and {(j.get('arm'),j.get('seed')) for j in jobs}=={(a,s) for a in ('base','mix') for s in range(3)},'Complete six-cell scientific cohort required')
    require(len(pairs)==3 and all(type(p.get('seed')) is int for p in pairs) and {p.get('seed') for p in pairs}=={0,1,2} and all(all(p.get(k) is True for k in PAIR_FLAGS) for p in pairs),'Complete initial/RNG/frame/graph pairing required')
    for j in jobs:
        expected=next(x for x in S.SCHEDULE if x['id']==j['id']);directory=root/'jobs'/j['id'];base='jobs/'+j['id']+'/'
        require(all(j.get(k)==v for k,v in expected.items()) and j.get('status')=='verified_scientific_endpoint'
            and Path(j['directory'])==directory and j.get('history',{}).get('graph_updates')==U,'Verified endpoint cell identity differs')
        require(j.get('initial_checkpoint',{}).get('completed_steps')==0 and j.get('endpoint',{}).get('completed_steps')==U
            and all(j['endpoint'].get(k) is True and j['initial_checkpoint'].get(k) is True for k in
                ('all_optimizer_steps_equal_endpoint','all_state_and_moments_finite','cpu_cuda_rng_serialized','simulator_config_verified')),
            'Reported initial/final tensor/Adam audit incomplete')
        for name,d in {**j['checkpoint_sha256'],**j['artifact_sha256']}.items():
            require(files.get(base+name,{}).get('sha256')==d,'Worker checkpoint/scalar byte inventory differs')
        require({n[len(base):] for n in files if n.startswith(base+'checkpoint-')}==set(j['checkpoint_sha256']),'Checkpoint inventory coverage differs')
        for key,n in (('initial_pointer',0),('final_pointer',U)):
            ptr=j[key];require(ptr.get('completed_steps')==n and ptr.get('path')==f'checkpoint-{n:09d}.pt'
                and ptr.get('run_config_sha256')==j['config_sha256'] and files.get(base+ptr['path'],{}).get('sha256')==ptr.get('sha256'),'Endpoint pointer/configuration differs')
        config=saved(base+'protocol.json')
        require(hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==j['config_sha256']
            and config.get('schema')==T.SCHEMA and config.get('dataset')=='Goop-3D' and type(config.get('seed')) is int
            and all(config.get(k)==j[k] for k in ('objective','arm','seed'))
            and config.get('updates')==config.get('prospective_endpoint_updates')==U
            and config.get('research_protocol_sha256')==hashes['protocol']
            and config.get('source_sha256')=={**T.SOURCE_PINS,'train_goop3d_graph_support_cuda_v2.py':TRAINER_SHA,'goop3d_graph_support_vectorized_v1.py':GRAPH_SHA}
            and config.get('data',{}).get('manifest_sha256')==hashes['train_manifest']
            and config.get('data',{}).get('admission_sha256')==hashes['admission'], 'Independent worker configuration/source/endpoint digest differs')
        ex=j['external'];worker=saved(base+'status.json')
        require(ex.get('exit_code')==0 and ex.get('signals')==[] and ex.get('command')==launch['commands'][j['id']]
            and ex.get('pid')==ex.get('identity',{}).get('pid')==worker.get('process',{}).get('pid')
            and worker.get('state')=='complete' and worker.get('error') is None
            and worker.get('run_config_sha256')==j['config_sha256'] and worker.get('latest_checkpoint')==j['final_pointer']
            and saved(base+'latest.json')==j['final_pointer'] and worker.get('completed_steps')==worker.get('committed_steps')==worker.get('requested_steps')==U,
            'Complete stopped worker identity/status required')
    require(not any(Path(n).name=='run.lock' or n.endswith('.tmp') or 'unsuccessful' in Path(n).name
        or 'state_preservation_error' in Path(n).name for n in files),'Unsuccessful/live artifacts retained, never admitted')
    waves={w:S.verify_wave_receipt(SimpleNamespace(output_dir=root),w,launch,[j for j in jobs if j['wave']==w]) for w in ('A','B')}
    require(stamp(waves['A']['ended_utc'])<=stamp(waves['B']['started_utc']),'Sequential A/B chronology differs')
    for name in ('cohort_candidate.json','cohort_audit_candidate.json'):
        require(summary['candidate_files_sha256'].get(name)==files[name]['sha256'],'Supervisor candidate byte binding differs')
    require(candidate.get('status')=='candidate_requires_root_freeze' and audit.get('status')=='all_six_verified_candidate_requires_root_review'
        and candidate.get('issued_by')==audit.get('issued_by')=='scientific_supervisor_not_root'
        and candidate.get('evaluation_admitted') is False and audit.get('evaluation_admitted') is False
        and candidate.get('cohort_audit_sha256')==files['cohort_audit_candidate.json']['sha256'],'Original unadmitted candidates required')
    require(all(type(m.get('seed')) is int for m in candidate.get('models',[])+audit.get('models',[]))
        and all(type(p.get('seed')) is int for p in audit.get('paired_seeds',[]))
        and candidate.get('models')==audit.get('models') and audit.get('paired_seeds')==pairs
        and candidate.get('cohort_id')==audit.get('cohort_id')==summary.get('cohort_id'),'Candidate/supervisor model-pairing/cohort correspondence differs')
    for model in candidate['models']:
        j=next(j for j in jobs if (j['arm'],j['seed'])==(model['arm'],model['seed']))
        require(model['checkpoint_sha256']==j['final_pointer']['sha256'] and model['checkpoint_path']==str(root/'jobs'/j['id']/j['final_pointer']['path'])
            and model['config_sha256']==j['config_sha256'] and model['completed_steps']==model['graph_history_updates']==U,'Candidate final pointer differs')
    require(len({m['checkpoint_sha256'] for m in candidate['models']})==6,'Distinct six final checkpoints required')
    return launch,summary,candidate,audit,waves


def freeze(args,now=utc):
    root=args.science_dir.resolve();initial={str(p.resolve()):sha(p) for p in (args.inventory,args.stopped_receipt,args.root_release,Path(__file__))}
    release=read(args.root_release);inv=read(args.inventory);stop=read(args.stopped_receipt)
    require(release.get('schema')==RELEASE_SCHEMA and release.get('status')=='approved_for_byte_checked_cohort_freeze'
        and release.get('issued_by')=='root' and release.get('source_sha256')==sha(__file__)
        and release.get('science_dir')==str(root) and release.get('output_dir')==str(args.output_dir.resolve())
        and release.get('inventory_sha256')==initial[str(args.inventory.resolve())]
        and release.get('stopped_receipt_sha256')==initial[str(args.stopped_receipt.resolve())]
        and release.get('supervisor_sha256')==SUPERVISOR_SHA and release.get('evaluator_sha256')==EVALUATOR_SHA
        and release.get('independent_completed_cohort_review') is True,'Explicit exact root cohort-freeze release required')
    require(inv.get('schema')==INVENTORY_SCHEMA and inv.get('status')=='retained_bytes_inventoried_requires_root_review'
        and inv.get('science_dir')==str(root) and inv.get('files')==tree_inventory(root),'Complete current retained-byte inventory required')
    S=load('supervise_goop3d_science_v1.py',SUPERVISOR_SHA,'_goop3d_freeze_supervisor')
    E=load('evaluate_goop3d_graph_support_v1.py',EVALUATOR_SHA,'_goop3d_freeze_evaluator')
    launch,summary,candidate,audit,waves=checked_receipts(root,inv['files'],S)
    U=summary['endpoint_updates'];require(release.get('endpoint_updates')==U,'Root endpoint differs from completed cohort')
    require(stop.get('schema')==STOP_SCHEMA and stop.get('issued_by')=='root'
        and stop.get('status')=='all_scientific_processes_stopped' and stop.get('science_dir')==str(root)
        and stop.get('summary_sha256')==inv['files']['summary.json']['sha256'] and stop.get('hostname')==launch['hostname']
        and stop.get('matching_owned_processes')==[] and stop.get('gpu_processes')==[]
        and stop.get('all_owned_processes_reaped_or_independently_verified_absent') is True
        and {launch['pid'],*(j['external']['pid'] for j in summary['jobs'])}<=set(stop.get('known_owned_pids',[]))
        and len(stop['known_owned_pids'])==len(set(stop['known_owned_pids'])) and all(type(p) is int and p>0 for p in stop['known_owned_pids'])
        and stamp(stop['checked_utc'])>=stamp(waves['B']['ended_utc']),'Fresh root post-cohort stopped-process receipt required')
    output=args.output_dir.resolve();require(not output.exists() and output!=root and not output.is_relative_to(root) and not root.is_relative_to(output),'Fresh separate cohort output required')
    output.mkdir(parents=True,mode=0o700)
    try:
        frozen_utc=now().isoformat();common={'issued_by':'root','created_utc':frozen_utc,'evaluation_admitted':True,
            'freeze_adapter_sha256':initial[str(Path(__file__).resolve())],'root_freeze_release_sha256':initial[str(args.root_release.resolve())],
            'completed_science_inventory_sha256':initial[str(args.inventory.resolve())],'source_summary_sha256':inv['files']['summary.json']['sha256'],
            'tensor_audit_scope':'Inherited from reviewed completed scientific supervisor; this adapter independently rechecks bytes/receipts, not model tensors.'}
        audit={**audit,**common,'status':'all_six_endpoints_and_pairing_verified'}
        write(output/'cohort_audit.json',audit)
        candidate={**candidate,**common,'status':'frozen_for_final_evaluation','cohort_audit_sha256':sha(output/'cohort_audit.json')}
        protocol=safe(root,launch['input_snapshots']['protocol']);trainer=HERE/'train_goop3d_graph_support_cuda_v2.py'
        require(sha(trainer)==TRAINER_SHA,'Frozen trainer bytes differ')
        for model in candidate['models']:
            E.cohort_gate(candidate,audit,{'scientific_endpoint_updates':U},SimpleNamespace(protocol=protocol,trainer_source=trainer,
                cohort_audit=output/'cohort_audit.json',arm=model['arm'],seed=model['seed'],checkpoint_sha256=model['checkpoint_sha256']))
        report={'schema':SCHEMA,'status':'validated_root_cohort_candidate_before_last_publication','created_utc':frozen_utc,
            'endpoint_updates':U,'science_dir':str(root),'cohort_sha256':hashlib.sha256(encode(candidate)).hexdigest(),
            'cohort_audit_sha256':sha(output/'cohort_audit.json'),'input_sha256':initial,'all_six_frozen_evaluator_scalar_gates_passed':True,
            'model_deserialization_performed':False,'test_accessed':False,'processes_launched':0,'inherited_tensor_audit_scope':common['tensor_audit_scope']}
        write(output/'freeze_report.json',report)
        require(tree_inventory(root)==inv['files'] and all(sha(p)==v for p,v in initial.items()),'Input/science bytes changed during cohort freeze')
        require(sha(HERE/'supervise_goop3d_science_v1.py')==SUPERVISOR_SHA and sha(HERE/'evaluate_goop3d_graph_support_v1.py')==EVALUATOR_SHA
            and sha(trainer)==TRAINER_SHA and sha(HERE/'validate_goop3d_vectorized_cuda_v1.py')==S.VALIDATOR_SHA
            and sha(output/'cohort_audit.json')==candidate['cohort_audit_sha256']
            and sha(output/'freeze_report.json')==hashlib.sha256(encode(report)).hexdigest(),'Source/output evidence changed before cohort publication')
        current=now();require(stamp(stop['checked_utc'])<=current<DEADLINE and (current-stamp(stop['checked_utc'])).total_seconds()<=300,'Stopped receipt stale or freeze deadline reached')
        write(output/'cohort.json',candidate)
        return report
    except BaseException as error:
        write(output/'failed_freeze.json',{'schema':SCHEMA,'status':'failed_cohort_freeze','error_type':type(error).__name__,
            'error':str(error),'all_outputs_retained':True,'test_accessed':False,'cohort_published':(output/'cohort.json').exists()});raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--execute',action='store_true');p.add_argument('--mode',choices=('inventory','freeze'))
    for name in ('science-dir','output-file','inventory','stopped-receipt','root-release','output-dir'):p.add_argument('--'+name,type=Path)
    a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps({'schema':SCHEMA,'status':'description_only','test_accessed':False,'model_deserialization':False,'processes_launched':0}));return 0
    require(a.science_dir and a.mode,'Explicit science directory and mode required')
    if a.mode=='inventory':require(a.output_file,'Fresh inventory output required');inventory(a.science_dir,a.output_file)
    else:
        require(all((a.inventory,a.stopped_receipt,a.root_release,a.output_dir)),'Explicit root inventory/stopped/freeze/output inputs required');freeze(a)
    return 0


if __name__=='__main__':raise SystemExit(main())
