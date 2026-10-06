#!/usr/bin/env python3
"""Root-gated recovered-cohort Sand test acquisition; frozen conversion and census.

Default only describes. Every operation follows complete six100k cohort freeze;
no model/tensor loading or inference. All modes share one<=900second preparation
window, ending before final evaluation's latest start. Source/failed staging stay.
"""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SCHEMA='adaptgns_sand_reserved_test_preparation_scoped_v1'
RELEASE_SCHEMA='adaptgns_sand_reserved_test_preparation_release_scoped_v1'
COHORT_ADAPTER_SHA='41bef815edf151fddd9ef9d667a339b5cb0f46ae405d8e4ed6bf00df69e75ced'
DOWNLOADER_SHA='864d9c974deabceb19b5f3b8e88ced1ea7281595a0d3a526e8d39f382aa4605a'
CONVERTER_SHA='37c135a9eaa484a5fdbed2176780913b71cd3be4df602983fcd3b644c1a57ff9'
EVALUATOR_SHA='952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58'
BENCH_SHA='8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13'
PROTOCOL_SHA='e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d'
TRAINER_SHA='fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124'
TRAIN_ADMISSION_SHA='fbc8ddd94d90adea6e8f4b07caa73931dfcea41e86c8f2f830e3f01f6a177b73'
TRAIN_MANIFEST_SHA='f133a629a0d94b67875267245ad7bda67464731caf83e2411743a5a85a71bb1f'
METADATA_SHA='cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0'
AMENDMENT_SHA='411547c3355859906f59792b3b493841030b0afd2e105e17e466114df6743738'
PLAN_SHA='403a96c1304341ea1899c994e0ab7474280add44ea5986ec41570e08d16c337b'
COUNTS={'train':1000,'valid':30,'test':30}
TEST_BYTES=85825802
LAST_EVALUATION_START=datetime(2026,10,6,23,44,tzinfo=timezone.utc)


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_bytes())
def stamp(value):
    t=datetime.fromisoformat(value);require(t.tzinfo is not None,'Timezone-aware timestamp required');return t.astimezone(timezone.utc)
def now():return datetime.now(timezone.utc)
def write(path,value):
    path=Path(path)
    with path.open('x') as f:f.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def load(name,pin,label):
    p=HERE/name;require(sha(p)==pin,'Reviewed source differs: '+name)
    spec=importlib.util.spec_from_file_location(label,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def cohort_gate(cohort,audit):
    models=cohort.get('models',[]);expected={(a,s) for a in ('base','mix') for s in range(3)}
    require(cohort.get('schema')=='adaptgns_sand_graph_support_final_cohort_v1' and cohort.get('status')=='frozen_for_final_evaluation'
        and cohort.get('issued_by')=='root' and cohort.get('dataset')=='Sand' and cohort.get('updates')==100000
        and cohort.get('protocol_sha256')==PROTOCOL_SHA and cohort.get('training_admission_sha256')==TRAIN_ADMISSION_SHA
        and cohort.get('trainer_source_sha256')==TRAINER_SHA and cohort.get('adapter_sha256')==COHORT_ADAPTER_SHA
        and cohort.get('operational_amendment_sha256')==AMENDMENT_SHA and cohort.get('schedule_plan_sha256')==PLAN_SHA,
        'Complete reviewed Sand scoped cohort required before test access')
    require(len(models)==6 and {(m.get('arm'),m.get('seed')) for m in models}==expected
        and all(type(m.get('seed')) is int and m.get('objective')=='faithful' and m.get('completed_steps')==100000 for m in models)
        and all(isinstance(m.get('checkpoint_sha256'),str) and len(m['checkpoint_sha256'])==64
            and all(c in '0123456789abcdef' for c in m['checkpoint_sha256']) for m in models)
        and len({m['checkpoint_sha256'] for m in models})==6
        and cohort.get('policies')==['base','dense','random25','speed25','laggedrisk25','relative-velocity-RMS25'],
        'All six distinct fixed100k endpoints and six declared policies required')
    require(audit.get('schema')=='adaptgns_sand_graph_support_complete_cohort_audit_v1'
        and audit.get('status')=='all_six_endpoints_and_pairing_verified' and audit.get('adapter_sha256')==COHORT_ADAPTER_SHA
        and audit.get('cohort_id')==cohort.get('cohort_id') and audit.get('trainer_source_sha256')==TRAINER_SHA
        and audit.get('issued_by')=='root' and audit.get('protocol_sha256')==PROTOCOL_SHA
        and audit.get('training_admission_sha256')==TRAIN_ADMISSION_SHA,
        'Matching complete source/tensor/pairing audit required')
    rows=audit.get('models',[]);pairs=audit.get('paired_seeds',[])
    require(len(rows)==6 and {(m.get('arm'),m.get('seed')) for m in rows}==expected,'Complete endpoint audit grid required')
    for model in models:
        ev=next(m for m in rows if (m['arm'],m['seed'])==(model['arm'],model['seed']))
        require(ev.get('checkpoint_sha256')==model['checkpoint_sha256'] and ev.get('completed_steps')==ev.get('graph_history_updates')==100000
            and ev.get('checkpoint_every')==10000 and ev.get('log_every')==100 and all(ev.get(k) is True for k in
            ('all_optimizer_steps_equal_100000','all_state_and_moments_finite','source_data_protocol_verified','checkpoint_bytes_verified')),
            'Incomplete endpoint proof')
    require(len(pairs)==3 and {p.get('seed') for p in pairs}=={0,1,2} and all(type(p.get('seed')) is int and all(p.get(k) is True for k in
        ('initial_model_tensor_identity','initial_cpu_cuda_rng_identity','all_frame_noise_lr_schedules_equal','all_graph_budgets_and_rng_material_verified')) for p in pairs),
        'Complete three-seed initialization/schedule pairing required')


def gate(args):
    raw={k:getattr(args,k).read_bytes() for k in ('cohort','cohort_audit','root_release')}
    cohort,audit,release=[json.loads(raw[k]) for k in ('cohort','cohort_audit','root_release')]
    hashes={k:hashlib.sha256(v).hexdigest() for k,v in raw.items()}
    cohort_gate(cohort,audit)
    require(cohort.get('cohort_audit_sha256')==hashes['cohort_audit'] and release.get('schema')==RELEASE_SCHEMA
        and release.get('issued_by')=='root' and release.get('status')=='approved_for_'+args.mode
        and release.get('preparation_source_sha256')==sha(__file__) and release.get('cohort_sha256')==hashes['cohort']
        and release.get('cohort_audit_sha256')==hashes['cohort_audit'] and release.get('output_dir')==str(args.output_dir)
        and release.get('no_retry_or_checkpoint_selection') is True,'Exact mode-specific root preparation release required')
    start=stamp(release['preparation_started_utc']);stop=stamp(release['preparation_stop_utc']);error=release.get('clock_error_bound_seconds')
    require(type(error) in (int,float) and math.isfinite(error) and 0<=error<=5
        and stamp(cohort['created_utc'])<=stamp(release['issued_utc'])<=now()
        and 0<(stop-start).total_seconds()<=900 and start<=now()<stop-timedelta(seconds=error)
        and stop+timedelta(seconds=error)<=LAST_EVALUATION_START,'One shared900second preparation window must precede final evaluation')
    pins=release.get('files_sha256',{});require(isinstance(pins,dict),'Pinned source/control inputs required')
    fixed={str(HERE/'prepare_sand_recovered_final_cohort_v1.py'):COHORT_ADAPTER_SHA,str(HERE/'download_sand_public_v2.py'):DOWNLOADER_SHA,
        str(HERE/'repackage_designsafe_sand.py'):CONVERTER_SHA,str(HERE/'evaluate_sand_graph_support_final.py'):EVALUATOR_SHA,
        str(HERE/'benchmark_sand_graph_support_rollout.py'):BENCH_SHA,str(HERE/'sand_graph_support_100k_protocol_v1.md'):PROTOCOL_SHA,
        str(HERE/'sand_scoped_operational_amendment_v1.md'):AMENDMENT_SHA,str(HERE/'sand_scoped_schedule_fixed_spec_v2.json'):PLAN_SHA,
        str(args.cohort):hashes['cohort'],str(args.cohort_audit):hashes['cohort_audit'],str(Path(__file__).resolve()):sha(__file__)}
    require(all(pins.get(p)==d for p,d in fixed.items()),'Complete frozen source/cohort closure required')
    # Root may now bind/read acquired test inputs; all six endpoints were checked first.
    require(all(Path(p).is_absolute() and sha(p)==d for p,d in pins.items()),'Root-bound source/input bytes differ')
    require(not args.output_dir.exists(),'Fresh separate preparation output required')
    for p in pins:require(args.output_dir!=Path(p) and args.output_dir not in Path(p).parents,'Output must not contain protected inputs')
    args._bindings={**pins,str(args.root_release):hashes['root_release']};args._stop=stop-timedelta(seconds=error)
    args._cohort_sha=hashes['cohort'];args._release_sha=hashes['root_release'];return release


def stable(args):
    require(now()<args._stop and all(sha(p)==d for p,d in args._bindings.items()) and now()<args._stop,
        'Preparation window or immutable binding changed')


def publish(args,path,value,extra_bindings=None):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    stable(args)
    require(all(sha(p)==d for p,d in (extra_bindings or {}).items()) and now()<args._stop,
        'Numeric/output binding or preparation deadline changed before ready publication')
    with Path(path).open('xb') as f:f.write(raw)


def bound(args,path):
    require(path is not None and str(path) in args._bindings,'Explicit root-bound preparation input required')
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==args._bindings[str(path)],'Explicit root-bound preparation input required')
    return json.loads(raw)


def acquisition(args):
    D=load('download_sand_public_v2.py',DOWNLOADER_SHA,'_sand_reserved_downloader')
    seconds=int((args._stop-now()).total_seconds());require(seconds>0,'Acquisition window exhausted')
    result=D.main(['--output',str(args.output_dir),'--files','test.npz','--max-seconds',str(seconds)])
    report_raw=(args.output_dir/'download_report.json').read_bytes();report=json.loads(report_raw)
    report_sha=hashlib.sha256(report_raw).hexdigest()
    require(result in (0,2) and report.get('status')=='complete_with_object_dtypes','Acquisition incomplete; preserve source and report')
    checked_acquisition(args.output_dir,report)
    outputs={str(args.output_dir/r['saved_name']):r['received_sha256'] for r in report['files']}
    outputs[str(args.output_dir/'download_report.json')]=report_sha
    publish(args,args.output_dir/'root_preparation_receipt.json',{'schema':SCHEMA,'mode':'acquire','cohort_sha256':args._cohort_sha,
        'root_release_sha256':args._release_sha,'download_report_sha256':report_sha,
        'publisher_test_sha256_available':False,'arrays_decoded':False,'test_evaluation_executed':False},outputs)


def checked_acquisition(root,report):
    require(report.get('schema')=='sand_public_acquisition_v2' and report.get('status')=='complete_with_object_dtypes'
        and report.get('dataset')=='Sand' and report.get('downloader_sha256')==DOWNLOADER_SHA
        and report.get('metadata_reference_sha256')==METADATA_SHA and report.get('requested_files')==['metadata.json','test.npz'],
        'Exact reviewed test-only acquisition receipt required')
    rows={r['name']:r for r in report['files']};require(set(rows)=={'metadata.json','test.npz'} and len(report['files'])==2,'Exact acquired file set required')
    for name,size in [('metadata.json',363),('test.npz',TEST_BYTES)]:
        r=rows[name];p=root/name
        require(r.get('status')=='downloaded_and_inspected' and r.get('stage')=='complete' and r.get('saved_name')==name
            and r.get('received_bytes')==r.get('expected_bytes')==p.stat().st_size==size and r.get('received_sha256')==sha(p),
            'Acquired test/metadata bytes differ')
    require(rows['metadata.json']['received_sha256']==METADATA_SHA and rows['test.npz']['zip']['member_count']==COUNTS['test'],
        'Exact metadata and all30ZIP members required')
    return rows['test.npz']


def convert(args):
    report=bound(args,args.acquisition_report)
    require(args.input_dir is not None and str(args.input_dir/'test.npz') in args._bindings
        and str(args.input_dir/'metadata.json') in args._bindings,'Root must bind acquired test and metadata bytes')
    row=checked_acquisition(args.input_dir,report);C=load('repackage_designsafe_sand.py',CONVERTER_SHA,'_sand_reserved_numeric')
    import numpy as np
    args.output_dir.mkdir(mode=0o700);metadata=bound(args,args.input_dir/'metadata.json')
    records,details=C.repackage_split(np,args.input_dir/'test.npz','test',row,args.output_dir/'.test.staging',metadata,set(),COUNTS['test'])
    require(details['frame_lengths']==[320] and details['position_dtypes']==['<f4'] and details['particle_type_ids']==[6]
        and details['kinematic_type3_particles']==0,'Reserved test differs from fixed Sand dimensions/types/precision')
    manifest={'format':'gns-trajectory-manifest','version':1,'split':'test','dataset':'Sand','metadata':metadata,'metadata_sha256':METADATA_SHA,
        'record_count':len(records),'records':records,'converter_sha256':CONVERTER_SHA,
        'source':{'family':'designsafe_published_npz','dataset':'Sand','file':'test.npz','size_bytes':TEST_BYTES,'sha256':row['received_sha256'],
            'ZIP_CRC_verified':True,'member_count':len(records),'acquisition_report_sha256':sha(args.acquisition_report)}}
    manifest_raw=(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    details['manifest_sha256']=hashlib.sha256(manifest_raw).hexdigest()
    structural={'schema':'designsafe_sand_numeric_repackage_v1','status':'complete_structural_only','dataset':'Sand',
        'source_family':'designsafe_published_npz','metadata_sha256':METADATA_SHA,'converter_sha256':CONVERTER_SHA,
        'acquisition_report_sha256':sha(args.acquisition_report),'splits':{'test':details},
        'test_preparation_adapter_sha256':sha(__file__),'cohort_sha256':args._cohort_sha,'root_release_sha256':args._release_sha,
        'duplicate_trajectories_within_and_across_selected_splits':False,'duplicate_scope':'test only; full cross-split census still required'}
    (args.output_dir/'.test.staging').replace(args.output_dir/'test')
    (args.output_dir/'metadata.json').write_bytes((args.input_dir/'metadata.json').read_bytes())
    structural_raw=(json.dumps(structural,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with (args.output_dir/'structural_report.json').open('xb') as f:f.write(structural_raw)
    outputs={str(args.output_dir/r[k]['path']):r[k]['sha256'] for r in records for k in ('positions','particle_types')}
    outputs.update({str(args.output_dir/'metadata.json'):args._bindings[str(args.input_dir/'metadata.json')],
        str(args.output_dir/'structural_report.json'):hashlib.sha256(structural_raw).hexdigest()})
    publish(args,args.output_dir/'test.json',manifest,outputs)


def numeric_identity(np,C,root,record):
    arrays=[];files={}
    for key in ('positions','particle_types'):
        d=record[key];relative=Path(d['path']);require(not relative.is_absolute() and '..' not in relative.parts,'Relative array path required')
        p=root/relative;require(p.resolve().is_relative_to(root.resolve()) and not p.is_symlink() and p.stat().st_size==d['size_bytes'] and sha(p)==d['sha256'],
            'Numeric array size/bytes/path differs')
        value=np.load(p,allow_pickle=False,mmap_mode='r');require(list(value.shape)==d['shape'] and value.dtype.str==d['dtype'],'Numeric shape/dtype differs')
        arrays.append(value);files[str(p)]=d['sha256']
    p,t=arrays;require(p.ndim==3 and p.shape[0]==320 and p.shape[1]>0 and p.shape[2]==2 and p.dtype.str=='<f4' and bool(np.isfinite(p).all())
        and t.dtype.str=='<i8' and t.shape in ((),(p.shape[1],)) and bool((t==6).all()),'Complete finite T320/D2/type6 source required')
    logical=hashlib.sha256((C.value_hash(p)+':'+C.value_hash(t)).encode()).hexdigest()
    require(logical==record['logical_content_sha256'],'Decoded logical content hash differs')
    return logical,files


def census(args):
    C=load('repackage_designsafe_sand.py',CONVERTER_SHA,'_sand_reserved_census')
    import numpy as np
    paths={'train':args.train_manifest,'valid':args.valid_manifest,'test':args.test_manifest};seen={};files={};rows=[]
    require(args.train_manifest is not None and args._bindings.get(str(args.train_manifest))==TRAIN_MANIFEST_SHA,'Original full training manifest required')
    for split,path in paths.items():
        manifest=bound(args,path);require(manifest.get('dataset')=='Sand' and manifest.get('split')==split
            and manifest.get('record_count')==len(manifest['records'])==COUNTS[split] and manifest.get('metadata_sha256')==METADATA_SHA,
            'Complete original split/count/metadata required')
        for index,record in enumerate(manifest['records']):
            require(record.get('source_index')==index and record.get('id')==f'{split}:{index:06d}','Exact source ordering required')
            logical,bound_files=numeric_identity(np,C,path.parent,record);require(logical not in seen,'Duplicate numeric trajectory within/across full splits')
            seen[logical]=[split,index];files.update(bound_files);rows.append({'split':split,'source_index':index,'logical_content_sha256':logical})
    args.output_dir.mkdir(mode=0o700)
    publish(args,args.output_dir/'all_split_census.json',{'schema':'adaptgns_sand_complete_numeric_census_v1','status':'complete_no_duplicates',
        'cohort_sha256':args._cohort_sha,'preparation_source_sha256':sha(__file__),'root_release_sha256':args._release_sha,
        'counts':COUNTS,'manifests_sha256':{s:sha(p) for s,p in paths.items()},'numeric_files_sha256':files,'all_split_numeric_duplicates_checked':True,
        'definition':'Exact dtype/shape/C-order bytes of both position/type arrays','records':rows,'evaluation_executed':False},files)


def split_inputs(args):
    require(args.split in ('valid','test'),'Explicit final split required')
    census_report=bound(args,args.census_report);manifest_path=args.valid_manifest if args.split=='valid' else args.test_manifest
    paths={'train':args.train_manifest,'valid':args.valid_manifest,'test':args.test_manifest};manifests={};numeric={}
    require(args._bindings.get(str(args.train_manifest))==TRAIN_MANIFEST_SHA,'Original full training manifest required')
    for split,path in paths.items():
        m=bound(args,path);manifests[split]=m
        require(m.get('dataset')=='Sand' and m.get('split')==split and m.get('record_count')==len(m.get('records',[]))==COUNTS[split]
            and m.get('metadata_sha256')==METADATA_SHA,'Complete census source manifests required')
        for index,record in enumerate(m['records']):
            require(record.get('source_index')==index and record.get('id')==f'{split}:{index:06d}','Exact census source ordering required')
            for key in ('positions','particle_types'):
                desc=record[key];relative=Path(desc['path']);p=path.parent/relative;digest=desc.get('sha256')
                require(not relative.is_absolute() and '..' not in relative.parts and p.resolve().is_relative_to(path.parent.resolve())
                    and not p.is_symlink() and str(p) not in numeric and isinstance(digest,str) and len(digest)==64
                    and all(c in '0123456789abcdef' for c in digest),'Exact safe distinct census numeric paths/hashes required')
                numeric[str(p)]=digest
    require(census_report.get('schema')=='adaptgns_sand_complete_numeric_census_v1' and census_report.get('status')=='complete_no_duplicates'
        and census_report.get('cohort_sha256')==args._cohort_sha and census_report.get('counts')==COUNTS
        and census_report.get('all_split_numeric_duplicates_checked') is True and census_report.get('preparation_source_sha256')==sha(__file__)
        and census_report.get('manifests_sha256')=={s:args._bindings[str(p)] for s,p in paths.items()}, 'Full root-bound numeric census required')
    require(census_report.get('numeric_files_sha256')==numeric,'Complete exact numeric census file bindings required')
    args._numeric_bindings=numeric
    manifest=manifests[args.split];structural=bound(args,args.structural_report)
    require(args._bindings.get(str(args.protocol))==PROTOCOL_SHA and args._bindings.get(str(args.train_admission))==TRAIN_ADMISSION_SHA,
        'Original protocol/training admission required')
    return manifest,structural,manifest_path


def candidate(args):
    manifest,structural,manifest_path=split_inputs(args)
    E=load('evaluate_sand_graph_support_final.py',EVALUATOR_SHA,'_sand_reserved_candidate');B=load('benchmark_sand_graph_support_rollout.py',BENCH_SHA,'_sand_reserved_candidate_contract')
    admission={'schema':E.ADMISSION_SCHEMA,'status':'admitted_for_final_evaluation','dataset':'Sand','split':args.split,
        'cohort_manifest_sha256':args._cohort_sha,'training_admission_sha256':TRAIN_ADMISSION_SHA,'protocol_sha256':PROTOCOL_SHA,
        'manifest_sha256':sha(manifest_path),'structural_report_sha256':sha(args.structural_report),
        'frames_per_trajectory':320,'record_count':30,'particle_type_ids':[6],'position_dtype':'<f4',
        'source_sha256':manifest['source']['sha256'],'metadata_sha256':METADATA_SHA,'converter_sha256':manifest['converter_sha256']}
    ns=argparse.Namespace(split=args.split,protocol=args.protocol,train_admission=args.train_admission,manifest=manifest_path,structural_report=args.structural_report)
    E.check_split(B,manifest,admission,structural,ns,args._cohort_sha)
    admission.update(status='candidate_requires_root_review',issued_by='preparation_wrapper_not_root',census_sha256=sha(args.census_report),
        preparation_source_sha256=sha(__file__),root_preparation_release_sha256=args._release_sha)
    args.output_dir.mkdir(mode=0o700);publish(args,args.output_dir/'split_admission_candidate.json',admission,args._numeric_bindings)


def preflight(args):
    manifest,structural,manifest_path=split_inputs(args);admission=bound(args,args.admission)
    require(admission.get('issued_by')=='root' and admission.get('census_sha256')==args._bindings[str(args.census_report)],
        'Separate root admission must bind the complete numeric census')
    E=load('evaluate_sand_graph_support_final.py',EVALUATOR_SHA,'_sand_reserved_preflight');B=load('benchmark_sand_graph_support_rollout.py',BENCH_SHA,'_sand_reserved_contract')
    ns=argparse.Namespace(split=args.split,protocol=args.protocol,train_admission=args.train_admission,manifest=manifest_path,structural_report=args.structural_report)
    E.check_split(B,manifest,admission,structural,ns,args._cohort_sha)
    args.output_dir.mkdir(mode=0o700)
    publish(args,args.output_dir/'split_preflight.json',{'schema':'adaptgns_sand_final_split_preflight_v1','status':'complete_split_contract_passed','split':args.split,
        'cohort_sha256':args._cohort_sha,'manifest_sha256':sha(manifest_path),'admission_sha256':sha(args.admission),
        'structural_report_sha256':sha(args.structural_report),'census_sha256':sha(args.census_report),'all_split_numeric_duplicates_checked':True,
        'evaluator_sha256':EVALUATOR_SHA,'preparation_source_sha256':sha(__file__),'root_release_sha256':args._release_sha,
        'record_count':30,'frames':320,'horizon':314,'evaluation_executed':False},args._numeric_bindings)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--execute',action='store_true')
    p.add_argument('--mode',choices=['acquire','convert','census','candidate','preflight']);p.add_argument('--split',choices=['valid','test'])
    for n in ('cohort','cohort-audit','root-release','output-dir','input-dir','acquisition-report','train-manifest','valid-manifest',
        'test-manifest','census-report','structural-report','admission','protocol','train-admission'):p.add_argument('--'+n,type=Path)
    args=p.parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema':SCHEMA,'status':'description_only','modes':['acquire','convert','census','candidate','preflight'],
            'complete_cohort_before_test':True,'all_modes_share_preparation_window_seconds':900,'training_or_evaluation':False},indent=2));return 0
    require(args.mode and all(getattr(args,k) is not None for k in ('cohort','cohort_audit','root_release','output_dir')),'Explicit mode/cohort/root/new output required')
    for k,v in vars(args).copy().items():
        if isinstance(v,Path):setattr(args,k,v.resolve())
    gate(args)
    def expired(sig,frame):raise TimeoutError('Shared Sand source-preparation window exhausted')
    previous=signal.getsignal(signal.SIGALRM);signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,max(.001,(args._stop-now()).total_seconds()))
    try:globals()[{'acquire':'acquisition'}.get(args.mode,args.mode)](args)
    except BaseException as error:
        if args.output_dir.exists():write(args.output_dir/'failed_preparation.json',{'schema':SCHEMA,'mode':args.mode,'error_type':type(error).__name__,
            'error':str(error),'all_sources_staging_and_failures_retained':True,'root_release_sha256':args._release_sha})
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    return 0


if __name__=='__main__':raise SystemExit(main())
