#!/usr/bin/env python3
"""Sand final allocation using unchanged reviewed scoped lifecycle/coverage.

Description only by default. Fresh root-admitted six100k cohort and both split
preflights are mandatory. No acquisition, training, endpoint choice or retry.
"""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SCHEMA='adaptgns_sand_evaluation_gpu_scoped_v1'
RELEASE_SCHEMA='adaptgns_sand_evaluation_gpu_scoped_release_v1'
ENGINE_SHA='a13ca1bc30161f89cf79b6c0b632c65a272913302ef0cc37dcfd7ed762c6aa21'
TRAINING_SUPERVISOR_SHA='2970922593f41f74713aff9208e4c6964bb1b3f351fee906724bbf52f11d4d13'
EVALUATOR_SHA='952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58'
BENCH_SHA='8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13'
TRAINER_SHA='fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124'
PROTOCOL_SHA='e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d'
TRAIN_ADMISSION_SHA='fbc8ddd94d90adea6e8f4b07caa73931dfcea41e86c8f2f830e3f01f6a177b73'
PLAN_SHA='403a96c1304341ea1899c994e0ab7474280add44ea5986ec41570e08d16c337b'
AMENDMENT_SHA='411547c3355859906f59792b3b493841030b0afd2e105e17e466114df6743738'
DEADLINE=datetime(2026,10,7,4,tzinfo=timezone.utc)
ANALYSIS_RESERVE=3600
SCHEDULE={'A':[('base',1,0),('mix',1,1),('base',2,2),('mix',2,3)],'B':[('base',0,2),('mix',0,3)]}


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_bytes())
def load(name,pin,label):
    path=HERE/name;require(sha(path)==pin,'Reviewed source differs: '+name)
    spec=importlib.util.spec_from_file_location(label,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def modules():
    G=load('supervise_goop_evaluation_gpu_scoped_v3.py',ENGINE_SHA,'_sand_final_scoped_engine')
    # These are documented operational constants in this private module only.
    # Its source, lifecycle, coverage functions and numerical children are unchanged.
    G.SCHEMA=SCHEMA;G.DEADLINE=DEADLINE
    S=load('supervise_sand_scoped_science_v1.py',TRAINING_SUPERVISOR_SHA,'_sand_final_environment')
    E=load('evaluate_sand_graph_support_final.py',EVALUATOR_SHA,'_sand_final_gate')
    B=load('benchmark_sand_graph_support_rollout.py',BENCH_SHA,'_sand_final_split_contract')
    return G,S,E,B


def validate_release(args,release,mods,now=None):
    G,S,E,B=mods;clock=(lambda:datetime.now(timezone.utc)) if now is None else lambda:now
    now=clock()
    require(release.get('schema')==RELEASE_SCHEMA and release.get('status')=='admitted_for_execution_allocation'
        and release.get('issued_by')=='root' and release.get('dataset')=='Sand' and release.get('host_role') in SCHEDULE
        and release.get('host')==socket.gethostname() and release.get('environment')==G.ENVIRONMENT
        and release.get('cost_basis')==G.COST_BASIS and release.get('all_six100k_models_frozen_before_test') is True,
        'Exact root Sand final-allocation release required')
    require(G.utc(release['compute_analysis_deadline_utc'])==DEADLINE and release.get('outer_processing_reserve_seconds')==ANALYSIS_RESERVE
        and release.get('stage_quotas_seconds')=={n:q for n,_,_,q in G.STAGES}
        and release.get('cleanup_seconds_per_invocation')==G.CLEANUP
        and release.get('preparation_and_cohort_reserves_already_accounted_before_this_queue') is True,
        'Four fixed stage allocations plus3600analysis reserve; do not double-count prior preparation')
    error=release.get('clock_error_bound_seconds');latest=G.utc(release['latest_start_utc'])
    require(G.finite(error) and error<=5 and -5<=(now-G.utc(release['process_clock_checked_utc'])).total_seconds()<=300
        and now+timedelta(seconds=error)<=latest<=DEADLINE-timedelta(seconds=G.STREAM_SECONDS+ANALYSIS_RESERVE+error),
        'Fresh clock and complete final stream allocation no longer fit')
    pins=release.get('files_sha256',{});require(isinstance(pins,dict),'Complete file map required')
    fixed={str(Path(__file__).resolve()):sha(__file__),str(HERE/'supervise_goop_evaluation_gpu_scoped_v3.py'):ENGINE_SHA,
        str(HERE/'supervise_sand_scoped_science_v1.py'):TRAINING_SUPERVISOR_SHA,str(HERE/'evaluate_sand_graph_support_final.py'):EVALUATOR_SHA,
        str(HERE/'benchmark_sand_graph_support_rollout.py'):BENCH_SHA,str(HERE/'train_sand_graph_support_cuda.py'):TRAINER_SHA,
        str(HERE/'sand_graph_support_100k_protocol_v1.md'):PROTOCOL_SHA,str(HERE/'sand_scoped_schedule_fixed_spec_v2.json'):PLAN_SHA,
        str(HERE/'sand_scoped_operational_amendment_v1.md'):AMENDMENT_SHA,str(args.lifecycle_source.resolve()):G.LIFECYCLE_SHA}
    require(all(pins.get(p)==v for p,v in fixed.items()),'Frozen operational/numerical source closure required')
    interpreter=S.python_environment(sys.executable);require(release.get('python_environment')==interpreter,'Exact lexical Python/venv receipt required')
    pyfiles={interpreter['lexical_path']:interpreter['binary_sha256'],interpreter['resolved_binary_path']:interpreter['binary_sha256']}
    if interpreter['pyvenv_config_path']:pyfiles[interpreter['pyvenv_config_path']]=interpreter['pyvenv_config_sha256']
    require(all(pins.get(p)==d for p,d in pyfiles.items()),'Interpreter target/config must be pinned')
    amendment=release.get('operational_amendment')
    require(amendment=={'path':str(HERE/'sand_scoped_operational_amendment_v1.md'),'sha256':AMENDMENT_SHA},'Exact amended mapping/window required')
    helper=G.load_lifecycle(args.lifecycle_source);uuids=release.get('gpu_uuids');expected=SCHEDULE[release['host_role']]
    require(isinstance(uuids,list) and len(uuids)==4 and len({helper.canonical_gpu_uuid(u) for u in uuids})==4,'All four physical UUIDs required')
    require(release.get('gpu_scope')=={'owned_indices':sorted({g for _,_,g in expected}),'unassigned_devices':'observe_without_control',
        'timing_scope':'shared_host_operational_measurement','live_training_handoff':False},'Exact Sand scoped GPU ownership required')
    streams=release.get('streams',[])
    require([(s.get('arm'),s.get('seed'),s.get('gpu')) for s in streams]==expected,'Exact Sand model/host/GPU mapping required')
    common=None;split_files={}
    for stream in streams:
        require(stream.get('id')==f"{stream['arm']}_seed{stream['seed']}" and len(stream.get('commands',[]))==4,'Four stages for each model required')
        for command,(name,mode,split,quota) in zip(stream['commands'],G.STAGES):
            options=G.flags(command)
            needed={'--repo','--benchmark-helper','--benchmark-sha256','--cohort','--manifest','--admission','--structural-report',
                '--train-admission','--trainer-source','--protocol','--checkpoint','--checkpoint-sha256','--cohort-audit',
                '--output-dir','--mode','--split','--objective','--arm','--seed','--cuda-index','--threads','--max-seconds'}
            require(set(options)==needed and command[0]==interpreter['lexical_path']
                and command[1]==str(HERE/'evaluate_sand_graph_support_final.py') and options['--mode']==mode and options['--split']==split
                and options['--objective']=='faithful' and options['--arm']==stream['arm'] and options['--seed']==str(stream['seed'])
                and options['--cuda-index']==str(stream['gpu']) and options['--threads']=='2' and options['--max-seconds']==str(quota),
                'Exact frozen Sand evaluator/model/stage command required')
            file_options=needed-{'--repo','--benchmark-sha256','--checkpoint-sha256','--output-dir','--mode','--split','--objective','--arm','--seed','--cuda-index','--threads','--max-seconds'}
            require(Path(options['--repo']).is_absolute() and all(Path(options[k]).is_absolute() and options[k] in pins for k in file_options),
                'All original evaluator source/cohort/split inputs must be pinned')
            require(pins[options['--checkpoint']]==options['--checkpoint-sha256'] and pins[options['--benchmark-helper']]==options['--benchmark-sha256']==BENCH_SHA
                and pins[options['--trainer-source']]==TRAINER_SHA and pins[options['--protocol']]==PROTOCOL_SHA
                and pins[options['--train-admission']]==TRAIN_ADMISSION_SHA,'Checkpoint and frozen scientific inputs differ')
            require(Path(options['--output-dir'])==args.output_dir/'jobs'/stream['id']/name,'Fresh exact output tree required')
            shared={k:options[k] for k in ('--cohort','--cohort-audit','--protocol','--trainer-source','--train-admission','--benchmark-helper','--repo')}
            require(common in (None,shared),'One frozen complete cohort/source required');common=shared
            split_identity={k:options[k] for k in ('--manifest','--admission','--structural-report')}
            require(split_files.get(split,split_identity)==split_identity,'One full source per split required');split_files[split]=split_identity
            # Complete cohort gate first. No split manifest, preflight or test arrays read yet.
            ns=argparse.Namespace(**{k[2:].replace('-','_'):Path(v) if k in file_options else v for k,v in options.items()})
            ns.seed=int(ns.seed);ns.arm=stream['arm']
            for k in ('--cohort','--cohort-audit','--protocol','--trainer-source','--train-admission','--benchmark-helper','--checkpoint'):
                require(sha(options[k])==pins[options[k]],'Frozen cohort/source/model bytes changed')
            E.check_cohort(read(ns.cohort),ns,read(ns.train_admission))
            require(all(pins.get(str(Path(options['--repo'])/p))==d for p,d in B.SOURCE_PINS.items()),'Numerical evaluator closure required')
    preflights=release.get('split_preflight',{});require(set(preflights)=={'valid','test'},'Both complete split preflights required')
    for split,files in split_files.items():
        receipt=preflights[split];require(set(receipt)=={'path','sha256'} and pins.get(receipt['path'])==receipt['sha256']==sha(receipt['path']),
            'Root-bound split preflight bytes required')
        p=read(receipt['path'])
        require(p.get('schema')=='adaptgns_sand_final_split_preflight_v1' and p.get('status')=='complete_split_contract_passed'
            and p.get('split')==split and p.get('cohort_sha256')==pins[common['--cohort']]
            and p.get('manifest_sha256')==pins[files['--manifest']] and p.get('admission_sha256')==pins[files['--admission']]
            and p.get('structural_report_sha256')==pins[files['--structural-report']] and p.get('all_split_numeric_duplicates_checked') is True
            and p.get('evaluator_sha256')==EVALUATOR_SHA and p.get('evaluation_executed') is False,'Complete checksum/conversion/census preflight required')
        ns=argparse.Namespace(split=split,cohort=Path(common['--cohort']),protocol=Path(common['--protocol']),train_admission=Path(common['--train-admission']),
            manifest=Path(files['--manifest']),structural_report=Path(files['--structural-report']))
        E.check_split(B,read(ns.manifest),read(files['--admission']),read(ns.structural_report),ns,pins[common['--cohort']])
    G.recheck_files(release)
    require(clock()+timedelta(seconds=error)<=latest,'Setup consumed launch window')
    return helper


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--execute',action='store_true')
    for n in ('release','lifecycle-source','output-dir'):p.add_argument('--'+n,type=Path)
    a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps({'schema':SCHEMA,'status':'description_only','schedule':SCHEDULE,'required_rollouts':1080,'horizon':314,
            'analysis_reserve_seconds':ANALYSIS_RESERVE,'compute_analysis_deadline_utc':DEADLINE.isoformat(),'test_acquisition_or_training':False},indent=2));return 0
    require(all(getattr(a,k) is not None for k in ('release','lifecycle_source','output_dir')) and sys.platform.startswith('linux')
        and hasattr(os,'wait4') and os.environ.get('CUDA_VISIBLE_DEVICES') is None,'Explicit unremapped Linux inputs required')
    a.release=a.release.resolve();a.lifecycle_source=a.lifecycle_source.resolve();a.output_dir=a.output_dir.resolve()
    require(not a.output_dir.exists(),'Fresh queue output required')
    raw=a.release.read_bytes();release=json.loads(raw);mods=modules();helper=validate_release(a,release,mods)
    require(a.release.read_bytes()==raw,'Root release changed before launch')
    result=mods[0].execute_queue(a,release,helper)
    require(a.release.read_bytes()==raw,'Root release changed; retain queue and review outputs')
    return result


if __name__=='__main__':raise SystemExit(main())
