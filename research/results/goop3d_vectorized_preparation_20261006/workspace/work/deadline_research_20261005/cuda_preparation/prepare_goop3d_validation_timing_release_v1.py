#!/usr/bin/env python3
"""Write exact six-call timing contracts from one explicit root approval spec.

Scalar/byte preparation only: never launches a subprocess, reads numeric arrays,
deserializes a checkpoint or selects a scientific endpoint. Contracts remain a
separate inspectable artifact before root invokes the timing supervisor.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCHEMA = 'adaptgns_goop3d_timing_contract_preparation_v1'
SUPERVISOR_SHA = '104e7887a2e9fd9760e2cdacbcab6e850856120a9b6e941a8908b764ccccbb29'
# The generator independently checks all scientific identity fields, including
# the received source hash, before deriving the admitted source-only copy.
VALID_SOURCE_SHA = '5242810fa2057fbcf8e2d4da203823855f6b71e77dcfa848a618fb6871f6778b'
PINNED_SIBLINGS = {
    'goop3d_native_evaluation_v1.py':'a742123093aff433f3a4e302929a52bae4f5fee9610195d86df726852575b1d5',
    'goop3d_diagnostic_metrics_v1.py':'5ef3de96e470eea495dd56c1f60396bbd166b9ea5c5bc340864715cb9e2c173b',
    'goop3d_graph_support_vectorized_v1.py':'ca5898ec3f487d8138adf7479743156874a7796df09841f8ff7711b97536ad50',
    'audit_goop3d_auxiliary.py':'1b5a8c24fde395b2633ee197557fde3e5bf30ca74908a77f1fb0e011b85e41fb',
}


def require(value, message):
    if not value: raise ValueError(message)


def supervisor():
    path = HERE/'run_goop3d_validation_timing_v1.py'
    require(hashlib.sha256(path.read_bytes()).hexdigest() == SUPERVISOR_SHA, 'Reviewed supervisor differs')
    spec = importlib.util.spec_from_file_location('_goop3d_timing_contract_supervisor', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prepare(spec_path, directory, S):
    root = S.read(spec_path)
    require(root.get('schema') == 'adaptgns_goop3d_timing_root_preparation_spec_v1'
            and root.get('status') == 'approved_for_contract_generation_and_six_bounded_validation_calls'
            and root.get('issued_by') == 'root' and root.get('scientific_training_admitted') is False
            and root.get('scientific_endpoint_selected') is False and root.get('test_access_allowed') is False
            and root.get('root_approved_valid_source_admission') is True
            and root.get('reviewed_generator_sha256') == S.sha(__file__)
            and root.get('reviewed_supervisor_sha256') == SUPERVISOR_SHA,
            'Explicit root approval for exact source-only contract derivation required')
    E,W,B,T = S.modules()
    require(all(S.sha(HERE/name)==value for name,value in PINNED_SIBLINGS.items()), 'Reviewed numerical/auxiliary sibling source differs')
    common = {k:str(Path(v).resolve()) for k,v in root['common'].items()}
    require(set(common) == set(S.COMMON)-{'split_admission'}, 'Exact common input spec required')
    capacity_root = Path(root['capacity_root']).resolve(); summary_path = capacity_root/'summary.json'
    _,_,pointers = W.capacity_inputs(S.read(summary_path))
    template_path = HERE/'goop3d_valid_timing_split_admission.template.json'
    require(S.sha(template_path) == root.get('reviewed_valid_admission_template_sha256'), 'Reviewed valid template differs')
    admission = S.read(template_path)
    expected = {'manifest_sha256':E.VALID_MANIFEST_SHA, 'metadata_sha256':E.METADATA_SHA,
        'source_sha256':VALID_SOURCE_SHA, 'context_semantics_sha256':E.CONTEXT_SHA,
        'converter_sha256':E.CONVERTER_SHA, 'split':'valid', 'dataset':'Goop-3D', 'frames':301,
        'dimension':3, 'record_count':100, 'particle_type_ids':[7]}
    require(admission.get('schema') == 'adaptgns_goop3d_evaluation_split_admission_v1'
            and admission.get('status') == 'not_admitted' and all(admission.get(k)==v for k,v in expected.items()),
            'Exact original100-record valid-only source template required')
    for field in ('manifest','structural_report','acquisition_report','context_semantics','auxiliary_report'):
        require(S.sha(common[field]) == admission[field+'_sha256'], 'Actual valid evidence differs: '+field)
    manifest = S.read(common['manifest'])
    require(manifest.get('record_count') == 100 and manifest.get('split') == 'valid'
            and manifest.get('source',{}).get('sha256') == VALID_SOURCE_SHA, 'Original complete valid scalar manifest required')
    directory = Path(directory).resolve()
    protected = [HERE, Path(common['manifest']).parent, capacity_root, Path(root['output_dir']).resolve()]
    require(all(directory != p and p not in directory.parents and directory not in p.parents for p in protected),
            'Contracts directory must be separate from data, capacity, sources and future output')
    directory.mkdir(parents=True, exist_ok=False)
    source_bindings = {str(Path(spec_path).resolve()):S.sha(spec_path), str(Path(__file__).resolve()):S.sha(__file__),
        str(template_path):S.sha(template_path)}
    try:
        admission.update(status='admitted',issued_by='root',scope='bounded_capacity_validation_only',scientific_training_admitted=False,
                         root_preparation_spec_sha256=S.sha(spec_path))
        admission_path = directory/'valid_split_admission.json'; S.write(admission_path,admission)
        common['split_admission'] = str(admission_path)
        checkpoints = {}
        for arm in ('base','mix'):
            path = capacity_root/'jobs'/f'{arm}_seed0'/'checkpoint-000000512.pt'
            require(S.sha(path) == pointers[arm,0], 'Verified capacity endpoint bytes differ')
            checkpoints[arm] = {'path':str(path),'sha256':pointers[arm,0]}
        parent = {'schema':S.RELEASE_SCHEMA,'status':'admitted_for_six_validation_timing','issued_by':'root',
            'scientific_training_admitted':False,'scientific_endpoint_selected':False,'test_access_allowed':False,
            'supervisor_sha256':SUPERVISOR_SHA,'schedule':S.SCHEDULE,'environment':S.ENVIRONMENT,
            'output_dir':str(Path(root['output_dir']).resolve()),'hostname':root['hostname'],
            'process_identity_checked_utc':root['process_identity_checked_utc'],'gpu_uuids':root['gpu_uuids'],
            'evaluation_overlap_credit':False,'cleanup_seconds':15.,'whole_supervisor_outer_timeout_required':True,
            'inner_seconds':root['inner_seconds'],'outer_seconds':root['outer_seconds'],
            'python':str(Path(root['python']).absolute()),'capacity_summary':str(summary_path),
            'capacity_stop_receipt':str(Path(root['capacity_stop_receipt']).resolve()),'common':common,
            'checkpoints':checkpoints,'evaluator_releases':{},'root_preparation_spec_sha256':S.sha(spec_path)}
        files = {str(Path(v).resolve()):S.sha(v) for k,v in common.items() if k!='repo'}
        for name in ('evaluate_goop3d_graph_support_v1.py','goop3d_native_evaluation_v1.py',
            'goop3d_diagnostic_metrics_v1.py','goop3d_graph_support_vectorized_v1.py','goop3d_deadline_worksheet_v1.py',
            'measure_sand_cuda_capacity_v2.py','audit_goop3d_auxiliary.py'):
            files[str(HERE/name)] = S.sha(HERE/name)
        for job in S.SCHEDULE:
            selected = checkpoints[job['arm']]
            release = {'schema':'adaptgns_goop3d_evaluation_release_v1','status':'admitted_for_execution','issued_by':'root',
                'purpose':'capacity_timing','mode':job['mode'],'split':'valid','arm':job['arm'],'seed':0,
                'checkpoint_sha256':selected['sha256'],'checkpoint_updates':512,'scientific_training_admitted':False,
                'evaluator_sha256':S.EVALUATOR_SHA,'graph_sha256':E.GRAPH_SHA,'cuda_index':job['gpu'],
                'gpu_uuid':root['gpu_uuids'][job['gpu']],'max_seconds':root['inner_seconds'][job['mode']],
                'whole_invocation_outer_timeout_required':True,'numerical_source_sha256':E.NUMERICAL_SOURCE_PINS,
                'files_sha256':{**files,selected['path']:selected['sha256']},'root_preparation_spec_sha256':S.sha(spec_path)}
            path=directory/(job['id']+'.release.json');S.write(path,release);parent['evaluator_releases'][job['id']]=str(path)
        parent_files = {**files, **source_bindings}
        for path in [parent['python'],parent['capacity_summary'],parent['capacity_stop_receipt'],*parent['evaluator_releases'].values(),
                     *(v['path'] for v in checkpoints.values())]:
            parent_files[str(Path(path).resolve())]=S.sha(path)
        parent['files_sha256']=parent_files
        release_path=directory/'supervisor_release.json'
        # Preflight the in-memory parent contract against the already bound root
        # spec. Publish its active parent release only after every gate passes.
        args=SimpleNamespace(release=Path(spec_path).resolve(),output_dir=Path(parent['output_dir']))
        # Exact reviewed preflight executes scalar/source/byte checks only. It
        # does not inspect /proc, invoke nvidia-smi or create timing output.
        children,bindings=S.preflight(args,parent,E,W,B,T)
        require(all(S.sha(p)==v for p,v in source_bindings.items()),'Preparation source/spec/template changed')
        S.write(release_path,parent)
        bindings=E.merge_bindings(bindings,{release_path:S.sha(release_path)})
        report={'schema':SCHEMA,'status':'six_exact_contracts_prepared_no_process_launched',
            'root_spec_sha256':S.sha(spec_path),'source_sha256':S.sha(__file__),'supervisor_release_sha256':S.sha(release_path),
            'commands':{job['id']:argv for job,cargs,argv in children},'all_preflight_bindings':bindings,
            'scientific_training_admitted':False,'scientific_endpoint_selected':False,'test_accessed':False}
        S.write(directory/'preparation_report.json',report)
        return report
    except BaseException as error:
        S.write(directory/'failed_preparation.json',{'schema':SCHEMA,'status':'failed_contract_preparation',
            'error_type':type(error).__name__,'error':str(error),'all_outputs_retained':True,'processes_launched':0})
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--write-contracts',action='store_true');parser.add_argument('--spec',type=Path);parser.add_argument('--contracts-dir',type=Path)
    args=parser.parse_args(argv)
    if not args.write_contracts:
        print(json.dumps({'schema':SCHEMA,'status':'description_only','processes_launched':0,'test_accessed':False}));return 0
    require(args.spec and args.contracts_dir,'Explicit root spec and fresh contracts directory required')
    prepare(args.spec,args.contracts_dir,supervisor());return 0


if __name__=='__main__':raise SystemExit(main())
