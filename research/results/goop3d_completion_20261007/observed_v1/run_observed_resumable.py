"""Resumable saved-row audit; unchanged scientific functions, no elapsed cutoff."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
import fcntl
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
DEFAULT_FROZEN = '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/goop3d_observed_history_analysis_v1'
PRIOR_FAILED_PHASE = 'fb5e6c098f83ebf344ce00a9bb7e31a8b135788d9850ad6d713a9283a83358f2'
SCIENCE = None


def need(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def atomic_write(path, value):
    """Data fsync, atomic rename, directory fsync; old committed files survive crashes."""
    path = Path(path)
    need(not path.is_symlink(), 'refuse symlink output')
    raw = encode(value)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name+'.pending.', dir=path.parent)
    temporary = Path(temporary_name)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return digest(raw)


class Checkpoints:
    def __init__(self, output, identity):
        self.output = Path(output)
        self.rows = self.output / 'rows'
        self.rows.mkdir(exist_ok=True)
        self.path = self.output / 'checkpoint_index.json'
        self.identity = identity
        if self.path.exists():
            need(not self.path.is_symlink(), 'refuse symlink index')
            self.index = strict(self.path.read_bytes())
            need(self.index['identity'] == identity, 'checkpoint source/input/runtime identity differs')
            need(self.index['schema'] == 'goop3d_observed_row_checkpoints_v1', 'checkpoint schema')
        else:
            self.index = {'schema': 'goop3d_observed_row_checkpoints_v1', 'identity': identity,
                          'completed': {}, 'failed': {}}
            atomic_write(self.path, self.index)

    def read(self, key, task_sha256):
        entry = self.index['completed'].get(key)
        if entry is None:
            return None
        need(entry['task_sha256'] == task_sha256, 'checkpoint cell/source pin differs')
        path = self.rows / (key + '.json')
        need(not path.is_symlink() and path.stat().st_size <= 16 << 20, 'bounded regular checkpoint')
        raw = path.read_bytes()
        need(digest(raw) == entry['sha256'], 'checkpoint bytes differ from committed index')
        record = strict(raw)
        need(record['key'] == key and record['task_sha256'] == task_sha256 and
             record['identity_sha256'] == digest(encode(self.identity)), 'checkpoint identity differs')
        return record['result']

    def commit(self, key, task_sha256, result):
        need(key not in self.index['completed'], 'checkpoint already committed')
        record = {'key': key, 'task_sha256': task_sha256,
                  'identity_sha256': digest(encode(self.identity)), 'result': result}
        row_path = self.rows / (key + '.json')
        if row_path.exists():
            need(not row_path.is_symlink() and row_path.read_bytes()==encode(record),
                 'unindexed row conflicts with recomputed result; preserve for inspection')
            pin = file_hash(row_path)
        else:
            pin = atomic_write(row_path, record)
        candidate = {**self.index,
                     'completed': {**self.index['completed'], key: {'sha256': pin, 'task_sha256': task_sha256}},
                     'failed': {k:v for k,v in self.index['failed'].items() if k!=key}}
        atomic_write(self.path, candidate)
        self.index = candidate

    def failure(self, key, task_sha256, error, attempt):
        record = {'key': key, 'task_sha256': task_sha256, 'error_type': type(error).__name__,
                  'error': str(error), 'traceback': ''.join(traceback.format_exception(error)),
                  'scientific_admission': False}
        path = attempt / (key + '.failure.json')
        pin = atomic_write(path, record)
        candidate = {**self.index, 'failed': {**self.index['failed'], key: {
            'path': str(path.relative_to(self.output)), 'sha256': pin, 'task_sha256': task_sha256}}}
        atomic_write(self.path, candidate)
        self.index = candidate


def load_science(frozen, pins):
    frozen = Path(frozen).resolve()
    for name, pin in pins['frozen_files_sha256'].items():
        need(file_hash(frozen / name) == pin, 'frozen science source differs: ' + name)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path.insert(0, str(frozen))
    names = ('observed_common', 'audit_goop3d_observed_histories_v1',
             'goop3d_observed_history_arithmetic_v1', 'summarize_goop3d_observed_histories_v1',
             'check_goop3d_observed_history_summary_v1')
    modules = tuple(importlib.import_module(name) for name in names)
    for module in modules:
        need(Path(module.__file__).resolve().parent == frozen, 'unexpected science module path')
    return modules


def initialize_worker(frozen, pins):
    global SCIENCE
    SCIENCE = load_science(frozen, pins)


def audit_cell(task):
    """Frozen per-row checks in original order; only scheduling/storage is new."""
    common, audited, a, _, _ = SCIENCE
    np = audited.np
    checks = a.Checks()
    arm, seed, name, mode, split = (task[k] for k in ('arm', 'seed', 'stage', 'mode', 'split'))
    row, c, directory = task['row'], task['cell'], task['directory']
    ident = a.unit(c, mode)
    bounds = np.asarray(task['bounds'], dtype=np.float64)
    allowed = task['files']
    def bound(path):
        need(path in allowed, 'path outside exact row allowlist')
        entry = allowed[path]
        return common.read_bound(path, entry['sha256'], common.CAP_ARCHIVE if path.endswith('.npz') else common.CAP_JSON,
                                 expected_bytes=entry['bytes'])
    checks.context=f'{arm}/seed{seed}/{name}/{ident}'
    checks.equal(common.strict(bound(directory+'/'+c['row_file'])),row,'exact original observed row JSON')
    checks.equal((row['arm'],row['training_seed'],row['objective']),(arm,seed,'faithful'),'observed model identity')
    checks.equal(row['status'],'complete' if c['state']=='completed_required_outcome' else 'failed','observed cell status')
    checks.equal(row.get('failure'),c.get('failure'),'original observed failure')
    checks.equal(row['protocol_sha256'],task['protocol_sha256'],'row protocol pin')
    arrays=audited.numeric_archive(bound(directory+'/'+row['artifact_file']))
    source=task['source']
    checks.equal(arrays['particle_types'].shape,(source['particles'],),'original source particle count')
    metrics=a.diagnostic.audit_row(row,arrays,bounds,checks,mode=mode)
    if mode=='clean-validation' and row['status']=='complete':
        a.graph_scalar(row['graph'],'base',len(arrays['particle_types']),checks)
        a.graph_arrays(arrays['native_edges'],np.empty((0,2),dtype=np.int64),row['graph'],len(arrays['particle_types']),checks)
    if mode=='same-state':
        checks.equal(row['random_seed_material'],[20261006,93000,seed,0 if split=='valid' else 1,ident[0],ident[1]],'fixed policy RNG material')
        metrics.update(a.extra_diagnostic_metrics(row,arrays,bounds,checks))
    pairings=a.diagnostic_shared_hashes(arrays,split,ident,mode,checks)
    if mode=='same-state':
        for method in ('base','dense','random25','speed25','relative-velocity-RMS25','natural_base_reference'):
            for kind in ('edges','selected_optional_pairs'):
                key=f'r1_{method}__{kind}'
                if key in arrays:pairings[f'{split}/graph/{method}/{kind}/{ident}'+(f'/seed{seed}' if method=='random25' else '')]=a.array_hash(arrays[key])
    detail={'unit':list(ident),'status':row['status'],'failure':row.get('failure'),'metrics':metrics,'policy_completion':[p for p,v in row.get('policies',{}).items() if v.get('status')=='complete']}
    return {'detail': detail, 'pairings': pairings, 'checks': checks.count}


def prepare(common, audited, a, frozen, pins):
    """Original accounting, inventory and protocol checks, with no clock guard."""
    checks = a.Checks()
    raw = common.read_bound(common.COLLECTION, common.COLLECTION_SHA, common.CAP_COLLECTION,
                            expected_bytes=common.COLLECTION_BYTES)
    collection = common.strict(raw)
    del raw
    full_accounting, schedules, populations = audited.accounting(collection, checks)
    ref = collection['physical_reference']
    checks.equal((ref['dataset'],ref['frames'],ref['horizon'],ref['dimension'],ref['particle_type'],ref['metadata_sha256']),
                 ('Goop-3D',301,295,3,7,a.METADATA_SHA),'original physical reference')
    bounds = audited.np.asarray(ref['metadata']['bounds'], dtype=audited.np.float64)
    checks.require(bounds.shape==(3,2) and audited.np.isfinite(bounds).all() and audited.np.all(bounds[:,1]>bounds[:,0]),'valid saved bounds')
    bindings_path = Path(frozen) / 'worker_source_bindings.json'
    bindings = common.strict(common.read_bound(bindings_path, pins['frozen_files_sha256']['worker_source_bindings.json'], common.CAP_JSON))
    sources = bindings['files_sha256']
    need(type(sources) is dict and sources, 'source bindings required')
    for path, pin in sources.items():
        common.read_bound(path, pin, common.CAP_JSON, retain=False)
    files = collection['files_sha256']
    allowed, prepared, tasks = {}, [], []
    for stage in collection['stages']:
        if stage['stage'] not in common.OBSERVED:
            continue
        arm,seed,name=stage['arm'],stage['seed'],stage['stage'];mode,split=stage['mode'],stage['split']
        directory=common.QUEUE+'/jobs/'+arm+'_seed'+str(seed)+'/'+name
        rows={a.unit(r,mode):r for r in stage['rows']}
        checks.equal(len(rows),len(stage['rows']),'unique committed observed rows')
        committed={a.unit(c,mode):c for c in stage['cells'] if c['state'] in ('completed_required_outcome','recorded_failed_outcome')}
        checks.equal(set(rows),set(committed),'committed rows equal required observed cell states')
        protocol=directory+'/protocol.json'
        if rows:
            checks.require(protocol in files,'committed observed rows require protocol');allowed[protocol]=files[protocol]
        for ident,row in rows.items():
            c=committed[ident];basename=f'trajectory_{ident[0]:06d}_target_{ident[1]:03d}'
            checks.equal(c['row_file'],basename+'.json','exact observed row basename')
            checks.equal(row['artifact_file'],basename+'.npz','exact observed archive basename')
            checks.equal(c['row_sha256'],files[directory+'/'+basename+'.json']['sha256'],'row inventory pin')
            checks.equal(row['artifact_sha256'],c['artifact_sha256'],'row and cell archive pins')
            checks.equal(row['artifact_sha256'],files[directory+'/'+basename+'.npz']['sha256'],'archive inventory pin')
            for suffix in ('.json','.npz'):
                path=directory+'/'+basename+suffix;allowed[path]=files[path]
        if rows:
            protocol_row=common.strict(common.read_bound(protocol,files[protocol]['sha256'],common.CAP_JSON,expected_bytes=files[protocol]['bytes']))
            checks.equal((protocol_row['schema'],protocol_row['purpose'],protocol_row['mode'],protocol_row['split'],protocol_row['arm'],protocol_row['seed'],protocol_row['horizon'],protocol_row['frames'],protocol_row['checkpoint_updates']),
                         (a.EVALUATION_SCHEMA,'final_evaluation',mode,split,arm,seed,295,301,25000),'observed protocol identity')
            checks.equal(protocol_row['schedule'],schedules[split][mode],'exact observed protocol schedule')
            checks.equal(protocol_row['source_population_count'],populations[split],'original source population count')
        keys = {}
        for cell in stage['cells']:
            ident = a.unit(cell, mode)
            if ident not in rows:
                checks.require(isinstance(cell.get('reason'),str) and bool(cell['reason']),'missing observed cell reason')
                continue
            row = rows[ident]
            key = f'{arm}_seed{seed}__{name}__{ident[0]:06d}_{ident[1]:03d}'
            task = {'key':key,'arm':arm,'seed':seed,'stage':name,'mode':mode,'split':split,'directory':directory,
                    'row':row,'cell':cell,'bounds':bounds.tolist(),
                    'source':next(x for x in schedules[split]['full-rollout'] if x['source_index']==ident[0]),
                    'protocol_sha256':files[protocol]['sha256'],
                    'files':{directory+'/'+suffix:files[directory+'/'+suffix] for suffix in (cell['row_file'],row['artifact_file'])}}
            task_sha = digest(encode(task))
            tasks.append((key, task_sha, task))
            keys[ident] = (key, task_sha)
        prepared.append((stage, keys))
    need(len(prepared)==18,'complete 18 observed stages')
    need(all(type(v['bytes'])is int and 0<=v['bytes']<=(common.CAP_ARCHIVE if p.endswith('.npz') else common.CAP_JSON) and common.is_pin(v['sha256']) for p,v in allowed.items()),'all selected file bounds/pins')
    need(len({key for key,_,_ in tasks})==len(tasks),'unique complete row worklist')
    return checks, collection, full_accounting, schedules, ref, sources, allowed, prepared, tasks


def merge(common, a, checks, full_accounting, schedules, ref, sources, allowed, prepared, store, identity, pins):
    """Merge in original fixed cell order, including shared identity and denominator checks."""
    models, shared = [], {}
    for stage, keys in prepared:
        arm,seed,name=stage['arm'],stage['seed'],stage['stage'];mode,split=stage['mode'],stage['split'];values={};details=[]
        for cell in stage['cells']:
            ident = a.unit(cell, mode)
            if ident not in keys:
                continue
            key, task_sha = keys[ident]
            result = store.read(key, task_sha)
            need(result is not None, 'required audited row missing: ' + key)
            detail = result['detail']
            checks.equal(detail['unit'],list(ident),'checkpoint row identity')
            checks.equal(detail['status'],'complete' if cell['state']=='completed_required_outcome' else 'failed','checkpoint original status')
            checks.equal(detail['failure'],cell.get('failure'),'checkpoint original failure')
            checks.count += result['checks']
            checks.context=f'{arm}/seed{seed}/{name}/{ident}'
            for key,value in result['pairings'].items():
                if key in shared:checks.equal(shared[key],value,'observed shared input or model-independent graph identity')
                shared[key]=value
            values[ident]=detail['metrics'];details.append(detail)
        expected=[a.unit(x,mode) for x in schedules[split][mode]];aggregates={}
        for metric in a.diagnostic_keys(mode):
            computed=a.aggregate(expected,{k:v.get(metric) for k,v in values.items()});aggregates[metric]=computed
            if metric in a.diagnostic.expected_metric_keys(mode):
                original=a.tree_leaf(stage['diagnostic_summary'],metric)
                checks.close({k:original[k] for k in computed},computed,'original diagnostic aggregation arithmetic')
        models.append({'arm':arm,'seed':seed,'stage':name,'mode':mode,'split':split,'cells':stage['cells'],'coverage':dict(Counter(c['state'] for c in stage['cells'])),'rows':details,'aggregates':aggregates})
    return {'schema':common.AUDIT_SCHEMA,'status':'passed_scoped_observed_history_checks','dataset':'Goop-3D','endpoint_updates':25000,
            'collection_sha256':common.COLLECTION_SHA,'cohort_sha256':common.COHORT_SHA,'phase_sha256':None,
            'prior_failed_phase_sha256':PRIOR_FAILED_PHASE,'original_expired_phase_sha256':common.OLD_PHASE_SHA,
            'successor_protocol_sha256':identity['protocol_sha256'],'successor_source_pins_sha256':identity['source_pins_sha256'],
            'source_manifest_sha256':identity['original_source_manifest_sha256'],'physical_reference':ref,'checks':checks.count,
            'source_schedules':schedules,'all_original_accounting':full_accounting,'required_all_cells':4728,'required_observed_cells':2568,
            'models':models,'shared_identity_sha256':shared,'verified_observed_files_sha256':{p:v['sha256'] for p,v in allowed.items()},
            'source_sha256':sources,'source_bindings_sha256':pins['frozen_files_sha256']['worker_source_bindings.json'],
            'original_full_audit_rerun':False,'whole_collection_parsed_for_accounting':True,
            'autonomous_queue_files_or_arrays_opened':False,'autonomous_numerical_aggregation_performed':False,'scientific_admission':False,
            'limitations':['Conditional observed-history arithmetic only; no autonomous rollout conclusion.',
                           'Saved normalization/source truth inherited; model and original source arrays not reopened.',
                           'Fixed source-order grid is not a probability sample; all missing values retain their denominators.',
                           'Diagnostic timing arithmetic does not establish clock truth or end-to-end speedup.',
                           'This successor has no timed phase: legacy phase_sha256 is null; protocol/source hashes identify this run.']}


def event(stream, value):
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    stream.flush()
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def execute(args, pins, protocol_pin, source_pin):
    common, audited, a, summarized, checked = load_science(args.frozen_dir, pins)
    output = args.output.resolve()
    protected = [Path(common.QUEUE),Path(common.COLLECTION).parent,Path(common.OUTPUT),Path(args.frozen_dir).resolve()]
    need(args.output.is_absolute() and output==args.output and all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected), 'separate canonical successor output required')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output/'completion.json').exists():
        receipt = strict((output/'completion.json').read_bytes())
        need(receipt['status']=='all_required_observed_rows_audited_and_arithmetic_verified' and
             receipt['source_pins_sha256']==source_pin and receipt['protocol_sha256']==protocol_pin,
             'existing completion belongs to a different run')
        need(file_hash(output/'checkpoint_index.json')==receipt['checkpoint_index_sha256'],'completed checkpoint index changed')
        need(set(receipt['products_sha256'])=={'audit.json','summary.json','arithmetic_check.json'},'complete product set')
        for name,pin in receipt['products_sha256'].items():need(file_hash(output/name)==pin,'completed product changed')
        print(json.dumps({'status':'already_completed','completion_sha256':file_hash(output/'completion.json'),
                          'no_scientific_reexecution':True}),flush=True)
        lock.close()
        return
    attempts = output / 'attempts';attempts.mkdir(exist_ok=True)
    attempt = attempts / f'{len(list(attempts.iterdir()))+1:06d}';attempt.mkdir()
    started = time.monotonic()
    progress = (attempt / 'progress.jsonl').open('x')
    try:
        checks, collection, full_accounting, schedules, ref, sources, allowed, prepared, tasks = prepare(common,audited,a,args.frozen_dir,pins)
        identity = {'source_pins_sha256':source_pin,'protocol_sha256':protocol_pin,'collection_sha256':common.COLLECTION_SHA,
                    'original_source_manifest_sha256':collection['source_manifest_sha256'],
                    'python_version':platform.python_version(),'python_executable_sha256':file_hash(Path(sys.executable).resolve()),
                    'numpy_version':audited.np.__version__,'frozen_files_sha256':pins['frozen_files_sha256']}
        store = Checkpoints(output, identity)
        expected = {key:task_sha for key,task_sha,_ in tasks}
        need(set(store.index['completed'])<=set(expected) and set(store.index['failed'])<=set(expected),'checkpoint index contains unexpected cells')
        pending = []
        for key, task_sha, task in tasks:
            if store.read(key, task_sha) is not None:
                continue
            if key in store.index['failed'] and not args.retry_failed:
                need(store.index['failed'][key]['task_sha256']==task_sha,'failed task binding differs')
                continue
            pending.append((key, task_sha, task))
        event(progress, {'status':'prepared','required_observed_cells':2568,'required_all_cells':4728,'required_committed_rows':len(tasks),
                         'reused_rows':len(store.index['completed']),'pending_rows':len(pending),'previous_failed_rows':len(store.index['failed']),
                         'selected_files':len(allowed),'selected_bytes':sum(v['bytes'] for v in allowed.values()),'workers':args.workers})
        iterator = iter(pending)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize_worker, initargs=(str(args.frozen_dir),pins)) as pool:
            inflight = {}
            def fill():
                while len(inflight)<2*args.workers:
                    try:key,task_sha,task=next(iterator)
                    except StopIteration:return
                    inflight[pool.submit(audit_cell,task)]=(key,task_sha)
            fill()
            while inflight:
                done,_=wait(inflight,return_when=FIRST_COMPLETED)
                for future in done:
                    key,task_sha=inflight.pop(future)
                    try:
                        result=future.result()
                    except Exception as error:
                        store.failure(key,task_sha,error,attempt);status='row_failed_not_admitted'
                    else:
                        store.commit(key,task_sha,result);status='row_checkpoint_committed'
                    event(progress,{'status':status,'key':key,'completed_rows':len(store.index['completed']),
                                    'failed_rows':len(store.index['failed']),'required_rows':len(tasks),
                                    'elapsed_seconds':round(time.monotonic()-started,3)})
                fill()
        need(not store.index['failed'] and set(store.index['completed'])==set(expected), 'audit has failed or missing rows; durable successful rows retained')
        result=merge(common,a,checks,full_accounting,schedules,ref,sources,allowed,prepared,store,identity,pins)
        event(progress,{'status':'final_input_rehash','files':len(allowed),'bytes':sum(v['bytes'] for v in allowed.values())})
        for path,entry in allowed.items():
            common.read_bound(path,entry['sha256'],common.CAP_ARCHIVE if path.endswith('.npz') else common.CAP_JSON,
                              expected_bytes=entry['bytes'],retain=False)
        common.read_bound(common.COLLECTION,common.COLLECTION_SHA,common.CAP_COLLECTION,expected_bytes=common.COLLECTION_BYTES,retain=False)
        for path,pin in sources.items():common.read_bound(path,pin,common.CAP_JSON,retain=False)
        for name,pin in pins['frozen_files_sha256'].items():need(file_hash(Path(args.frozen_dir)/name)==pin,'final frozen source changed')
        need(file_hash(HERE/'run_observed_resumable.py')==pins['runner_sha256'] and file_hash(HERE/'protocol.json')==protocol_pin and file_hash(HERE/'source_pins.json')==source_pin,'successor source/protocol changed')
        summary=summarized.summarize(result)
        arithmetic=checked.verify(result,summary)
        for product in (summary,arithmetic):
            product.update(successor_protocol_sha256=protocol_pin,successor_source_pins_sha256=source_pin,prior_failed_phase_sha256=PRIOR_FAILED_PHASE)
        audit_pin=atomic_write(output/'audit.json',result)
        summary['audit_sha256']=audit_pin;summary_pin=atomic_write(output/'summary.json',summary)
        arithmetic.update(audit_sha256=audit_pin,summary_sha256=summary_pin)
        arithmetic_pin=atomic_write(output/'arithmetic_check.json',arithmetic)
        receipt={'schema':'goop3d_resumable_observed_completion_v1','status':'all_required_observed_rows_audited_and_arithmetic_verified',
                 'required_observed_cells':2568,'required_all_cells':4728,'audited_rows':len(tasks),'source_pins_sha256':source_pin,
                 'protocol_sha256':protocol_pin,'checkpoint_index_sha256':file_hash(store.path),
                 'products_sha256':{'audit.json':audit_pin,'summary.json':summary_pin,'arithmetic_check.json':arithmetic_pin},
                 'workers':args.workers,'elapsed_seconds':time.monotonic()-started,'prior_attempts_unchanged':True,
                 'automatic_scientific_admission':False,'arbitrary_elapsed_cutoff':False}
        receipt_pin=atomic_write(output/'completion.json',receipt)
        event(progress,{'status':'completed','completion_sha256':receipt_pin,'products_sha256':receipt['products_sha256']})
    except BaseException as error:
        atomic_write(attempt/'failure.json',{'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
                                          'completed_checkpoints_retained':True,'scientific_admission':False})
        raise
    finally:
        progress.close()
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--frozen-dir',type=Path,default=Path(DEFAULT_FROZEN))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--source-pins-sha256')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--retry-failed',action='store_true',help='Explicitly retry prior failed rows; keep original failure records.')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'status':'inert_resumable_observed_audit','scientific_execution':False}));return
    need(args.output is not None and 1<=args.workers<=32,'explicit output and1..32workers required')
    need(file_hash(HERE/'source_pins.json')==args.source_pins_sha256 and file_hash(HERE/'protocol.json')==args.protocol_sha256,'exact source/protocol pins required')
    pins=strict((HERE/'source_pins.json').read_bytes())
    need(file_hash(Path(__file__))==pins['runner_sha256'],'runner source differs')
    execute(args,pins,args.protocol_sha256,args.source_pins_sha256)


if __name__=='__main__':
    main()
