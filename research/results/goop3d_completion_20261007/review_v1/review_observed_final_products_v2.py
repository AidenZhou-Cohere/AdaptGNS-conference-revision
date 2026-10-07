"""Actual-product scalar review. No scientific imports, arrays or inference."""
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE / 'root'
TRANSFER = ROOT / 'observed_finalized_v2_collection'
FILES = TRANSFER / 'files'
PRODUCTS = FILES / 'observed_finalized_v2'
CACHE = ROOT / 'observed_failure_collection_v1/files/observed_results_v1'
FROZEN = BASE.parent / 'deadline_research_20261005/cuda_preparation/goop3d_observed_history_analysis_UNADMITTED_transition_v1'
REMOTE = '/root/repos/AdaptGNS-cuda-20261006/goop3d_completion_20261007/observed_results_v1'
JOB = 'observed_cache_finalize_v2_attempt1'
counts = Counter()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    def unique(pairs):
        value = {}
        for k, v in pairs:
            if k in value:
                raise ValueError('duplicate key')
            value[k] = v
        return value
    def bad(value):
        raise ValueError('nonfinite JSON constant ' + value)
    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=bad)

def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()

def check(ok, label):
    if not ok:
        raise AssertionError(label)
    counts[label.split(':', 1)[0].split('/', 1)[0]] += 1

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)

def equal(got, expected, label):
    if isinstance(expected, dict):
        check(type(got) is dict and got.keys() == expected.keys(), label + ':keys')
        for k, value in expected.items():
            equal(got[k], value, label + '/' + str(k))
    elif isinstance(expected, list):
        check(type(got) is list and len(got) == len(expected), label + ':length')
        for k, value in enumerate(expected):
            equal(got[k], value, label + '/' + str(k))
    elif expected is None or type(expected) in (str, bool):
        check(got == expected, label + ':value')
    else:
        check(finite(got) and math.isclose(got, expected, rel_tol=1e-10, abs_tol=1e-12), label + ':numeric')

def mean(values):
    return math.fsum(values) / len(values) if values and all(finite(v) for v in values) else None

def stats(values):
    check(len(values) == 3, 'statistic:three fixed seeds')
    complete = all(finite(v) for v in values)
    sd = None
    if complete:
        exact = [Fraction(v) for v in values]
        center = sum(exact) / 3
        variance = sum((v - center) ** 2 for v in exact) / 2
        with localcontext() as context:
            context.prec = 80
            sd = float((Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt())
    return {'seed_values': dict(zip(('0', '1', '2'), values)), 'required_seed_pairs': 3,
            'defined_seed_pairs': sum(finite(v) for v in values), 'mean': mean(values), 'sample_sd': sd}

def delta(a, b):
    return a - b if finite(a) and finite(b) else None

manifest = load(FILES / 'transfer_manifest.json')
check(sha(FILES / 'transfer_manifest.json') == 'a2b9d1c9b5ee09834acc2859032aad8ad8bbe9c43e7c83e38a709dd4e05efcaf', 'transport:exact manifest')
check(len(manifest['files']) == 11 and sum(v['bytes'] for v in manifest['files'].values()) == 34079036, 'transport:complete eleven files')
check({str(p.relative_to(FILES)) for p in FILES.rglob('*') if p.is_file()} == set(manifest['files']) | {'transfer_manifest.json'}, 'transport:exact local file set')
for name, entry in manifest['files'].items():
    path = FILES / name
    check(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'transport:exact file ' + name)
transport = load(TRANSFER / 'original_transfer.json')
check(transport['exit_code'] == 0 and transport['archive_sha256'] == sha(TRANSFER / 'original_archive.tar.gz'), 'transport:actual successful archive')
check(manifest['remaining_group_members'] == [] and manifest['actual_status']['returncode'] == 0, 'closure:actual zero exit and empty group')
status = load(FILES / 'jobs' / JOB / 'status.json')
check(status == manifest['actual_status'], 'closure:status retained exactly')
check(status['owner'] == {'pid':87032, 'start_ticks':832705409, 'pgid':87032}, 'closure:original owner')
check(status['child'] == {'pid':87033, 'start_ticks':832705413, 'pgid':87032}, 'closure:original child')
check(status['status'] == 'exited' and status['finished_utc_seconds'] == 1791340522.5070531, 'closure:actual completion time')
spec = load(FILES / 'jobs' / JOB / 'spec.json')
check(spec == load(ROOT / (JOB + '.json')) and sha(FILES / 'jobs' / JOB / 'spec.json') == status['spec_sha256'], 'closure:issued argv exact')
launch = load(ROOT / (JOB + '.launch.json'))
launch_record = json.loads(launch['stdout'])
check(launch['exit_code'] == 0 and launch_record['owner_pid'] == 87032 and launch_record['owner_start_ticks'] == 832705409, 'closure:original launch matches')
check(manifest['host'] == spec['hostname'] == launch_record['hostname'] and manifest['boot_id'] == spec['boot_id'] == launch_record['boot_id'], 'closure:same native host boot')
observed = load(ROOT / 'observed_finalizer_status_023531.json')
native = json.loads(observed['stdout'])['jobs'][JOB]
check(observed['exit_code'] == 0 and native['native'] == {'owner':None, 'child':None}, 'closure:separate root observation absent')
check(native['status.json'] == status and native['spec.json'] == spec, 'closure:observation status and spec exact')

completion = load(PRODUCTS / 'completion.json')
audit = load(PRODUCTS / 'audit.json')
summary = load(PRODUCTS / 'summary.json')
arithmetic = load(PRODUCTS / 'arithmetic_check.json')
check(sha(PRODUCTS / 'completion.json') == '148c81a072f33458a8a3f69862c33ef033ebee96027cd7e9c8198b4b282429a5', 'products:exact final completion')
for name, pin in completion['products_sha256'].items():
    check(sha(PRODUCTS / name) == pin, 'products:completion binds ' + name)
check(summary['audit_sha256'] == arithmetic['audit_sha256'] == completion['products_sha256']['audit.json'], 'products:summary check audit binding')
check(arithmetic['summary_sha256'] == completion['products_sha256']['summary.json'], 'products:check summary binding')
check(arithmetic['checked_model_metric_aggregates'] == 2436, 'products:2436 model metric aggregates')
check(completion['cache_rows_consumed'] == 2568 and completion['saved_row_array_audit_calls'] == completion['original_arrays_decoded'] == 0, 'products:cache-only completion')
check(completion['opaque_original_files_rehashed'] == 5154 and completion['opaque_original_bytes_rehashed'] == 45775548716, 'products:full original opaque input rehash')
check(completion['original_cache_and_failed_attempt_unchanged'] is True and completion['arbitrary_elapsed_cutoff'] is False, 'products:original history and untimed successor')
check(len(audit['verified_observed_files_sha256']) == 5154, 'products:all selected input pins')
for product in (audit, summary, arithmetic, completion):
    check(product['required_all_cells'] == 4728 and product['required_observed_cells'] == 2568, 'products:fixed denominators')
    check(product['comparison_tolerance_unchanged'] is True, 'products:original comparison tolerance')
    check(product['finisher_source_sha256'] == sha(BASE / 'observed_finalize_v2/finish_from_cache.py') == '8b6f46e24d2cc729fa7c585adce8ad6acae0cae85764dc4f3fc7ceaaaf698a6e', 'sources:exact finalizer')
    check(product['corrected_checker_source_sha256'] == sha(BASE / 'observed_finalize_v2/check_observed_exact_variance_v2.py') == '9dce53c8f9e87ef1d0a828b163469f34c9249f5b242f26c3a0287f11f489bccb', 'sources:exact corrected checker')
    check(product['old_checker_source_sha256'] == sha(FROZEN / 'check_goop3d_observed_history_summary_v1.py'), 'sources:old failed checker retained')
    check(product['original_cache_index_sha256'] == sha(CACHE / 'checkpoint_index.json'), 'cache:original index pin')
    check(product['original_failed_check_sha256'] == sha(CACHE / 'attempts/000001/failure.json'), 'cache:original failed attempt pin')
check((PRODUCTS / 'retained_original_failure.json').read_bytes() == (CACHE / 'attempts/000001/failure.json').read_bytes(), 'cache:byte-identical original failure')
for label, path in [('runner', BASE / 'observed_v1/run_observed_resumable.py'), ('source_pins', BASE / 'observed_v1/source_pins.json'), ('protocol', BASE / 'observed_v1/protocol.json')]:
    check(completion['original_' + label + '_sha256'] == sha(path), 'sources:original ' + label)
for name, pin in launch['sources'].items():
    local_source = ROOT / 'job_owner.py' if name == 'execution_v1/job_owner.py' else BASE / name
    check(sha(local_source) == pin, 'sources:actual launched ' + name)

index = load(CACHE / 'checkpoint_index.json')
check(len(index['completed']) == 2568 and not index['failed'], 'cache:exact all passing rows')
identity = index['identity']
identity_sha = hashlib.sha256(encode(identity)).hexdigest()
check(identity['python_version'] == '3.12.3' and identity['numpy_version'] == '2.5.3', 'cache:original runtime identity')
check(identity['python_executable_sha256'] == '6242e0e8650d7dbdebbc25e08bf4c9359ddaf65f54fcae0e57fa99395fa5357a', 'cache:original interpreter hash')
for name, pin in identity['frozen_files_sha256'].items():
    check(sha(FROZEN / name) == pin, 'sources:frozen scientific source ' + name)
expected_bindings = {REMOTE + '/checkpoint_index.json': sha(CACHE / 'checkpoint_index.json')}
cache_rows = {}
for key, entry in index['completed'].items():
    path = CACHE / 'rows' / (key + '.json')
    check(sha(path) == entry['sha256'], 'cache:row byte hash ' + key)
    value = load(path)
    check(value['key'] == key and value['task_sha256'] == entry['task_sha256'] and value['identity_sha256'] == identity_sha, 'cache:row source task identity ' + key)
    cache_rows[key] = value['result']
    expected_bindings[REMOTE + '/rows/' + key + '.json'] = entry['sha256']
check(completion['cache_row_sha256'] == expected_bindings, 'cache:all2569 final completion bindings exact')

plan = load(BASE / 'autonomous_v1/plan.json')
ledger_path = Path(plan['original_local_ledger_path'])
check(sha(ledger_path) == plan['original_ledger_sha256'], 'ledger:original bytes')
ledger = load(ledger_path)
original = {(s['arm'],s['seed'],s['stage']):s for s in ledger['stages']}
accounting = audit['all_original_accounting']
check(len(accounting) == 30 and len(original) == 30, 'ledger:all30 original stages')
check(summary['all_original_accounting'] == accounting, 'ledger:summary retains all states')
for stage in accounting:
    old = original[stage['arm'],stage['seed'],stage['stage']]
    check(stage == {k:old[k] for k in stage}, 'ledger:unaltered full stage ' + stage['stage'])
check(Counter(c['state'] for s in accounting for c in s['cells']) == {'completed_required_outcome':2899,'not_completed_before_invocation_end':1817,'timed_out_current':12}, 'ledger:all4728 historical states retained')
families = ('clean_validation','same_state_valid','same_state_test')
models = {(m['arm'],m['seed'],m['stage']):m for m in audit['models']}
check(set(models) == {(a,s,n) for a in ('base','mix') for s in range(3) for n in families}, 'grid:all18 observed model stages')
check(sum(len(m['rows']) for m in models.values()) == 2568, 'grid:all2568 audited row results')
shared = {}; means = {}; used = set()
for key, model in models.items():
    arm, seed, stage = key
    check(model['cells'] == original[key]['cells'], 'grid:exact original cells ' + str(key))
    schedule = audit['source_schedules'][model['split']][model['mode']]
    expected = [(r['source_index'],r['target_frame']) for r in schedule]
    check(len(expected) == (128 if stage == 'clean_validation' else 150), 'grid:fixed family row count')
    check([(r['source_index'],r['target_frame']) for r in model['cells']] == expected, 'grid:exact schedule')
    check(Counter(c['state'] for c in model['cells']) == {'completed_required_outcome':len(expected)} == model['coverage'], 'grid:full family complete')
    if stage != 'clean_validation':
        check(set(expected) == {(j*99//29,t) for j in range(30) for t in (7,80,153,226,300)}, 'grid:fixed30source150history grid')
    metrics = {}
    for row in model['rows']:
        source,target = row['unit']
        name = f'{arm}_seed{seed}__{stage}__{source:06d}_{target:03d}'
        cached = cache_rows[name]; used.add(name)
        check(row == cached['detail'], 'rows:final row equals original passing cache ' + name)
        check(row['status'] == 'complete' and row['failure'] is None, 'rows:complete without failure')
        metrics[source,target] = row['metrics']
        for label,pin in cached['pairings'].items():
            check(label not in shared or shared[label] == pin, 'pairing:cross-model/source shared identity')
            shared[label] = pin
    check(set(metrics) == set(expected), 'grid:rows exactly complete schedule')
    for metric, aggregate in model['aggregates'].items():
        trajectories = {}
        for source in sorted({u[0] for u in expected}):
            values = [metrics[u].get(metric) for u in expected if u[0] == source]
            trajectories[str(source)] = {'expected':len(values), 'defined':sum(finite(x) for x in values), 'mean':mean(values)}
        wanted = {'expected_frames':len(expected), 'defined_frames':sum(v['defined'] for v in trajectories.values()),
                  'expected_trajectories':len(trajectories), 'defined_trajectories':sum(v['mean'] is not None for v in trajectories.values()),
                  'equal_trajectory_mean':mean([v['mean'] for v in trajectories.values()]), 'trajectories':trajectories}
        equal(aggregate,wanted,'aggregate')
        means[arm,seed,stage,metric] = wanted['equal_trajectory_mean']
check(used == set(cache_rows), 'rows:every cached row consumed exactly')
check(shared == audit['shared_identity_sha256'], 'pairing:complete original shared identity map')
check(audit['autonomous_queue_files_or_arrays_opened'] is False and audit['autonomous_numerical_aggregation_performed'] is False, 'scope:no autonomous numerical admission')

policies = ('base','dense','random25','speed25','relative-velocity-RMS25','previous-observed-base-risk25')
statistic_objects = 0
for stage, family in summary['families'].items():
    check(stage in families and family['full_family_complete'] is True, 'summary:full family')
    check(family['required_cells'] == (768 if stage == 'clean_validation' else 900), 'summary:family fixed denominator')
    check(family['coverage_by_model'] == {f'{a}_seed{s}':models[a,s,stage]['coverage'] for a in ('base','mix') for s in range(3)}, 'summary:per-model coverage')
    keys = set(models['base',0,stage]['aggregates'])
    check(all(set(models[a,s,stage]['aggregates']) == keys for a in ('base','mix') for s in range(3)), 'summary:full metric key set')
    def value(arm,seed,key): return means[arm,seed,stage,key]
    for arm in ('base','mix'):
        check(set(family['absolute'][arm]) == keys, 'summary:all absolute metrics')
        for metric in keys:
            equal(family['absolute'][arm][metric],stats([value(arm,s,metric) for s in range(3)]),'summary_stats')
            statistic_objects += 1
    check(set(family['mix_minus_base_training']) == keys, 'summary:all paired exposure metrics')
    for metric in keys:
        equal(family['mix_minus_base_training'][metric],stats([delta(value('mix',s,metric),value('base',s,metric)) for s in range(3)]),'summary_stats')
        statistic_objects += 1
    if stage != 'clean_validation':
        check(set(family['accuracy_policy_contrasts']) == {'position_coordinate_mse','normalized_coordinate_mse'}, 'summary:both accuracy metrics')
        for metric, contrasts in family['accuracy_policy_contrasts'].items():
            def loss(arm,seed,p): return value(arm,seed,'accuracy/'+p+'/'+metric)
            expected = {arm:{p+'_minus_'+ref:stats([delta(loss(arm,s,p),loss(arm,s,ref)) for s in range(3)])
                      for ref in ('base','random25','speed25','relative-velocity-RMS25') for p in policies if p != ref} for arm in ('base','mix')}
            equal(contrasts['within_arm'],expected,'summary_contrasts')
            statistic_objects += sum(len(v) for v in expected.values())
            expected = stats([delta(delta(loss('mix',s,policies[-1]),loss('mix',s,'random25')),delta(loss('base',s,policies[-1]),loss('base',s,'random25'))) for s in range(3)])
            equal(contrasts['risk_minus_random_mix_minus_base'],expected,'summary_interaction')
            statistic_objects += 1
check(statistic_objects == 1382, 'summary:all1382 statistics and5528 scalar leaves')
diagnosis_path = HERE / 'observed_arithmetic_diagnosis_v1/diagnosis.json'
diagnosis = load(diagnosis_path)
check(sha(diagnosis_path) == '02685d533fff4f3e43da3eb7354003b315bf5f376fbcb5971f84b40993821375', 'diagnosis:original evidence exact')
check(diagnosis['mismatch_count'] == 36 and diagnosis['scalar_statistic_leaves_compared'] == 5528, 'diagnosis:all36 false SDs')
for mismatch in diagnosis['mismatches']:
    target = summary
    for name in mismatch['path_components']:
        target = target[name]
    check(target == mismatch['summarizer_value'] == 0, 'diagnosis:unchanged summary exact zero SD')
    check(mismatch['exact_rational_oracle']['all_three_input_floats_identical'] is True, 'diagnosis:only repeated-count issue')

tables = {}
for stage in ('same_state_valid','same_state_test'):
    family = summary['families'][stage]
    tables[stage] = {
        'all_six_policy_position_mse': {a:{p:family['absolute'][a]['accuracy/'+p+'/position_coordinate_mse'] for p in policies} for a in ('base','mix')},
        'all_six_policy_exposure_position_mse': {p:family['mix_minus_base_training']['accuracy/'+p+'/position_coordinate_mse'] for p in policies},
        'risk_minus_random_position_mse': {a:family['accuracy_policy_contrasts']['position_coordinate_mse']['within_arm'][a]['previous-observed-base-risk25_minus_random25'] for a in ('base','mix')},
        'risk_minus_random_exposure_interaction':family['accuracy_policy_contrasts']['position_coordinate_mse']['risk_minus_random_mix_minus_base'],
    }
report = {
    'status':'passed_independent_actual_product_scalar_review',
    'recommended_admission':'Complete scoped observed-history comparison only; autonomous completion remains separate.',
    'checks':sum(counts.values()), 'check_categories':dict(sorted(counts.items())),
    'transfer_manifest_sha256':sha(FILES/'transfer_manifest.json'), 'completion_sha256':sha(PRODUCTS/'completion.json'),
    'products_sha256':completion['products_sha256'], 'actual_original_child':status['child'], 'actual_original_exit':0,
    'actual_finish_utc_seconds':status['finished_utc_seconds'], 'native_closure':'Original owner/child absent in root saved observation; complete group empty in collection manifest.',
    'source_review_sha256':sha(HERE/'observed_finalizer_v2_source_review.json'),
    'original_index_sha256':sha(CACHE/'checkpoint_index.json'), 'original_failure_sha256':sha(CACHE/'attempts/000001/failure.json'),
    'cached_rows_bound_exactly':2568, 'all_original_accounting_states':4728, 'observed_model_stages':18,
    'independent_model_metric_aggregates':2436, 'independent_statistic_objects':statistic_objects,
    'original36rounding_mismatches_preserved_and_resolved':True, 'array_model_or_source_execution':False,
    'initial_reviewer_path_expectation_error':'Remote execution_v1/job_owner.py maps to local root/job_owner.py; initial missing-path tool and original reviewer source preserved. Exact launched hash matches.',
    'scientific_scope':['Six25kmodels/threepairedseeds; 768clean-validation and900same-state cells per valid/test family.',
        'Observed states and histories only. Previous-observed-base-risk placement does not establish autonomous feedback quality.',
        'Means use the complete fixed source grids with equal source weights; SD is sample SD over three training seeds, not frames.',
        'Saved normalization/source values are inherited; no unsaved trajectory, source execution or model inference replay.',
        'Original D3 autonomous timeouts and all4728 accounting states remain historical; final autonomous completion uses separate products.'],
    'tables':tables,
}
output = HERE / 'observed_final_products_v2_review.json'
output.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps({'review':str(output),'sha256':sha(output),'checks':sum(counts.values()),'statistic_objects':statistic_objects,'status':report['status']}))
