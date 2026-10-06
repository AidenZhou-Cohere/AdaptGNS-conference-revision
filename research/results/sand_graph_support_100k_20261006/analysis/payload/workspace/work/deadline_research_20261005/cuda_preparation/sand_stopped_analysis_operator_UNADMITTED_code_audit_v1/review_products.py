"""Pure decoded-metadata checks for the six frozen Sand stopped operations.

The caller binds every decoded product/control to captured bytes before calling.
No files, clocks, scientific modules, arrays or models are opened or imported.
Counts, identity, null propagation and predecessor links are checked here; the
frozen remote workers supply numerical arithmetic and supported array checks.
Passing is neither complete scientific coverage nor scientific admission.
"""
from collections import Counter
from functools import wraps
import hashlib
import json
import math
from pathlib import PurePosixPath

REMOTE = '/root/repos/AdaptGNS-cuda-20261006'
PREP = REMOTE + '/cuda_preparation'
ANALYSIS = REMOTE + '/sand_final_analysis_20261006_v1'
COHORT_PATH = REMOTE + '/sand_post_training_A_20261006_v1/cohort/cohort.json'
COLLECTOR = '1e70f1689a75c6f68eb970048aab7837153c19f70caf30ec19143cf66ab5d272'
COHORT = '0e73e272a4ff386ca1e89bd010c476a82f993ced8f32176144f53b779f32d8ba'
PROTOCOL = 'e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d'
TRAINER = 'fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124'
BENCHMARK = '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13'
AMENDMENT = '411547c3355859906f59792b3b493841030b0afd2e105e17e466114df6743738'
SAVED = 'a7e47c8e4740cc8c1767dc778ef4fa5291453f4142a0753cc62a0bf82bea7740'
DIAGNOSTIC = '700695b69691b9da45e31ce7a9821900a12ff882be5b4a09927ff8b6dd0f8cdd'
PAIRED = '541bdc67a99e44ccb263b1bbe439fa37132ef95ab597e1dd440742c0bb4a7a2d'
SOURCES = {
    'summarize_goop_graph_support_quota_v2.py': 'da85058ea2ce0fc0f1b67e6cad442369835dfe1e8926d8f26c595d143f1ec33c',
    'supervise_sand_final_evaluation_scoped_v1.py': 'efebe762ea60b1b711925fb9bb3bc91461a16b1440491b2fa554c99171773829',
    'supervise_goop_evaluation_gpu_scoped_v3.py': 'a13ca1bc30161f89cf79b6c0b632c65a272913302ef0cc37dcfd7ed762c6aa21',
    'evaluate_sand_graph_support_final.py': '952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58',
}
SOURCE_INPUTS = {PREP + '/' + k: v for k, v in SOURCES.items()}
SOURCE_INPUTS[PREP + '/summarize_sand_graph_support_scoped_v1.py'] = COLLECTOR
SAVED_SOURCES = {PREP + '/sand_saved_array_audit_v2/audit_sand_saved_arrays_v1.py': SAVED,
                 PREP + '/sand_saved_array_audit_v2/sand_saved_diagnostic_audit_v1.py': DIAGNOSTIC}
PAIRED_SOURCES = {**SAVED_SOURCES, PREP + '/sand_saved_array_audit_v2/audit_sand_paired_arrays_v1.py': PAIRED}
MODELS = {'A': (('base', 1), ('mix', 1), ('base', 2), ('mix', 2)), 'B': (('base', 0), ('mix', 0))}
GPUS = {'A': (0, 1, 2, 3), 'B': (2, 3)}
STAGES = {'full_rollout_test': ('full-rollout', 'test'), 'same_state_valid': ('same-state', 'valid'),
          'same_state_test': ('same-state', 'test'), 'clean_validation': ('clean-validation', 'valid')}
POLICIES = ('base', 'dense', 'random25', 'speed25', 'laggedrisk25', 'relative-velocity-RMS25')
DIAG_POLICIES = ('base', 'dense', 'random25', 'speed25', 'relative-velocity-RMS25', 'previous-observed-base-risk25')
STATES = frozenset(('completed_required_outcome', 'recorded_failed_outcome', 'timed_out_current',
                    'not_completed_before_invocation_end', 'never_started'))
COMMITTED = frozenset(('completed_required_outcome', 'recorded_failed_outcome'))
BOUNDARIES = ('fraction_particles_outside', 'fraction_particles_outside_by_more_than_1e-6',
              'maximum_coordinate_excursion', 'mean_particle_maximum_excursion')
FULL_METRICS = ('mean_rollout_mse', 'mse_forecast200', 'mse_forecast314') + tuple(
    side + '_boundary_' + key for side in ('predicted', 'ground_truth')
    for key in tuple('mean_' + name for name in BOUNDARIES) + ('trajectory_maximum_excursion',))
SAVED_SCOPE = {
    'array_recomputed': 'All saved forecast errors, available prediction/truth boundary states, and supported saved diagnostic scalar formulas.',
    'scalar_recomputed': 'Full/prefix error and full-H boundary aggregates from complete recorded scalar series; equal-frame/equal-source diagnostic hierarchy.',
    'unsupported': ['Unsaved full-trajectory positions/errors and complete temporal replay.',
        'Official source-position/normalization truth independently reloaded from dataset.',
        'Model/checkpoint/optimizer execution or fresh native parity/model verification.',
        'Completeness of geometric candidate sets, spatial radius search, or unsaved graph actions.',
        'Measured clock truth, hardware isolation, or causal speedup.']}
PAIRED_SCOPE = 'Same saved-array/scalar-series scope as input audits; all fixed seed means/sampleSD/contrasts and null propagation independently recomputed. Operational runtime and external source/model truth not remeasured.'
COLLECTION_SCHEMA = 'adaptgns_sand_evaluation_collection_scoped_v1'
SUMMARY_SCHEMA = 'adaptgns_sand_graph_support_paired_scalar_summary_scoped_v1'
Q_SCHEMA = 'adaptgns_sand_evaluation_gpu_scoped_v1'


def _digest(value): return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
def _finite(value): return type(value) in (int, float) and math.isfinite(value)
def _same(a, b):
    if type(a) is not type(b): return False
    if type(a) is dict: return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if type(a) in (list, tuple): return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


class Checks:
    def __init__(self): self.count = 0
    def need(self, condition, message):
        self.count += 1
        if not condition: raise ValueError(message)
    def equal(self, actual, expected, message): self.need(_same(actual, expected), message)


def _guard(function):
    @wraps(function)
    def run(*args, **kwargs):
        try: return function(*args, **kwargs)
        except ValueError: raise
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
            raise ValueError(function.__name__ + ': malformed product or expected controls: ' + str(error)) from error
    return run


def _json(value, c):
    if type(value) is dict:
        c.need(all(type(k) is str for k in value), 'JSON object keys must be strings')
        for item in value.values(): _json(item, c)
    elif type(value) is list:
        for item in value: _json(item, c)
    else: c.need(value is None or type(value) in (str, int, bool) or _finite(value), 'Finite JSON values required')


def _begin(product, actual, expected, keys, schema, status):
    c = Checks()
    c.need(type(product) is dict and type(expected) is dict, 'Decoded JSON objects required')
    c.equal(set(expected), set(keys), 'Exact expected-control keys required')
    c.need(_digest(actual) and _digest(expected['artifact_sha256']), 'Artifact byte pins required')
    c.equal(actual, expected['artifact_sha256'], 'Artifact byte pin differs')
    c.equal(product['schema'], schema, 'Product schema differs'); c.equal(product['status'], status, 'Product did not pass')
    _json(product, c)
    return c


def schedule(stage):
    if stage == 'full_rollout_test': return [(i, p) for i in range(30) for p in POLICIES]
    if stage == 'clean_validation': return [(n // 314, n % 314 + 6) for n in (i * 9419 // 127 for i in range(128))]
    return [(i, t) for i in range(30) for t in (7, 85, 163, 241, 319)]


def diagnostic_keys(stage):
    if stage == 'clean_validation':
        return tuple('metrics/' + k for k in ('normalized_acceleration_coordinate_mse', 'realized_normalized_vector_se',
                     'predicted_normalized_vector_se', 'constant_free_gaussian_nll'))
    keys = ['accuracy/' + p + '/' + k for p in DIAG_POLICIES for k in ('position_coordinate_mse', 'normalized_coordinate_mse')]
    for p in DIAG_POLICIES[1:]:
        names = ('mean_position_vector_benefit', 'mean_normalized_vector_benefit', 'positive_fraction', 'negative_fraction', 'zero_fraction')
        if p != 'dense': names += ('dense_sparse_sign_disagreement_fraction', 'dense_positive_sparse_nonpositive_fraction')
        keys += ['benefit/' + p + '/' + k for k in names]
    keys += ['timing/' + p + '/' + k for p in DIAG_POLICIES + ('natural_base_reference',) for k in
             ('end_to_end_seconds', 'score_generation_seconds', 'score_graph_seconds', 'score_forward_seconds',
              'current_graph_and_selection_seconds', 'current_forward_seconds')]
    keys += ['correlations/previous_risk_vs_base_error']
    keys += ['correlations/previous_risk_vs_' + p + '_benefit' for p in DIAG_POLICIES[1:]]
    keys += ['correlations/dense_vs_' + p + '_benefit' for p in DIAG_POLICIES[2:]]
    return tuple(keys)


def _unit(row, stage): return (row['source_index'], row['policy' if stage == 'full_rollout_test' else 'target_frame'])
def _queue(role): return REMOTE + '/sand_final_evaluation_' + role + '_20261006_v1'
def _rel(arm, seed, stage): return 'jobs/' + arm + '_seed' + str(seed) + '/' + stage
def _leaf(tree, path):
    for key in path.split('/'): tree = tree[key]
    return tree


def _pin_map(value, c):
    c.need(type(value) is dict and all(type(k) is str and _digest(v) for k, v in value.items()), 'Complete SHA256 map required')


def _inventory(value, c):
    c.equal(set(value), {'entries', 'files'}, 'Exact whole queue inventory required')
    entries = value['entries']; files = value['files']
    c.need(type(entries) is list and type(files) is dict and entries == sorted(entries), 'Sorted inventory required')
    names = set(); wanted = set()
    for row in entries:
        c.need(type(row) is list and len(row) == 2 and row[1] in ('file', 'directory'), 'Ordinary inventory entry required')
        name = row[0]; p = PurePosixPath(name)
        c.need(type(name) is str and name not in ('', '.') and not p.is_absolute() and '..' not in p.parts
               and str(p) == name and name not in names, 'Unique safe queue-relative entry required')
        names.add(name)
        if row[1] == 'file': wanted.add(name)
    c.equal(set(files), wanted, 'Whole file membership differs')
    for entry in files.values():
        c.equal(set(entry), {'sha256', 'bytes'}, 'Exact inventory file shape required')
        c.need(_digest(entry['sha256']) and type(entry['bytes']) is int and entry['bytes'] >= 0, 'File byte length/pin required')
    return hashlib.sha256((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()).hexdigest()


def _rollout(row, c):
    c.equal(row['horizon'], 314, 'Sand H314 rollout required'); c.equal(row['objective'], 'faithful', 'Faithful objective required')
    n = row['completed_steps']; complete = row['status'] == 'complete'
    c.need(type(n) is int and 0 <= n <= 314 and complete == (n == 314), 'Exact completed-prefix/horizon relation required')
    c.equal(row['requested_trace_steps'], [1, 10, 50, 200, 314], 'Sparse saved trace schedule differs')
    c.need(type(row['mse_per_step']) is list and len(row['mse_per_step']) == n
           and all(_finite(x) and x >= 0 for x in row['mse_per_step']), 'Complete accepted scalar-series prefix required')
    for key in ('mean_rollout_mse', 'mse_at_final_horizon'):
        c.need(_finite(row[key]) if complete else row[key] is None, 'Failed full-horizon scalar must remain null')
    c.equal(set(row['mse_at_declared_trace_steps']), {'1', '10', '50', '200', '314'}, 'Declared trace scalar grid differs')
    for step in (1, 10, 50, 200, 314):
        c.equal(row['mse_at_declared_trace_steps'][str(step)], row['mse_per_step'][step - 1] if step <= n else None, 'Pointwise forecast scalar differs')
    for side in ('predicted', 'ground_truth'):
        c.need(type(row[side + '_boundary_per_step']) is list and len(row[side + '_boundary_per_step']) == n, 'Full accepted boundary scalar series required')


def _row_values(row, key):
    parts = key.split('/')
    if parts[0] == 'accuracy': return (row.get('policies', {}).get(parts[1], {}).get('metrics') or {}).get(parts[2])
    if parts[0] == 'timing':
        policy = row.get('policies', {}).get(parts[1], {})
        return policy.get('timing', {}).get(parts[2], {}).get('mean') if policy.get('status') == 'complete' else None
    if parts[0] == 'correlations': return row.get('correlations', {}).get(parts[1], {}).get('value')
    value = row
    for part in parts:
        if not isinstance(value, dict): return None
        value = value.get(part)
    return value


def _diagnostic_supported_keys(row, stage):
    if stage == 'clean_validation': return sorted(diagnostic_keys(stage)) if row['status'] == 'complete' else []
    keys = []
    for policy, record in row.get('policies', {}).items():
        keys += [k for k in diagnostic_keys(stage) if k.startswith('timing/' + policy + '/')]
        if record['status'] == 'complete' and policy in DIAG_POLICIES:
            keys += [k for k in diagnostic_keys(stage) if k.startswith('accuracy/' + policy + '/')]
    for policy, values in row.get('benefit', {}).items(): keys += ['benefit/' + policy + '/' + key for key in values]
    keys += ['correlations/' + key for key in row.get('correlations', {})]
    return sorted(keys)


def _diagnostics(stage, c):
    value = stage['diagnostic_summary']; name = stage['stage']; rows = stage['rows']; units = schedule(name)
    c.need(type(value) is dict, 'Diagnostic summary required even for missing work')
    for k, v in [('expected_frames', len(units)), ('returned_frames', len(rows)),
                 ('complete_frames', sum(r['status'] == 'complete' for r in rows)), ('failed_frames', sum(r['status'] == 'failed' for r in rows))]:
        c.equal(value[k], v, 'Diagnostic frame denominator differs: ' + k)
    actual_keys = set()
    def visit(obj, path=()):
        if type(obj) is dict:
            if 'equal_trajectory_mean' in obj: actual_keys.add('/'.join(path))
            else:
                for k, v in obj.items(): visit(v, path + (k,))
    visit(value); c.equal(actual_keys, set(diagnostic_keys(name)), 'Complete fixed diagnostic metric grid required')
    lookup = {_unit(r, name): r for r in rows}
    for metric in diagnostic_keys(name):
        leaf = _leaf(value, metric); sources = sorted({s for s, _ in units})
        c.equal(leaf['expected_frames'], len(units), 'Diagnostic leaf frame denominator differs')
        c.equal(leaf['expected_trajectories'], len(sources), 'Diagnostic leaf source denominator differs')
        c.equal(set(leaf['trajectories']), {str(s) for s in sources}, 'All diagnostic sources required')
        defined_frames = 0; defined_sources = 0
        for source in sources:
            wanted = [u for u in units if u[0] == source]
            defined = sum(_finite(_row_values(lookup.get(u, {}), metric)) for u in wanted)
            item = leaf['trajectories'][str(source)]
            c.equal(item['expected'], len(wanted), 'Per-source denominator differs'); c.equal(item['defined'], defined, 'Defined frame count differs')
            c.need(_finite(item['mean']) if defined == len(wanted) else item['mean'] is None, 'No survivor mean within source')
            defined_frames += defined; defined_sources += defined == len(wanted)
        c.equal(leaf['defined_frames'], defined_frames, 'Defined diagnostic frame total differs')
        c.equal(leaf['defined_trajectories'], defined_sources, 'Defined diagnostic source total differs')
        c.need(_finite(leaf['equal_trajectory_mean']) if defined_sources == len(sources) else leaf['equal_trajectory_mean'] is None, 'No survivor mean across sources')


def _collection_base(product, c):
    for key, value in [('schema', COLLECTION_SCHEMA), ('status', 'stopped_outputs_collected'), ('issued_by', 'root'),
                       ('collector_sha256', COLLECTOR), ('cohort_sha256', COHORT), ('protocol_sha256', PROTOCOL),
                       ('source_sha256', SOURCES), ('operational_amendment_sha256', AMENDMENT), ('timing_scope', 'shared_host_operational_measurement')]:
        c.equal(product[key], value, 'Frozen collection identity differs: ' + key)
    role = product['host_role']; c.need(role in MODELS, 'Exact Sand host role required')
    c.equal(product['queue_root'], _queue(role), 'Original local queue root required')
    c.equal(product['original_queue_root'], _queue(role), 'Original queue identity differs')
    wanted = {(a, s, name) for a, s in MODELS[role] for name in STAGES}
    stages = product['stages']; c.need(type(stages) is list and len(stages) == len(wanted), 'All original role model stages required')
    grid = {(x['arm'], x['seed'], x['stage']): x for x in stages}; c.equal(set(grid), wanted, 'Exact original model-stage grid required')
    counts = {}
    for (arm, seed, name), stage in grid.items():
        c.need(type(seed) is int, 'Strict integer seed required')
        c.equal([stage['mode'], stage['split']], list(STAGES[name]), 'Stage mode/split differs')
        cells = stage['cells']; units = schedule(name)
        c.need(type(cells) is list and len(cells) == len(units), 'Complete fixed cell denominator required')
        c.equal([_unit(x, name) for x in cells], units, 'Exact Sand T320/H314 target grid required')
        for cell in cells:
            c.need(cell['state'] in STATES, 'Unknown cell state')
            c.need(type(cell['source_index']) is int and (name == 'full_rollout_test' or type(cell['target_frame']) is int), 'Strict integer cell identity required')
            if cell['state'] not in COMMITTED: c.equal(cell['missing'], True, 'Missing/timeout/unstarted cell must remain missing')
        committed = {_unit(x, name): x for x in cells if x['state'] in COMMITTED}
        rows = stage['rows']; c.need(type(rows) is list, 'Committed row list required')
        lookup = {_unit(x, name): x for x in rows}
        c.need(len(lookup) == len(rows) and set(lookup) == set(committed), 'Exactly all committed rows and no fabricated missing rows required')
        for unit, row in lookup.items():
            cell = committed[unit]
            c.equal(row['status'], 'complete' if cell['state'] == 'completed_required_outcome' else 'failed', 'Cell/row completion differs')
            c.equal(row.get('failure'), cell.get('failure'), 'Original failure identity differs')
            c.need(row.get('failure') is None if row['status'] == 'complete' else type(row.get('failure')) is dict and type(row['failure'].get('category')) is str, 'Committed failure category required')
            if name == 'full_rollout_test':
                c.equal([row['arm'], row['training_seed']], [arm, seed], 'Committed rollout model differs'); _rollout(row, c)
        snapshot = stage['aggregate_snapshot']; c.equal(set(snapshot), {'state', 'recorded_rows', 'committed_rows'}, 'Aggregate snapshot fields differ')
        c.equal(snapshot['committed_rows'], len(rows), 'Committed snapshot count differs')
        n = snapshot['recorded_rows']; c.need(type(n) is int and 0 <= n <= len(rows), 'Valid aggregate prefix count required')
        c.need((snapshot['state'] == 'absent' and n == 0) or (snapshot['state'] == 'current' and n == len(rows))
               or (snapshot['state'] == 'stale_valid_prefix' and n < len(rows)), 'Aggregate must remain an exact current/stale/absent prefix')
        if name == 'full_rollout_test': c.equal(stage['diagnostic_summary'], None, 'Rollout has no diagnostic summary')
        else: _diagnostics(stage, c)
        counts[arm, seed, name] = dict(Counter(x['state'] for x in cells))
    c.equal(sum(sum(x.values()) for x in counts.values()), 608 * len(MODELS[role]), 'Full role cell denominator required')
    return grid, counts


def _finish(product, c, counts):
    total = Counter()
    for row in counts.values(): total.update(row)
    return {'status': 'validated_product_metadata', 'schema': product['schema'], 'checks': c.count,
            'required_model_stages': len(counts), 'required_cells': sum(total.values()), 'coverage': dict(total),
            'all_required_cells_completed': set(total) == {'completed_required_outcome'},
            'scientific_admission': False, 'statistics_recomputed': False, 'arrays_opened': False}


@_guard
def validate_collection(product, artifact_sha256, expected):
    c = _begin(product, artifact_sha256, expected,
        ('artifact_sha256', 'host_role', 'root_release_sha256', 'root_release', 'queue_release_sha256', 'queue_release',
         'inventory', 'ledger', 'queue_status', 'process_outcomes', 'gpu_observations', 'protocols'),
        COLLECTION_SCHEMA, 'stopped_outputs_collected')
    grid, counts = _collection_base(product, c); role = product['host_role']; queue = _queue(role)
    c.equal(role, expected['host_role'], 'Actual operation role differs')
    for name in ('root_release_sha256', 'queue_release_sha256'): c.need(_digest(expected[name]), 'Exact release SHA required')
    c.equal(product['collection_release_sha256'], expected['root_release_sha256'], 'Actual collection release differs')
    c.equal(product['queue_release_sha256'], expected['queue_release_sha256'], 'Original queue release differs')
    c.equal(product['output_tree_state'], expected['inventory'], 'Original stopped snapshot differs')
    c.equal(product['files'], expected['inventory']['files'], 'Stopped file views differ')
    inventory_pin = _inventory(expected['inventory'], c)
    c.equal(product['local_queue_inventory_sha256'], inventory_pin, 'Whole queue inventory digest differs')
    root = expected['root_release']
    for k, v in [('schema', 'adaptgns_sand_scalar_collection_release_scoped_v1'), ('status', 'approved_for_stopped_scalar_collection'),
                 ('issued_by', 'root'), ('cohort_sha256', COHORT), ('collector_sha256', COLLECTOR), ('source_sha256', SOURCES),
                 ('original_queue_root', queue), ('local_queue_inventory_sha256', inventory_pin),
                 ('queue_release_sha256', expected['queue_release_sha256']), ('operational_amendment_sha256', AMENDMENT)]:
        c.equal(root[k], v, 'Issued collection root control differs: ' + k)
    input_pins = {**SOURCE_INPUTS, COHORT_PATH: COHORT, PREP + '/sand_scoped_operational_amendment_v1.md': AMENDMENT,
                  ANALYSIS + '/controls/' + role + '.collection_release.json': expected['root_release_sha256']}
    c.equal(product['input_sha256'], input_pins, 'Exact collection input/source binding differs')
    for key in ('queue_release', 'queue_status', 'process_outcomes'): c.equal(product[key], expected[key], 'Original stopped metadata differs: ' + key)
    ledger, status, process = expected['ledger'], expected['queue_status'], expected['process_outcomes']
    for obj in (ledger, status, process):
        c.equal(obj['schema'], Q_SCHEMA, 'Original stopped schema differs'); c.equal(obj['unreaped_owned_children'], [], 'Unreaped original child')
    c.equal(ledger['dataset'], 'Sand', 'Sand ledger required')
    c.need(status['state'] in ('allocation_finished', 'stopped_requires_review') and type(status['all_pinned_inputs_reverified']) is bool, 'Stopped queue state required')
    c.equal(status['coverage_ledger_sha256'], product['files']['coverage_ledger.json']['sha256'], 'Original ledger byte pin differs')
    c.equal(expected['queue_release_sha256'], product['files']['release_snapshot.json']['sha256'], 'Original release snapshot byte pin differs')
    c.equal(product['coverage_audit_errors'], ledger['audit_errors'], 'Retained coverage audit errors differ')
    c.equal(product['abort_reason'], ledger['abort_reason'], 'Retained abort reason differs')
    release = product['queue_release']
    for k, v in [('schema', 'adaptgns_sand_evaluation_gpu_scoped_release_v1'), ('issued_by', 'root'), ('status', 'admitted_for_execution_allocation'), ('dataset', 'Sand'), ('host_role', role)]:
        c.equal(release[k], v, 'Original issued queue release differs: ' + k)
    c.equal([(s['arm'], s['seed'], s['gpu']) for s in release['streams']], [(a, s, g) for (a, s), g in zip(MODELS[role], GPUS[role])], 'Original released model/GPU mapping differs')
    scope = {'owned_indices': list(GPUS[role]), 'unassigned_devices': 'observe_without_control', 'timing_scope': 'shared_host_operational_measurement', 'live_training_handoff': False}
    c.equal(release['gpu_scope'], scope, 'Original GPU scope differs'); c.equal(process['gpu_scope'], scope, 'Stopped process GPU scope differs')
    streams = {x['id']: x for x in process['streams']}
    c.need(len(streams) == len(process['streams']) and set(streams) == {a + '_seed' + str(s) for a, s in MODELS[role]}, 'Complete original process streams required')
    entries = {(x['stream'], x['stage']): x for x in ledger['stages']}
    c.need(len(entries) == len(ledger['stages']) == len(grid), 'Complete unique original stage ledger required')
    protocols = expected['protocols']; wanted_protocols = set()
    for (arm, seed, name), stage in grid.items():
        stream_id = arm + '_seed' + str(seed); entry = entries[stream_id, name]
        released = next(s for s in release['streams'] if s['id'] == stream_id)
        c.equal(len(released['commands']), 4, 'Four exact released stages required')
        command = released['commands'][list(STAGES).index(name)]
        c.equal(command[:3], [REMOTE + '/.venv/bin/python', PREP + '/evaluate_sand_graph_support_final.py', '--execute'], 'Original direct evaluator command differs')
        c.need(len(command[3:]) % 2 == 0 and all(type(x) is str for x in command), 'Literal evaluator options required')
        options = dict(zip(command[3::2], command[4::2]))
        c.need(len(options) == len(command[3:]) // 2, 'No duplicated evaluator option')
        rel = _rel(arm, seed, name)
        for key, value in [('--mode', stage['mode']), ('--split', stage['split']), ('--arm', arm), ('--seed', str(seed)),
                           ('--objective', 'faithful'), ('--cuda-index', str(released['gpu'])), ('--output-dir', queue + '/' + rel)]:
            c.equal(options[key], value, 'Original evaluator model/stage option differs')
        required_inputs = {'--cohort': COHORT, '--protocol': PROTOCOL, '--trainer-source': TRAINER,
                           '--benchmark-helper': BENCHMARK, '--checkpoint': options['--checkpoint-sha256']}
        c.need(_digest(options['--checkpoint-sha256']), 'Original checkpoint byte pin required')
        for key, value in required_inputs.items(): c.equal(release['files_sha256'][options[key]], value, 'Original model/scientific input pin differs')
        c.equal(release['files_sha256'][command[1]], SOURCES['evaluate_sand_graph_support_final.py'], 'Frozen evaluator source pin differs')
        c.equal(entry['coverage_audit_state'], 'row_coverage_checked', 'Original row coverage not checked')
        c.equal(stage['cells'], entry['cells'], 'Original full cell states differ'); c.equal(stage['outcome'], entry['outcome'], 'Original stage outcome differs')
        outcomes = streams[stream_id]['outcomes']; by_stage = {o['stage']: o for o in outcomes}
        c.need(len(outcomes) == len(by_stage) and set(by_stage) <= set(STAGES), 'Unique released process outcomes required')
        for out in outcomes: c.need(out['state'] in ('exited', 'quota_expired', 'aborted_by_supervisor') and type(out['exit_code']) is int, 'Reaped original evaluator outcome required')
        c.equal(stage['outcome'], by_stage.get(name, {'stage': name, 'state': 'never_started'}), 'Process/ledger outcome differs')
        pp = rel + '/protocol.json'
        if pp in product['files']:
            wanted_protocols.add(pp); protocol = protocols[pp]
            c.equal(stage['protocol_sha256'], product['files'][pp]['sha256'], 'Original stage protocol byte pin differs')
            for k, v in [('schema', 'adaptgns_sand_graph_support_final_evaluation_v1'), ('mode', stage['mode']), ('split', stage['split']), ('source_frame_count', 320)]:
                c.equal(protocol[k], v, 'Original Sand protocol differs: ' + k)
            for k, v in [('kind', 'preselected_checkpoint'), ('completed_updates', 100000), ('arm', arm), ('seed', seed)]: c.equal(protocol['model'][k], v, 'Original100k model identity differs')
            c.equal(protocol['model']['checkpoint_sha256'], options['--checkpoint-sha256'], 'Original protocol checkpoint differs')
            for key, value in required_inputs.items(): c.equal(protocol['input_files_sha256'][options[key]], value, 'Original protocol input pin differs')
            c.equal(protocol['policies'], list(POLICIES if name == 'full_rollout_test' else DIAG_POLICIES), 'Original policy grid differs')
            c.equal([x['source_index'] for x in protocol['schedule']] if name == 'full_rollout_test' else [_unit(x, name) for x in protocol['schedule']], list(range(30)) if name == 'full_rollout_test' else schedule(name), 'Original protocol target schedule differs')
        else: c.need(stage['protocol_sha256'] is None and not stage['rows'], 'Missing protocol cannot contain committed rows')
        for cell in stage['cells']:
            if cell['state'] not in COMMITTED: continue
            p = PurePosixPath(cell['path']); c.equal(str(p.parent), queue + '/' + rel, 'Committed row path escaped original stage')
            c.equal(product['files'][rel + '/' + p.name]['sha256'], cell['sha256'], 'Committed row inventory pin differs')
            row = next(r for r in stage['rows'] if _unit(r, name) == _unit(cell, name))
            c.equal(row['protocol_sha256'], stage['protocol_sha256'], 'Committed row protocol pin differs')
            key = 'trace' if name == 'full_rollout_test' else 'artifact'; name_file = row[key + '_file']
            c.need(type(name_file) is str and PurePosixPath(name_file).name == name_file and name_file not in ('', '.', '..'), 'Artifact must be a stage basename')
            c.equal(product['files'][rel + '/' + name_file]['sha256'], row[key + '_sha256'], 'Committed numeric artifact opaque pin differs')
    c.equal(set(protocols), wanted_protocols, 'Exactly all present original protocol documents required')
    if expected['gpu_observations'] is None:
        c.need('gpu_observations.json' not in product['files'] and process['all_children'] == [] and all(s['outcomes'] == [] for s in streams.values())
               and process['gpu_observations_sha256'] is None and type(process['abort_reason']) is str and bool(process['abort_reason']), 'GPU observation absence only before any child')
        observations = {'schema': Q_SCHEMA, 'observations': [], 'observation_file_state': 'absent_before_any_child'}
        file_state = 'absent_before_any_child'
    else:
        observations = expected['gpu_observations']; file_state = 'present'
        c.equal(process['gpu_observations_sha256'], product['files']['gpu_observations.json']['sha256'], 'Original GPU observation pin differs')
    c.equal(product['gpu_observations'], observations, 'Raw original GPU observations differ')
    c.equal(product['gpu_observation_file_state'], file_state, 'GPU observation absence/presence differs')
    c.equal(observations['schema'], Q_SCHEMA, 'GPU observation schema differs')
    c.need(all(x['validation'] in ('passed', 'pending') for x in observations['observations']), 'Retain pending GPU observations')
    c.equal(product['gpu_observation_counts'], {k: sum(x['validation'] == k for x in observations['observations']) for k in ('passed', 'pending')}, 'Original GPU observation counts differ')
    result = _finish(product, c, counts); result['eligible_for_aggregation'] = status['all_pinned_inputs_reverified']
    return result


def _seed_shape(value, c):
    c.equal(set(value), {'seed_values', 'required_seed_pairs', 'defined_seed_pairs', 'mean', 'sample_sd'}, 'Exact paired seed-summary keys required')
    c.equal(set(value['seed_values']), {'0', '1', '2'}, 'All three ordered seeds required')
    c.equal(value['required_seed_pairs'], 3, 'Three-seed denominator required')
    values = list(value['seed_values'].values()); c.need(all(v is None or _finite(v) for v in values), 'Finite or null seed values required')
    count = sum(_finite(v) for v in values); c.equal(value['defined_seed_pairs'], count, 'Defined seed-pair count differs')
    c.need((_finite(value['mean']) and _finite(value['sample_sd']) and value['sample_sd'] >= 0) if count == 3 else value['mean'] is None and value['sample_sd'] is None, 'No survivor mean or SD across seeds')


def _walk_seeds(value, c):
    if type(value) is dict:
        if 'seed_values' in value: _seed_shape(value, c)
        else:
            for item in value.values(): _walk_seeds(item, c)
    elif type(value) is list:
        for item in value: _walk_seeds(item, c)


def _paired_nulls(result, operands, c):
    _seed_shape(result, c)
    for seed in ('0', '1', '2'):
        defined = all(_finite(values[seed]) for values in operands)
        c.need(_finite(result['seed_values'][seed]) if defined else result['seed_values'][seed] is None,
               'Every paired contrast must preserve operand nulls')


def _all_collections(collections, c):
    c.equal(set(collections), {'A', 'B'}, 'Both original role collections required')
    grid = {}; counts = {}
    for role in ('A', 'B'):
        c.equal(collections[role]['host_role'], role, 'Predecessor role differs')
        c.equal(collections[role]['queue_status']['all_pinned_inputs_reverified'], True, 'All original inputs must be reverified for aggregation')
        g, n = _collection_base(collections[role], c); grid.update(g); counts.update(n)
    c.equal(len(grid), 24, 'All24 original stages required'); c.equal(sum(sum(x.values()) for x in counts.values()), 3648, 'All3648 required cells retained')
    return grid, counts


def _full_shape(full, grid, counts, c):
    total = Counter(); by_model = []; failures = []
    for (arm, seed, name), row in grid.items():
        if name != 'full_rollout_test': continue
        n = counts[arm, seed, name]; total.update(n)
        by_model.append({'arm': arm, 'seed': seed, 'counts': n, 'guard_or_execution_failure_categories': dict(Counter(x['failure'].get('category', 'unspecified') for x in row['cells'] if x['state'] == 'recorded_failed_outcome'))})
        failures += [{'arm': arm, 'seed': seed, 'source_index': r['source_index'], 'policy': r['policy'], 'completed_steps': r['completed_steps'], 'failure': r['failure']} for r in row['rows'] if r['status'] == 'failed']
    for k, v in [('required_outcomes', 1080), ('coverage', dict(total)), ('all_required_outcomes_recorded', total['completed_required_outcome'] + total['recorded_failed_outcome'] == 1080), ('all_required_outcomes_complete', total['completed_required_outcome'] == 1080)]: c.equal(full[k], v, 'Full rollout denominator/state differs: ' + k)
    c.equal(full['coverage_by_model'], by_model, 'All original rollout model coverage/failure categories required')
    actual = full['failed_accepted_prefixes']; c.equal(len(actual), len(failures), 'Every failed accepted prefix must remain represented')
    for row, expected in zip(actual, failures):
        for k, v in expected.items(): c.equal(row[k], v, 'Failed prefix identity differs')
        c.equal(set(row['accepted_prefix_boundary']), set(FULL_METRICS[3:]), 'Failed prefix boundary grid differs')
    for k in ('absolute', 'within_arm_policy_contrasts', 'mix_minus_base', 'risk_minus_random_mix_minus_base_interaction'):
        c.equal(set(full[k]), set(FULL_METRICS), 'Complete H200/H314/boundary metric grid required')
    contrasts = {p + '_minus_' + r for r in ('base', 'random25') for p in POLICIES if p != r}
    for metric in FULL_METRICS:
        c.equal(set(full['absolute'][metric]), {'base', 'mix'}, 'Both training arms required')
        c.equal(set(full['within_arm_policy_contrasts'][metric]), {'base', 'mix'}, 'Both contrast arms required')
        c.equal(set(full['mix_minus_base'][metric]), set(POLICIES), 'All paired policies required')
        for arm in ('base', 'mix'):
            c.equal(set(full['absolute'][metric][arm]), set(POLICIES), 'All absolute policies required')
            c.equal(set(full['within_arm_policy_contrasts'][metric][arm]), contrasts, 'All fixed policy contrasts required')
            for policy in POLICIES:
                values = full['absolute'][metric][arm][policy]['seed_values']
                for seed in range(3):
                    rows = {r['source_index']: r for r in grid[arm, seed, 'full_rollout_test']['rows'] if r['policy'] == policy}
                    complete = len(rows) == 30 and all(r['completed_steps'] >= (200 if metric == 'mse_forecast200' else 314) for r in rows.values())
                    c.need(_finite(values[str(seed)]) if complete else values[str(seed)] is None, 'No survivor model/policy mean or shortened horizon substitution')
        absolute = full['absolute'][metric]
        for policy in POLICIES:
            _paired_nulls(full['mix_minus_base'][metric][policy], [absolute[a][policy]['seed_values'] for a in ('mix', 'base')], c)
        for arm in ('base', 'mix'):
            for ref in ('base', 'random25'):
                for policy in POLICIES:
                    if policy != ref:
                        _paired_nulls(full['within_arm_policy_contrasts'][metric][arm][policy + '_minus_' + ref],
                                      [absolute[arm][p]['seed_values'] for p in (policy, ref)], c)
        _paired_nulls(full['risk_minus_random_mix_minus_base_interaction'][metric],
                      [absolute[a][p]['seed_values'] for a in ('mix', 'base') for p in ('laggedrisk25', 'random25')], c)
    _walk_seeds(full, c)


def _summary_shape(product, collections, grid, counts, c):
    _full_shape(product['full_rollout'], grid, counts, c)
    c.equal(set(product['diagnostics']), set(STAGES) - {'full_rollout_test'}, 'All three diagnostic stages required')
    for name in list(STAGES)[1:]:
        result = product['diagnostics'][name]; selected = {(a, s): row for (a, s, n), row in grid.items() if n == name}
        c.equal(result['coverage'], {a + '_seed' + str(s): counts[a, s, name] for a, s in selected}, 'Every diagnostic cell state retained')
        keys = set(diagnostic_keys(name))
        c.equal(set(result['absolute']), {'base', 'mix'}, 'Both diagnostic arms required'); c.equal(set(result['mix_minus_base']), keys, 'Exact diagnostic contrast metrics required')
        for arm in ('base', 'mix'):
            c.equal(set(result['absolute'][arm]), keys, 'Complete diagnostic metric grid required')
            for key in keys:
                c.equal(result['absolute'][arm][key]['seed_values'], {str(seed): _leaf(selected[arm, seed]['diagnostic_summary'], key)['equal_trajectory_mean'] for seed in range(3)}, 'Diagnostic seed values differ from accepted collections')
        for key in keys:
            _paired_nulls(result['mix_minus_base'][key], [result['absolute'][arm][key]['seed_values'] for arm in ('mix', 'base')], c)
        if 'model_summaries' in result:
            c.equal(result['model_summaries'], {a + '_seed' + str(s): row['diagnostic_summary'] for (a, s), row in selected.items()}, 'Diagnostic model summaries differ')
        if name != 'clean_validation':
            c.equal(set(result['previous_observed_risk_minus_random_position_mse']), {'base', 'mix'}, 'Both observed-risk contrasts required')
            risk = 'accuracy/previous-observed-base-risk25/position_coordinate_mse'; random = 'accuracy/random25/position_coordinate_mse'
            for arm in ('base', 'mix'):
                _paired_nulls(result['previous_observed_risk_minus_random_position_mse'][arm],
                              [result['absolute'][arm][key]['seed_values'] for key in (risk, random)], c)
            _paired_nulls(result['risk_minus_random_mix_minus_base_interaction'],
                          [result['absolute'][arm][key]['seed_values'] for arm in ('mix', 'base') for key in (risk, random)], c)
        _walk_seeds(result, c)


@_guard
def validate_summary(product, artifact_sha256, expected, collections_by_role):
    c = _begin(product, artifact_sha256, expected, ('artifact_sha256', 'root_release_sha256', 'collection_sha256'), SUMMARY_SCHEMA, 'fixed_scalar_aggregation_complete')
    grid, counts = _all_collections(collections_by_role, c)
    c.need(_digest(expected['root_release_sha256']), 'Issued summary release SHA required')
    c.equal(product['analysis_release_sha256'], expected['root_release_sha256'], 'Issued summary release differs')
    c.equal(set(expected['collection_sha256']), {'A', 'B'}, 'Both collection byte pins required'); _pin_map(expected['collection_sha256'], c)
    for k, v in [('cohort_sha256', COHORT), ('protocol_sha256', PROTOCOL), ('summarizer_sha256', COLLECTOR), ('source_sha256', SOURCES), ('collection_sha256', expected['collection_sha256'])]: c.equal(product[k], v, 'Summary lineage differs: ' + k)
    c.equal(product['input_sha256'], {**SOURCE_INPUTS, ANALYSIS + '/controls/summary_release.json': expected['root_release_sha256'], **{ANALYSIS + '/' + r + '.stopped_collection.json': h for r, h in expected['collection_sha256'].items()}}, 'Exact summary input map required')
    fields = ('queue_status', 'queue_release_sha256', 'operational_amendment_sha256', 'abort_reason', 'coverage_audit_errors', 'gpu_observations', 'gpu_observation_counts', 'gpu_observation_file_state', 'process_outcomes')
    c.equal(product['queue_accounting'], {r: {k: col[k] for k in fields} for r, col in collections_by_role.items()}, 'Original full run accounting differs')
    _summary_shape(product, collections_by_role, grid, counts, c)
    c.equal(product['scope'], 'all three training seeds; null when required values missing/failed; no survivor means, p-values, replacement seeds or shortened-horizon substitution', 'Frozen summary limitations differ')
    return _finish(product, c, counts)


@_guard
def validate_saved_audit(product, artifact_sha256, expected, collection):
    c = _begin(product, artifact_sha256, expected, ('artifact_sha256', 'collection_sha256'), 'sand_saved_array_audit_v1', 'passed_supported_checks')
    grid, counts = _collection_base(collection, c); role = collection['host_role']
    c.equal(collection['queue_status']['all_pinned_inputs_reverified'], True, 'Original inputs must be reverified')
    c.need(_digest(expected['collection_sha256']), 'Accepted collection SHA required')
    for k, v in [('host_role', role), ('collection_sha256', expected['collection_sha256']), ('cohort_sha256', COHORT), ('audit_revision', 2), ('auditor_sha256', SAVED), ('diagnostic_helper_sha256', DIAGNOSTIC), ('audit_source_sha256', SAVED_SOURCES), ('scope', SAVED_SCOPE)]: c.equal(product[k], v, 'Saved-audit lineage/scope differs: ' + k)
    c.need(type(product['checks']) is int and product['checks'] > 0, 'Positive supported-check count required')
    models = {(m['arm'], m['seed'], m['stage']): m for m in product['models']}
    c.need(len(models) == len(product['models']) and set(models) == set(grid), 'All original saved-audit model stages required')
    expected_rows = {}; verified = {}
    for key, model in models.items():
        arm, seed, name = key; stage = grid[key]
        c.equal(model['cells'], stage['cells'], 'Saved-audit original cell states differ'); c.equal(model['coverage'], counts[key], 'Saved-audit denominator differs')
        metric_keys = {m + '/' + p for m in FULL_METRICS for p in POLICIES} if name == 'full_rollout_test' else set(diagnostic_keys(name))
        c.equal(set(model['metrics']), metric_keys, 'Complete supported audit metric grid required')
        if name != 'full_rollout_test':
            for metric in metric_keys:
                want = _leaf(stage['diagnostic_summary'], metric)['equal_trajectory_mean']; actual = model['metrics'][metric]
                c.need((want is None and actual is None) or (_finite(actual) and _finite(want) and math.isclose(actual, want, rel_tol=1e-10, abs_tol=1e-12)), 'Audited diagnostic hierarchy differs')
        else:
            for metric in FULL_METRICS:
                for policy in POLICIES:
                    rows = [r for r in stage['rows'] if r['policy'] == policy]
                    defined = len(rows) == 30 and all(r['completed_steps'] >= (200 if metric == 'mse_forecast200' else 314) for r in rows)
                    c.need(_finite(model['metrics'][metric + '/' + policy]) if defined else model['metrics'][metric + '/' + policy] is None, 'Audited model metrics cannot use survivor means')
        rel = _rel(arm, seed, name)
        if stage['rows']: verified[_queue(role) + '/' + rel + '/protocol.json'] = collection['files'][rel + '/protocol.json']['sha256']
        for cell in stage['cells']:
            if cell['state'] not in COMMITTED: continue
            row = next(x for x in stage['rows'] if _unit(x, name) == _unit(cell, name)); expected_rows[(*key, *_unit(row, name))] = row
            row_name = PurePosixPath(cell['path']).name; verified[_queue(role) + '/' + rel + '/' + row_name] = cell['sha256']
            kind = 'trace' if name == 'full_rollout_test' else 'artifact'; verified[_queue(role) + '/' + rel + '/' + row[kind + '_file']] = row[kind + '_sha256']
    actual_rows = {tuple([x['arm'], x['seed'], x['stage'], *x['unit']]): x for x in product['row_checks']}
    c.need(len(actual_rows) == len(product['row_checks']) and set(actual_rows) == set(expected_rows), 'Exactly every committed row audit required')
    for key, row in actual_rows.items():
        original = expected_rows[key]
        c.equal(row['status'], original['status'], 'Audited row completion differs'); c.equal(row['failure'], original.get('failure'), 'Audited row failure differs')
        if key[2] == 'full_rollout_test':
            n = original['completed_steps']; steps = [x for x in (1, 10, 50, 200, 314) if x <= n]
            c.equal(row['completed_steps'], n, 'Audited accepted prefix differs'); c.equal(row['array_recomputed_mse_forecasts'], steps, 'Only exact saved forecasts may be array-checked')
            boundaries = sorted({s for t in steps for s in [t] + list(range(t - 6, t)) if s > 0})
            c.equal(row['array_recomputed_prediction_boundary_forecasts'], boundaries, 'Saved history boundary support differs')
            c.equal(row['full_and_prefix_MSE_verification'], 'aggregate arithmetic over complete recorded scalar series; unsaved prediction errors not array-recomputed', 'Sparse-save limitation must remain explicit')
            c.equal(set(row['truth_hashes']), {str(x) for x in steps}, 'Only saved truth steps may be claimed'); _pin_map(row['truth_hashes'], c)
            c.need(_digest(row['initial_history_sha256']), 'Saved initial history hash required')
            c.need(row['accepted_prefix_boundary'] is None if n == 314 else type(row['accepted_prefix_boundary']) is dict and set(row['accepted_prefix_boundary']) == set(FULL_METRICS[3:]), 'Failed accepted boundary prefix required')
        else: c.equal(row['array_recomputed_diagnostic_metrics'], _diagnostic_supported_keys(original, key[2]), 'Available diagnostic metric support differs')
    c.equal(product['verified_input_sha256'], verified, 'Exact files actually read by saved audit required')
    _pin_map(product['shared_saved_state_hashes'], c)
    return _finish(product, c, counts)


def _supported_equal(actual, expected, c, path='summary'):
    extensible = {'summary', 'summary/full_rollout'} | {'summary/diagnostics/' + name for name in list(STAGES)[1:]}
    if type(expected) is dict:
        c.need(type(actual) is dict, 'Paired supported object required')
        if path not in extensible: c.equal(set(actual), set(expected), 'Paired supported keys differ: ' + path)
        for key, value in expected.items(): _supported_equal(actual[key], value, c, path + '/' + key)
    elif type(expected) is list:
        c.need(type(actual) is list and len(actual) == len(expected), 'Paired supported list differs')
        for a, b in zip(actual, expected): _supported_equal(a, b, c, path)
    elif _finite(expected) and type(expected) is float:
        c.need(_finite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), 'Published scalar differs from accepted paired audit')
    else: c.equal(actual, expected, 'Paired supported scalar/state differs: ' + path)


@_guard
def validate_paired_audit(product, artifact_sha256, expected, collections_by_role, summary, audits_by_role):
    c = _begin(product, artifact_sha256, expected, ('artifact_sha256', 'collection_sha256', 'audit_sha256', 'summary_sha256'), 'sand_paired_saved_array_audit_v1', 'passed_supported_checks')
    grid, counts = _all_collections(collections_by_role, c)
    c.equal(set(audits_by_role), {'A', 'B'}, 'Both original saved audits required')
    for key in ('collection_sha256', 'audit_sha256'):
        c.equal(set(expected[key]), {'A', 'B'}, 'Both predecessor role hashes required'); _pin_map(expected[key], c)
    c.need(_digest(expected['summary_sha256']), 'Actual summary byte pin required')
    for key, value in [('audit_revision', 2), ('cohort_sha256', COHORT), ('summary_sha256', expected['summary_sha256']), ('scope', PAIRED_SCOPE)]: c.equal(product[key], value, 'Paired-audit lineage/scope differs')
    c.need(type(product['checks']) is int and product['checks'] > 0, 'Positive paired check count required')
    c.equal(summary['schema'], SUMMARY_SCHEMA, 'Accepted scalar summary required'); c.equal(summary['status'], 'fixed_scalar_aggregation_complete', 'Accepted summary status required')
    for key, value in [('summarizer_sha256', COLLECTOR), ('source_sha256', SOURCES), ('protocol_sha256', PROTOCOL), ('cohort_sha256', COHORT)]:
        c.equal(summary[key], value, 'Accepted summary source/cohort differs')
    c.equal(summary['collection_sha256'], expected['collection_sha256'], 'Summary/collection lineage differs')
    shared = {}
    for role, audit in audits_by_role.items():
        validate_saved_audit(audit, expected['audit_sha256'][role], {'artifact_sha256': expected['audit_sha256'][role], 'collection_sha256': expected['collection_sha256'][role]}, collections_by_role[role])
        for key, value in audit['shared_saved_state_hashes'].items():
            if key in shared: c.equal(shared[key], value, 'Cross-host saved state/target identity differs')
            shared[key] = value
    c.equal(product['verified_input_sha256'], {**PAIRED_SOURCES, **{ANALYSIS + '/' + role + '.saved_array_audit.json': h for role, h in expected['audit_sha256'].items()}, ANALYSIS + '/paired_scalar_summary.json': expected['summary_sha256']}, 'Exact paired predecessor/source map required')
    verified = product['verified_paired_scalars']; c.equal(set(verified), {'full_rollout', 'diagnostics'}, 'All paired supported containers required')
    _summary_shape(verified, collections_by_role, grid, counts, c); _supported_equal(summary, verified, c)
    return _finish(product, c, counts)
