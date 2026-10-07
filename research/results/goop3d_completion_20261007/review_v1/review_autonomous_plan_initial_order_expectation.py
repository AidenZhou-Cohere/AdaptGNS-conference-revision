"""Independent metadata-only partition review; no array or model reads."""
import collections
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PLAN = BASE / 'autonomous_v1/plan.json'
checks = []


def check(value, description):
    if not value:
        raise AssertionError(description)
    checks.append(description)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(arm, seed, split, cell):
    return arm, seed, split, cell['source_index'], cell['policy']


plan = json.loads(PLAN.read_text())
ledger_path = Path(plan['original_local_ledger_path'])
ledger = json.loads(ledger_path.read_text())
check(sha(ledger_path) == plan['original_ledger_sha256'], 'exact original ledger bytes')
check((plan['endpoint_updates'], plan['frames'], plan['horizon'], plan['trace_steps']) ==
      (25000, 301, 295, [1, 10, 50, 200, 295]), 'original endpoint and full horizon/trace grid')
check(plan['policies'] == ['base', 'dense', 'random25', 'speed25',
                          'relative-velocity-RMS25', 'laggedrisk25'], 'all six original policies')
check(plan['rng_seed_expression'] == '93000+1000*seed+source_index' and
      plan['rng_seed_has_arm_split_policy_worker_terms'] is False,
      'original autonomous per-cell RNG expression')
check(plan['scientific_arrays_or_models_opened_to_prepare_plan'] is False,
      'metadata-only planning declared')

original, retained, missing = {}, {}, {}
for stage in ledger['stages']:
    if stage['mode'] != 'full-rollout':
        continue
    for cell in stage['cells']:
        key = identity(stage['arm'], stage['seed'], stage['split'], cell)
        check(key not in original, 'unique original autonomous cell ' + str(key))
        original[key] = cell
        target = retained if cell['state'] in ('completed_required_outcome', 'recorded_failed_outcome') else missing
        target[key] = cell
check(len(original) == plan['full_autonomous_cells'] == 2160, 'fixed 2160 autonomous denominator')
check(len(retained) == plan['retained_original_cells'] == 331, 'all 331 committed original cells retained')
check(len(missing) == plan['missing_cells'] == 1829, 'exact 1829 missing cells')
check(collections.Counter(c['state'] for c in missing.values()) ==
      {'not_completed_before_invocation_end': 1817, 'timed_out_current': 12},
      'all quota/missing outcome states preserved')

retained_plan = {}
for cell in plan['retained_original_outcomes']:
    key = identity(cell['arm'], cell['seed'], cell['split'], cell)
    check(key not in retained_plan, 'unique retained plan cell ' + str(key))
    retained_plan[key] = cell
    for field, value in retained[key].items():
        check(cell[field] == value, 'exact retained original field ' + str(key) + '/' + field)
check(set(retained_plan) == set(retained), 'retained plan equals full committed original set')

assigned = {}
check(len(plan['workers']) == 12, '12 metadata-only worker shards')
check({w['worker_index'] for w in plan['workers']} == set(range(12)), 'unique full worker indices')
check(collections.Counter((w['arm'], w['seed']) for w in plan['workers']) ==
      {(a, s): 2 for a in ('base', 'mix') for s in range(3)}, 'exact two shards per existing model')
for worker in plan['workers']:
    check(worker['shard'] in (0, 1), 'two valid shard identifiers')
    weight = 0
    for cell in worker['cells']:
        key = identity(cell['arm'], cell['seed'], cell['split'], cell)
        check((cell['arm'], cell['seed']) == (worker['arm'], worker['seed']), 'shard retains model identity ' + str(key))
        check(key not in assigned and key in missing and key not in retained,
              'assigned once and only missing original cell ' + str(key))
        assigned[key] = cell
        for field, value in missing[key].items():
            check(cell[field] == value, 'exact missing original field ' + str(key) + '/' + field)
        weight += cell['particles'] ** 2
    check(weight == worker['particle_squared_balance_weight'], 'metadata particle-squared shard weight')
check(set(assigned) == set(missing), 'no omitted or extra missing cells')
check(not (set(assigned) & set(retained_plan)), 'retained/new work disjoint')

for split, info in plan['splits'].items():
    check(split in ('valid', 'test') and info['record_count'] == 100, 'fixed N100 original population')
    grid = info['source_index_grid']
    check(len(grid) == len(set(grid)) == 30, 'fixed 30-source grid ' + split)
    expected = {(a, s, split, source, policy) for a in ('base', 'mix') for s in range(3)
                for source in grid for policy in plan['policies']}
    check(expected == {k for k in original if k[2] == split}, 'full six-model six-policy source product ' + split)
check(len(plan['files']) == len({x['path'] for x in plan['files']}) == 208, 'unique 208-file transfer inventory')
check(len(plan['models']) == 6, 'six fixed model records')
for model in plan['models']:
    stages = [s for s in ledger['stages'] if (s['arm'], s['seed']) == (model['arm'], model['seed'])]
    check(len(stages) == 5 and all(s['checkpoint_sha256'] == model['checkpoint_sha256'] for s in stages),
          'unchanged existing model checkpoint ' + str((model['arm'], model['seed'])))

report = {'status': 'passed_metadata_partition_review', 'plan_sha256': sha(PLAN),
          'original_ledger_sha256': sha(ledger_path), 'checks': len(checks),
          'full_autonomous_cells': 2160, 'retained_original_cells': 331, 'missing_cells': 1829,
          'metadata_only': True, 'scientific_arrays_or_models_read': False,
          'worker_source_or_live_runtime_review': False,
          'check_descriptions_sha256': hashlib.sha256(json.dumps(checks).encode()).hexdigest()}
out = BASE / 'review_v1/autonomous_plan_review.json'
out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps(report, sort_keys=True))
