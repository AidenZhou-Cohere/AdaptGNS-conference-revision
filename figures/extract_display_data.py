"""Copy every declared observed-test policy statistic from a pinned public map."""
import argparse
import hashlib
import json
from pathlib import Path

MAP_SHA = '6b51a4de77719f5036e32ae79bddd280876deed0d648084b198b48f3e15a00b5'
DISPLAY_SHA = 'a054ab4d130f8e04e9013d57f1cb7f5ecff68108f397a06f8f63656e7f1c008f'
POLICIES = ('base', 'random25', 'speed25', 'previous-observed-base-risk25', 'relative-velocity-RMS25', 'dense')


def extract(source):
    raw = Path(source).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == MAP_SHA
    data = json.loads(raw)
    rows = []
    for material in ('Goop', 'WaterDrop', 'Sand'):
        for arm in ('base', 'mix'):
            for policy in POLICIES:
                if material == 'WaterDrop' and policy == 'relative-velocity-RMS25':
                    continue
                key = (f'populations/observed_test/metrics/position_coordinate_mse/absolute/{arm}/{policy}'
                       if material == 'WaterDrop' else f'diagnostics/same_state_test/absolute/{arm}/accuracy/{policy}/position_coordinate_mse')
                statistic = data[material][key]
                values = statistic['seed_values']
                values = [values[str(i)] for i in range(3)] if isinstance(values, dict) else values
                rows.append({'material': material, 'training_arm': arm, 'policy': policy, 'source_key': key,
                             'statistic_unchanged': statistic, 'ordered_seed_values': values,
                             'nominal_optional_budget_fraction': 0 if policy == 'base' else 1 if policy == 'dense' else .25,
                             'budget_class': 'zero' if policy == 'base' else 'all' if policy == 'dense' else 'at_most_quarter'})
    context = []
    for arm in ('base', 'mix'):
        for policy in POLICIES:
            if policy == 'relative-velocity-RMS25':
                continue
            key = f'populations/observed_test/metrics/directed_edges/absolute/{arm}/{policy}'
            context.append({'training_arm': arm, 'policy': policy, 'source_key': key, 'statistic_unchanged': data['WaterDrop'][key]})
    result = {'schema': 'observed_accuracy_optional_budget_display_v1', 'source_scalar_map_sha256': MAP_SHA,
              'source_scope': 'Already-published cross-material scalar map; observed test diagnostic only.',
              'metric': 'position-coordinate MSE for one-step prediction',
              'aggregation': 'Histories average within each trajectory, trajectories equally within each model seed; saved mean and sample SD span all three seeds.',
              'budget_definition': 'Native keeps original directed edges; dense appends both orientations of all annulus pairs; sparse appends both orientations of floor(0.25*number_of_annulus_pairs). The fraction is an upper bound on added directed messages relative to dense additions on that same state, not total-edge fraction or measured runtime.',
              'axis': 'Categorical budget classes; separate policy columns inside each class, not a continuous empirical edge-count axis.',
              'materials': {'Goop': {'sources': 30, 'histories_per_source': 5, 'histories_per_seed': 150, 'updates': '100k from scratch', 'policies': 6},
                            'WaterDrop': {'sources': 27, 'histories_per_source': 11, 'histories_per_seed': 297, 'updates': '100k to 110k paired', 'policies': 5},
                            'Sand': {'sources': 30, 'histories_per_source': 5, 'histories_per_seed': 150, 'updates': '100k from scratch', 'policies': 6}},
              'rows': rows, 'waterdrop_total_edges_context': context,
              'undefined_seed_values': sum(value is None for row in rows for value in row['ordered_seed_values']),
              'rows_required': 34, 'seed_values_required': 102, 'autonomous_data_included': False,
              'waterdrop_rms_policy': 'not declared; not plotted as a missing or failed outcome'}
    assert len(rows) == 34 and sum(len(row['ordered_seed_values']) for row in rows) == 102
    encoded = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
    assert hashlib.sha256(encoded).hexdigest() == DISPLAY_SHA
    return encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument('--source', type=Path, default=root / 'results/cross_material/all_scalar_statistic_map.json')
    parser.add_argument('--output', type=Path, default=root / 'results/cross_material/budget_display_data.json')
    args = parser.parse_args()
    raw = extract(args.source)
    args.output.write_bytes(raw)
    print(json.dumps({'display_sha256': DISPLAY_SHA, 'policy_arm_rows': 34, 'seed_values': 102}))


if __name__ == '__main__':
    main()
