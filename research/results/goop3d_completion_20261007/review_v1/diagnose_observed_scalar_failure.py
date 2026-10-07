"""Path-aware diagnosis from committed scalar cache only; no saved arrays or inference."""
import collections
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import sys

BASE = Path(__file__).resolve().parents[1]
CACHE = BASE/'root/observed_failure_collection_v1/files/observed_results_v1'
FROZEN = BASE.parent/'deadline_research_20261005/cuda_preparation/goop3d_observed_history_analysis_UNADMITTED_transition_v1'
OUT = BASE/'review_v1/observed_arithmetic_diagnosis_v1'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(FROZEN))
import goop3d_observed_history_arithmetic_v1 as A
import summarize_goop3d_observed_histories_v1 as S
import check_goop3d_observed_history_summary_v1 as V


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return digest(path.read_bytes())


index_path = CACHE/'checkpoint_index.json'
index = json.loads(index_path.read_bytes())
assert len(index['completed']) == 2568 and not index['failed']
identity_pin = digest(encode(index['identity']))
assert index['identity']['source_pins_sha256'] == '365c8071abe5016a9092813d57177705386f6568efe03ce1893b640d4b8e7c92'
assert index['identity']['protocol_sha256'] == 'd2c2815cfee9b1f0eb4aff854eb4acb6c00b0ec0b6f232e308bde9487078f167'
for name,pin in index['identity']['frozen_files_sha256'].items():
    assert sha(FROZEN/name) == pin
groups = collections.defaultdict(dict)
pins = {}
for key,entry in index['completed'].items():
    path = CACHE/'rows'/(key+'.json'); raw = path.read_bytes()
    assert digest(raw) == entry['sha256']; pins[key] = entry['sha256']
    row = json.loads(raw)
    assert row['key'] == key and row['task_sha256'] == entry['task_sha256'] and row['identity_sha256'] == identity_pin
    match = re.fullmatch(r'(base|mix)_seed([012])__(clean_validation|same_state_valid|same_state_test)__(\d{6})_(\d{3})',key)
    assert match
    arm,seed,stage,source,target = match.groups(); unit = (int(source),int(target))
    detail = row['result']['detail']
    assert detail['unit'] == list(unit) and detail['status'] == 'complete' and detail['failure'] is None
    group = groups[arm,int(seed),stage]; assert unit not in group; group[unit] = detail['metrics']
assert set(groups) == {(a,s,n) for a in ('base','mix') for s in range(3) for n in S.OBSERVED}
aggregates = {}
for (arm,seed,stage),rows in groups.items():
    mode = 'clean-validation' if stage == 'clean_validation' else 'same-state'
    assert len(rows) == (128 if mode == 'clean-validation' else 150)
    expected = sorted(rows)
    if mode == 'same-state':
        assert {source for source,_ in expected} == {j*99//29 for j in range(30)}
        assert {target for _,target in expected} == {7,80,153,226,300}
    aggregates[arm,seed,stage] = {
        metric:A.aggregate(expected,{unit:row.get(metric) for unit,row in rows.items()})['equal_trajectory_mean']
        for metric in A.diagnostic_keys(mode)}

differences = []
compared = 0


def oracle(values):
    if not all(V.valid(x) for x in values):
        return None
    exact = [Fraction(x) for x in values]; mean = sum(exact)/3
    variance = sum((x-mean)**2 for x in exact)/2
    pairwise = sum((exact[i]-exact[j])**2 for i in range(3) for j in range(i+1,3))/6
    assert pairwise == variance
    with localcontext() as context:
        context.prec = 100
        sd = (Decimal(variance.numerator)/Decimal(variance.denominator)).sqrt()
    rounded = V.average(values)
    return {'exact_input_mean_ratio':str(mean),'exact_unbiased_variance_ratio':str(variance),
            'pairwise_identity_variance_ratio':str(pairwise),'sample_sd_decimal100':str(sd),
            'sample_sd_float':float(sd),'all_three_input_floats_identical':values[0]==values[1]==values[2],
            'rounded_mean_minus_input_values':[rounded-x for x in values],
            'input_float_hex':[float(x).hex() for x in values]}


def compare(path, values):
    global compared
    actual = S.stats(values); expected = V.seed_values(values)
    for leaf in ('required_seed_pairs','defined_seed_pairs','mean','sample_sd'):
        x,y = actual[leaf],expected[leaf]; compared += 1
        equal = x==y if x is None or y is None else V.valid(x) and math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-12)
        if not equal:
            differences.append({'path_components':path+[leaf], 'seed_values':values,
                                'summarizer_value':x,'original_checker_value':y,
                                'original_relative_tolerance':1e-10,'original_absolute_tolerance':1e-12,
                                'absolute_difference':None if x is None or y is None else abs(x-y),
                                'exact_rational_oracle':oracle(values)})
    assert actual['seed_values'] == expected['seed_values']


for stage in S.OBSERVED:
    keys = sorted(aggregates['base',0,stage])
    assert all(set(aggregates[a,s,stage])==set(keys) for a in ('base','mix') for s in range(3))
    def value(a,s,key): return aggregates[a,s,stage][key]
    for arm in ('base','mix'):
        for key in keys:
            compare(['families',stage,'absolute',arm,key],[value(arm,s,key) for s in range(3)])
    for key in keys:
        compare(['families',stage,'mix_minus_base_training',key],[S.delta(value('mix',s,key),value('base',s,key)) for s in range(3)])
    if stage.startswith('same_state'):
        for metric in ('position_coordinate_mse','normalized_coordinate_mse'):
            def loss(arm,seed,policy): return value(arm,seed,'accuracy/'+policy+'/'+metric)
            for arm in ('base','mix'):
                for ref in ('base','random25','speed25','relative-velocity-RMS25'):
                    for policy in S.POLICIES:
                        if policy != ref:
                            compare(['families',stage,'accuracy_policy_contrasts',metric,'within_arm',arm,policy+'_minus_'+ref],
                                    [S.delta(loss(arm,s,policy),loss(arm,s,ref)) for s in range(3)])
            compare(['families',stage,'accuracy_policy_contrasts',metric,'risk_minus_random_mix_minus_base'],
                    [S.delta(S.delta(loss('mix',s,S.RISK),loss('mix',s,'random25')),S.delta(loss('base',s,S.RISK),loss('base',s,'random25'))) for s in range(3)])

assert sha(index_path) == digest(encode(index))
report = {'status':'diagnosis_only_not_scientific_admission',
          'checkpoint_index_sha256':sha(index_path),'cache_identity_sha256':identity_pin,
          'cached_rows_verified':2568,'rows_or_arrays_reaudited':0,'scalar_statistic_leaves_compared':compared,
          'mismatch_count':len(differences),'mismatches':differences,
          'frozen_sources_sha256':index['identity']['frozen_files_sha256'],
          'method':'Reconstruct unchanged equal-source means from hash-verified cached scalar results, compare both frozen seed-statistic functions using their original tolerances, then exact Fraction variance and Decimal100 square-root diagnosis.',
          'cache_rows_sha256':pins,'original_failure_preserved':True,'tolerances_changed':False,
          'inference_or_source_array_reads':False,'scalar_reconstruction_not_old_failed_result_publication':True}
path = OUT/'diagnosis.json';path.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps({'report_sha256':sha(path),'compared':compared,'mismatch_count':len(differences),
                  'first_mismatches':differences[:5]},sort_keys=True))
