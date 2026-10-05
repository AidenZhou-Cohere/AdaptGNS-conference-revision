"""Render concise final same-state findings from independently audited values."""
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
A=HERE/'audit_same_state_arrays.json'
audit=json.loads(A.read_text())
assert audit['status']=='passed'
POLICIES=('base','dense','random25','speed25','previous-observed-base-risk25')
def stats(values): return {'seed_values':values,'mean':statistics.fmean(values),'sample_sd':statistics.stdev(values)}
def vals(objective,metric): return [audit['per_run'][f'{objective}_seed{s}']['means'][metric] for s in range(3)]
def pm(values,scale=1,form='.6g'):
    q=stats(values)
    return f"{q['mean']*scale:{form}} ± {q['sample_sd']*scale:{form}}"
derived={}
lines=['# Final same-state and runtime audit','',
f"Independent audit passed **{audit['checks_passed']:,} checks**. It reopened all 1,782 numeric frame archives and raw JSON records; verified their checksums and stability; reconstructed graphs and allocations; recomputed residuals, four Spearman coefficients/frame, all repeated-call statistics, weighted alternatives, and every saved within-seed policy/objective comparison. The existing strict loader independently checked recorded protocol/checkpoint/data provenance. No model loading, inference, or result replacement occurred.",'',
'The scope is 27 official test trajectories (indices 3–29), each with 11 fixed observed histories, for each of six final 100k checkpoints. Current histories, preceding histories, targets and particle types match the pinned converted source and match across all models. Each policy receives the same current history; previous-observed-base risk uses an additional preceding-history base pass. It is distinct from autonomous own-graph cached risk. Prior inspection of other test records and historical aggregates remains disclosed.','',
'## Error and signed dense benefit','',
'All 1,782 frame outcomes completed, with **zero failed frames and zero undefined values among 7,128 required Spearman coefficients**. Values below are equal-particle means within frame, equal frames within trajectory, equal trajectories within seed, then mean ± sample SD across exactly three training seeds.','',
'| Objective | Policy | Position coordinate MSE | Decoder-equivalent normalized coordinate MSE |','|---|---|---:|---:|']
for obj in ('faithful','nll'):
    for policy in POLICIES:
        lines.append(f'| {obj} | {policy} | {pm(vals(obj,f"accuracy/{policy}/position_coordinate_mse"))} | {pm(vals(obj,f"accuracy/{policy}/normalized_coordinate_mse"))} |')
lines+=['','Dense increases seed-average one-step error in every seed of both objectives. Random25 gives the lowest seed-average one-step error among the five policies in all six models. Previous-observed-base risk has higher seed-average error than random25 in all six models, but its difference from base changes sign across seeds. These statements concern observed-history one-step errors, not autonomous rollout rankings.','',
'| Objective | Previous risk minus random, position MSE | Previous risk minus base, position MSE | Dense minus base, position MSE |','|---|---:|---:|---:|']
for obj in ('faithful','nll'):
    keys=[f'{obj}:previous-observed-base-risk25-minus-random25',f'{obj}:previous-observed-base-risk25-minus-base',f'{obj}:dense-minus-base']
    lines.append('| '+obj+' | '+' | '.join(pm(audit['paired_comparisons'][key]['accuracy/position_coordinate_mse']['seed_values']) for key in keys)+' |')
lines+=['','| Objective | Previous risk / base error | Previous risk / signed dense benefit | Current-base risk / base error | Current-base risk / signed dense benefit |','|---|---:|---:|---:|---:|']
for obj in ('faithful','nll'):
    fields=('previous_risk_vs_base_error','previous_risk_vs_dense_benefit','current_base_risk_vs_base_error','current_base_risk_vs_dense_benefit')
    lines.append('| '+obj+' | '+' | '.join(pm(vals(obj,'correlation/'+field),form='.6f') for field in fields)+' |')
lines+=['','These are averages of within-frame rank correlations, not pooled-particle correlations. The positive risk–error association does not supply positive average rank association with this signed dense intervention. Faithful risk–benefit seed means change sign. Neither result estimates marginal single-edge value.','',
'| Objective | Signed normalized-vector benefit | Positive fraction | Negative fraction | Zero fraction |','|---|---:|---:|---:|---:|']
for obj in ('faithful','nll'):
    fields=('mean_normalized_vector_benefit','positive_fraction','negative_fraction','zero_fraction')
    lines.append('| '+obj+' | '+' | '.join(pm(vals(obj,'benefit/'+field)) for field in fields)+' |')
lines+=['','Benefit is base-minus-dense vector squared error. Negative values are preserved. The sign fractions above use normalized residuals; unequal coordinate scales can change the sign relative to position space. All saved signed array entries were independently recomputed.','',
'## Exact placement and numerical repeat evidence','',
'Across the 297 unique observed states, the mean mandatory count is 2,113.879 undirected pairs (range 599–5,112), the mean annulus count is 1,233.512 (258–3,129), and the mean expanded candidate count is 3,347.391 (857–8,241). All three budgeted selectors retain floor(0.25 × annulus pairs), mean 308.010 optional pairs (64–782), plus every mandatory pair. Their mean total is 2,421.889 pairs; the mean within-state retained fraction of dense pairs is 0.726295. Directed edge counts are twice the retained pair counts. The candidate search still includes the expanded set.','',
'The audit independently reconstructed the natural float64 base graph, expanded-then-filtered float64 graph, and saved float32 squared-distance classification. **All 297 observed states have identical base/annulus sets under the two classifications**; all six models share these geometries. This empirical equality does not assert equivalence on unobserved or autonomous states.','',
'All exact allocations, random stream resets, maximum-endpoint score rankings, lexicographic cutoff ties and optional Jaccard overlaps were checked from arrays. Speed has a cutoff boundary tie at 201/297 histories in every model; previous-risk boundary ties occur at 189–202 histories depending on model. ID-based tie-breaking therefore matters.','']
variation=[]
for p in sorted((ROOT/'work/full-evaluation/same_state').glob('*/trajectory_*.json')):
    row=json.loads(p.read_text())
    for case,z in row['repeat_consistency'].items():
        if z['distinct_pair_hashes']>1:
            variation.append({'record':str(p.relative_to(ROOT)),'record_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_index':row['source_index'],'target_frame':row['target_frame'],'case':case,**z,'calls':[{k:c[k] for k in ('round','slot','pair_sha256')} for c in row['timed_calls'] if c['method']==case],'first_timed_cutoff_ties':row['graph_audit']['policies'][case]['cutoff_ties']})
assert len(variation)==1
lines+=['One case had a different selected pair set across timing repeats: **NLL seed 2, source index 6, target 106, previous-observed-base-risk25**. Rounds 0–4 shared the first pair hash; round 5 had a second hash. The maximum recorded prediction difference from the first timed call is 2.4497509e−5. The first-call cutoff contained three tied pairs, of which one was selected. All other recorded repeated prediction differences are at most 5.9604645e−8; natural/shared-base prediction differences are also at most that value. Per-repeat score/prediction arrays were not saved, so the audit verifies the committed hashes/counts and reported difference, but cannot reconstruct each repeat or establish a unique cause. The first timed call remains the declared residual/allocation reference. No repeat or failure was replaced.','',
'## Measured runtime','',
'There are 64,152 timed predictor calls and 10,692 warmup predictor calls. Each frame executes six cases once for warmup and six rotated rounds; the previous-risk case uses two network passes and two candidate builds per call, and every other case uses one. That is 49 network passes/frame, 87,318 total. All recorded ordering, phase inclusion, repeated means/medians/sample SDs, and source/target pairing were checked. Each case occupies every timing-order slot once per frame.','',
'| Objective | Natural base ms | Dense ms | Random25 ms | Speed25 ms | Previous observed risk ms | Shared-superset base ms |','|---|---:|---:|---:|---:|---:|---:|']
for obj in ('faithful','nll'):
    lines.append('| '+obj+' | '+' | '.join(pm(vals(obj,f'timing/{p}/end_to_end_seconds'),1000,'.3f') for p in POLICIES+('base_shared_superset',))+' |')
lines+=['','| Objective | Extra previous-risk scoring ms | Paired shared-superset minus natural base ms | Previous-risk / natural-base ratio |','|---|---:|---:|---:|']
for obj in ('faithful','nll'):
    ratio=[audit['per_run'][f'{obj}_seed{s}']['extra_descriptive']['risk_vs_base_runtime_ratio'] for s in range(3)]
    derived[obj]={'policy_position_mse_percent_change_vs_base':{p:stats([audit['per_run'][f'{obj}_seed{s}']['extra_descriptive']['policy_position_mse_percent_change_vs_base'][p] for s in range(3)]) for p in POLICIES[1:]},'previous_risk_over_natural_base_ratio':stats(ratio)}
    lines.append('| '+obj+' | '+pm(vals(obj,'timing/previous-observed-base-risk25/score_generation_seconds'),1000,'.3f')+' | '+pm(audit['paired_comparisons'][f'{obj}:base_shared_superset-minus-base']['timing/end_to_end_seconds']['seed_values'],1000,'.3f')+' | '+pm(ratio,form='.5f')+' |')
lines+=['','Timing repetitions first reduce within frame; they are not extra trained-model replicates. The documented native MPS configuration uses SciPy host graphs, two PyTorch threads, CPU fallback disabled and synchronized predictor timing. Times include graph construction/selection, features, transfers, network and decoding, and exclude residual/tie/overlap audits and file writing. Every policy runs the checkpoint variance head. The host is shared.','',
'The previous-observed-base risk measurement pays for its additional full scoring pass at every observed frame. Subtracting this cost would not measure autonomous cached deployment. Natural base versus shared-superset base isolates the imposed expanded-search reference overhead on these states; it is not a production speedup. Autonomous policies create different states and use fixed policy order; their durations cannot isolate placement cost, and failed rollouts are shorter. An optimized mean-only GNS, peak memory, general cached deployment speedup and physical conservation remain unmeasured.','',
'The complete observed-state diagnostic supports an allocation-analysis account with adverse outcomes retained. It does not establish general accuracy–runtime superiority or conference readiness.']
(HERE/'same_state_findings.md').write_text('\n'.join(lines)+'\n')
(HERE/'same_state_findings.json').write_text(json.dumps({'source_audit_sha256':hashlib.sha256(A.read_bytes()).hexdigest(),'derived_seed_descriptions':derived,'repeated_pair_variation':variation},indent=2)+'\n')
print('Wrote same_state_findings.md and same_state_findings.json')
