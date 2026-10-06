"""Draft complete Sand presentation from admitted saved scalar products only."""
from pathlib import Path
import gzip
import hashlib
import json
import math
import re
import statistics

from prepare_candidate import load, same_stats
from render_tables import POLICIES, NAMES, ordered, pm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'generated_v2'
INPUT_PIN = 'e99d325d6ae594db83b3a77f8e3b1cef46fdd00a71c5a328b6f2f9917e8f7d6c'


def digest(raw): return hashlib.sha256(raw).hexdigest()
def write(name, value):
    with (OUT / name).open('x') as stream:
        if isinstance(value, str): stream.write(value)
        else: json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
def number(value, scale=1, precision=5):
    return '---' if value is None else f'{value*scale:.{precision}f}'
def seeds(record, scale=1, precision=5):
    return [number(v, scale, precision) for v in ordered(record)]
def stats(values):
    return {'seed_values': {str(i): v for i, v in enumerate(values)}, 'required_seed_pairs': 3,
            'defined_seed_pairs': sum(v is not None for v in values),
            'mean': statistics.fmean(values) if all(v is not None for v in values) else None,
            'sample_sd': statistics.stdev(values) if all(v is not None for v in values) else None}
def subtract(a, b):
    return stats([None if x is None or y is None else x-y for x,y in zip(ordered(a), ordered(b))])
def signs(record):
    return ['null' if x is None else '+' if x > 0 else '-' if x < 0 else '0' for x in ordered(record)]
def table(caption, label, columns, header, rows):
    return '\n'.join([r'\begin{table*}[t]', r'\centering\scriptsize', r'\caption{' + caption + '}',
        r'\label{' + label + '}', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{' + columns + '}',
        r'\toprule', ' & '.join(header) + r'\\', r'\midrule'] +
        [' & '.join(row) + r'\\' for row in rows] + [r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])


def main():
    supplied = load({'path': str(HERE/'admitted_inputs.json'), 'sha256': INPUT_PIN})
    sand = load(supplied['materials']['Sand']['summary'])
    paired = load(supplied['materials']['Sand']['paired_audit'])
    admission = load(supplied['materials']['Sand']['admission'])
    assert admission['schema'] == 'sand_complete_cohort_interpretation_admission_root_v2'
    assert admission['status'] == 'admitted_fixed_complete_products_for_scientific_interpretation'
    assert admission['policies'] == list(POLICIES)
    assert admission['products']['sand_summarize']['sha256'] == paired['summary_sha256'] == supplied['materials']['Sand']['summary']['sha256']
    full, diagnostics = sand['full_rollout'], sand['diagnostics']
    assert full['coverage'] == {'completed_required_outcome': 1080}
    assert sum(sum(c.values()) for d in diagnostics.values() for c in d['coverage'].values()) + 1080 == 3648
    claims = {'summary_sha256': supplied['materials']['Sand']['summary']['sha256'],
              'paired_audit_sha256': supplied['materials']['Sand']['paired_audit']['sha256'],
              'admission_sha256': supplied['materials']['Sand']['admission']['sha256'], 'records': {}}
    def claim(key, path, record):
        claims['records'][key] = {'summary_key': path, 'statistic': record, 'seed_signs': signs(record)}
        return record
    for stage in ('same_state_valid', 'same_state_test'):
        d = diagnostics[stage]
        for arm in ('base','mix'):
            claim(stage + '_risk_minus_random_' + arm,
                  ['diagnostics',stage,'previous_observed_risk_minus_random_position_mse',arm],
                  d['previous_observed_risk_minus_random_position_mse'][arm])
        claim(stage + '_interaction', ['diagnostics',stage,'risk_minus_random_mix_minus_base_interaction'], d['risk_minus_random_mix_minus_base_interaction'])
        for policy in POLICIES:
            observed = 'previous-observed-base-risk25' if policy == 'laggedrisk25' else policy
            key = 'accuracy/' + observed + '/position_coordinate_mse'
            rec = d['mix_minus_base'][key]
            same_stats(rec, subtract(d['absolute']['mix'][key],d['absolute']['base'][key]))
            claim(stage + '_training_' + policy, ['diagnostics',stage,'mix_minus_base',key], rec)
    for policy in POLICIES:
        rec = full['mix_minus_base']['mean_rollout_mse'][policy]
        same_stats(rec, subtract(full['absolute']['mean_rollout_mse']['mix'][policy],full['absolute']['mean_rollout_mse']['base'][policy]))
        claim('H314_training_'+policy, ['full_rollout','mix_minus_base','mean_rollout_mse',policy], rec)
    for arm in ('base','mix'):
        for key, rec in full['within_arm_policy_contrasts']['mean_rollout_mse'][arm].items():
            policy, ref = key.split('_minus_')
            same_stats(rec, subtract(full['absolute']['mean_rollout_mse'][arm][policy],full['absolute']['mean_rollout_mse'][arm][ref]))
            claim('H314_'+arm+'_'+key, ['full_rollout','within_arm_policy_contrasts','mean_rollout_mse',arm,key], rec)
    claim('H314_interaction', ['full_rollout','risk_minus_random_mix_minus_base_interaction','mean_rollout_mse'],full['risk_minus_random_mix_minus_base_interaction']['mean_rollout_mse'])
    # Assert every seed sign used in prose, including the adverse comparisons.
    assert signs(claims['records']['H314_training_laggedrisk25']['statistic']) == ['+']*3
    assert signs(claims['records']['H314_interaction']['statistic']) == ['+']*3
    assert signs(claims['records']['H314_training_random25']['statistic']) == ['-','-','+']
    assert signs(claims['records']['H314_training_dense']['statistic']) == ['-']*3
    for arm in ('base','mix'):
        for key in ('random25_minus_base','dense_minus_base','dense_minus_random25','laggedrisk25_minus_random25'):
            assert signs(claims['records']['H314_'+arm+'_'+key]['statistic']) == ['+']*3
        for stage in ('same_state_valid','same_state_test'):
            assert signs(claims['records'][stage+'_risk_minus_random_'+arm]['statistic']) == ['+']*3
            assert signs(claims['records'][stage+'_interaction']['statistic']) == ['-']*3
        for metric in ('previous_risk_vs_base_error', 'previous_risk_vs_previous-observed-base-risk25_benefit'):
            key = 'correlations/' + metric
            rec = diagnostics['same_state_test']['absolute'][arm][key]
            claim('same_state_test_' + arm + '_' + metric,
                  ['diagnostics','same_state_test','absolute',arm,key], rec)
            assert signs(rec) == (['-']*3 if metric.endswith('_benefit') else ['+']*3)
        rms = claims['records']['H314_'+arm+'_relative-velocity-RMS25_minus_base']['statistic']
        assert signs(rms) == ['+','+','-']
    assert signs(claims['records']['same_state_test_training_base']['statistic']) == ['+','+','-']
    for policy in POLICIES[1:]:
        assert signs(claims['records']['same_state_test_training_'+policy]['statistic']) == ['-']*3
    for metric, expected in (('normalized_acceleration_coordinate_mse',['+','+','-']),
                             ('constant_free_gaussian_nll',['+','-','-'])):
        key = 'metrics/' + metric
        rec = diagnostics['clean_validation']['mix_minus_base'][key]
        claim('clean_validation_training_' + metric,
              ['diagnostics','clean_validation','mix_minus_base',key], rec)
        assert signs(rec) == expected
    runtime = {}
    for arm in ('base','mix'):
        runtime[arm] = {}
        for policy in POLICIES:
            records = sorted([v for v in full['runtime'] if v['arm']==arm],key=lambda v:v['seed'])
            assert [v['seed'] for v in records] == [0,1,2]
            assert all(v['per_policy'][policy]['committed_cases']==30 for v in records)
            values = [v['per_policy'][policy]['committed_call_seconds']/30 for v in records]
            runtime[arm][policy] = stats(values)
    outside = 'predicted_boundary_mean_fraction_particles_outside_by_more_than_1e-6'
    excursion = 'predicted_boundary_trajectory_maximum_excursion'
    for policy in POLICIES:
        e = subtract(full['absolute'][excursion]['mix'][policy],full['absolute'][excursion]['base'][policy])
        o = subtract(full['absolute'][outside]['mix'][policy],full['absolute'][outside]['base'][policy])
        assert signs(e) == ['+']*3 and o['mean'] < 0
        assert runtime['mix'][policy]['mean'] > runtime['base'][policy]['mean']
        claims['records']['excursion_training_'+policy] = {'derived_from_summary_keys': ['full_rollout','absolute',excursion], 'statistic': e, 'seed_signs': signs(e)}
    write('claim_source_map.json', claims)
    runtime_companion = {'formula': 'For each model/policy, committed_call_seconds / 30; then mean and sample SD over all three ordered training seeds.',
        'scope': 'Committed synchronized calls include parity; fixed policy order, shared hosts and differing rollout geometries. Not a causal speedup or full-invocation elapsed time.', 'statistics': runtime, 'original_records':full['runtime']}
    write('descriptive_runtime.json', runtime_companion)

    appendix = (HERE/'sand_methods_appendix_candidate.tex').read_text().split('\n',1)[1]
    appendix += '\nAll 24 model-stage families complete: 1,080 autonomous outcomes, 900 observed validation histories, 900 observed test histories and 768 clean-validation frames, totaling 3,648 mixed-stage cells. These cell types are not interchangeable rollout denominators. No required scientific outcome failed or remained uncompleted. Earlier transport/capture failures were retained and recovered operationally without rerunning scientific workers.\n\n'
    rows=[]
    for arm in ('base','mix'):
        for policy in POLICIES:
            r=full['absolute']['mean_rollout_mse'][arm][policy]
            rows.append([arm,NAMES[policy],pm(r,0,5)]+seeds(r,1,5))
    appendix += table('Sand full-H314 position MSE for every training arm, policy and seed. Each row includes all 90 required outcomes; failed and not-completed counts are zero. No shortened endpoint replaces the full horizon.', 'tab:sand-all-rollouts','llrrrr',['Training','Policy',r'Mean $\pm$ SD','Seed 0','Seed 1','Seed 2'],rows)
    rows=[]
    for arm in ('base','mix'):
        for policy in POLICIES:
            op='previous-observed-base-risk25' if policy=='laggedrisk25' else policy
            key='accuracy/'+op+'/position_coordinate_mse';r=diagnostics['same_state_test']['absolute'][arm][key]
            rows.append([arm,'Previous risk25' if policy=='laggedrisk25' else NAMES[policy],pm(diagnostics['same_state_valid']['absolute'][arm][key],-8,3),pm(r,-8,3)]+seeds(r,1e8,3))
    appendix += table(r'Sand position-coordinate MSE on observed histories ($\times10^{-8}$). Every model has 150 validation and 150 test histories. Risk uses the preceding observed base history. All test seed means are retained; validation seed values and normalized-coordinate metrics are in the complete scalar companion.', 'tab:sand-all-observed','llrrrrr',['Training','Policy','Validation','Test','Test seed 0','Test seed 1','Test seed 2'],rows)
    rows=[]
    for policy in POLICIES:
        op='previous-observed-base-risk25' if policy=='laggedrisk25' else policy
        obs=diagnostics['same_state_test']['mix_minus_base']['accuracy/'+op+'/position_coordinate_mse'];r=full['mix_minus_base']['mean_rollout_mse'][policy]
        rows.append([NAMES[policy],pm(obs,-9,3,True),pm(r,0,5,True)]+seeds(r,1,5))
    appendix += table(r'Every Sand policy training effect, mixed minus base. The observed-test column uses position-coordinate MSE $\times10^{-9}$; autonomous columns use full-H314 position MSE. In the observed column, the risk row uses previous-observed-base scores; autonomous risk uses its own cached selected-graph scores. Negative values favor mixed training.', 'tab:sand-all-training-effects','lrrrrr',['Policy','Observed test','H314','H314 seed 0','H314 seed 1','H314 seed 2'],rows)
    rows=[]
    for name,stage in [('Validation','same_state_valid'),('Test','same_state_test')]:
        d=diagnostics[stage]
        for arm in ('base','mix'):
            r=d['previous_observed_risk_minus_random_position_mse'][arm]
            rows.append([name+' ($10^{-9}$)',arm,pm(r,-9,3,True)]+seeds(r,1e9,3))
        r=d['risk_minus_random_mix_minus_base_interaction'];rows.append([name+' ($10^{-9}$)','interaction',pm(r,-9,3,True)]+seeds(r,1e9,3))
    for arm in ('base','mix'):
        r=full['within_arm_policy_contrasts']['mean_rollout_mse'][arm]['laggedrisk25_minus_random25'];rows.append(['H314',arm,pm(r,0,5,True)]+seeds(r,1,5))
    r=full['risk_minus_random_mix_minus_base_interaction']['mean_rollout_mse'];rows.append(['H314','interaction',pm(r,0,5,True)]+seeds(r,1,5))
    appendix += table(r'Sand risk minus random within each training arm, and the mixed-minus-base change in that gap. Observed validation and test use coordinate-squared units $\times10^{-9}$; autonomous rows use full-H314 position MSE. All three ordered seeds are retained. A negative interaction does not imply that either absolute risk-minus-random gap is negative.', 'tab:sand-all-placement','llrrrr',['Endpoint (scale)','Training/contrast',r'Mean $\pm$ SD','Seed 0','Seed 1','Seed 2'],rows)
    rows=[]
    for arm in ('base','mix'):
        for policy in POLICIES:
            rows.append([arm,NAMES[policy],pm(full['absolute'][outside][arm][policy],-2,2),pm(full['absolute'][excursion][arm][policy],0,4),pm(runtime[arm][policy],0,3)])
    appendix += table(r'Sand full-H314 geometry and descriptive committed-call time, as three-seed means $\pm$ sample SDs. Outside counts particles beyond the metadata box by more than $10^{-6}$. Excursion is the trajectory-maximum coordinate excursion, then averaged over trajectories. Matching truth is $0.061825\%$ outside and $0.00079537$ excursion for every row (zero across-seed SD). Calls include native parity; setup, publication and operational recovery are separate. Fixed policy order, shared hosts and different geometries prevent a causal speedup claim.', 'tab:sand-all-cost','llrrr',['Training','Policy',r'Outside (\%)','Excursion','s/case'],rows)
    rows=[]
    clean_names={'metrics/normalized_acceleration_coordinate_mse':'Normalized coordinate MSE','metrics/constant_free_gaussian_nll':'Constant-free NLL','metrics/predicted_normalized_vector_se':'Predicted vector squared error','metrics/realized_normalized_vector_se':'Realized vector squared error'}
    for arm in ('base','mix'):
        for key,label in clean_names.items():
            r=diagnostics['clean_validation']['absolute'][arm][key];rows.append([arm,label,pm(r,0,4)]+seeds(r,1,4))
    appendix += table('Sand clean validation under the base graph: all 128 declared frames per model, aggregated equally within trajectory and then across trajectories. All three seeds are shown. These residual-scale diagnostics do not establish action-value calibration or autonomous stability.', 'tab:sand-clean','llrrrr',['Training','Metric',r'Mean $\pm$ SD','Seed 0','Seed 1','Seed 2'],rows)
    appendix += r'''
Exposure lowers observed-test error under all five expanded policies in every seed, while base-policy error worsens in seeds 0 and 1. Previous-observed risk remains worse than random in every validation and test seed of both arms, despite a negative interaction in every seed. On test, its mean frame Spearman correlation with base error is $.251\pm.086$ for base training and $.272\pm.026$ for mixed training; correlation with its own sparse-action benefit is $-.119\pm.044$ and $-.062\pm.036$, respectively. All six of these per-model benefit correlations are negative. Clean-validation mean MSE decreases only in seed 2; NLL decreases in seeds 1 and 2. Aggregate residual fit therefore does not resolve the placement problem.

Autonomous behavior differs. Random25 improves under exposure in seeds 0 and 1 but worsens in seed 2; cached risk worsens in all three. The risk-minus-random interaction is positive in every seed. Native base outperforms random25 in every seed of both arms: mixed random's mean $.07240$ exceeds mixed native base's $.06781$, while their base-training means are $.08298$ and $.06785$. Dense exposure improves every seed, yet dense remains worse than both native base and random in every seed of both arms. Speed exposure worsens two seeds; RMS exposure worsens one. RMS beats native base only in seed 2 in each arm. None of these comparisons establishes a generally beneficial expansion policy.

Mixed training lowers mean outside-box fractions for every policy, yet increases trajectory-maximum excursion in every policy and every paired seed. Mean committed time also increases for every policy, with mixed seed signs. The complete policy families still show substantial physical mismatch. The complete scalar companion retains every fixed absolute metric, policy contrast, seed, null, physical reference, runtime record and accounting category; these tables do not discard inconvenient outcomes or substitute forecast 200 for H314.
'''.replace('\\\\pm','\\pm')
    write('appendix_sand.tex',appendix)

    # Preserve the single existing figure and replace prose rather than adding a
    # third main table or main diagnostic section.
    original=(ROOT/'work/conference_experiments_main.tex').read_text()
    main=(OUT/'main_tables_only_candidate.tex').read_text()
    substitutions=[]
    def replace_paragraph(starts,new):
        nonlocal main
        old=next(p for p in main.split('\n\n') if p.startswith(starts))
        assert main.count(old)==1
        main=main.replace(old,new)
        substitutions.append({'old':old,'new':new})
    replace_paragraph('We separate two questions:',r'''We separate two questions: does training on expanded graphs improve a simulator, and does residual risk improve placement of a fixed expansion budget? Fresh Goop and Sand models and paired WaterDrop continuations test both. Earlier WaterDrop controls examine why error ranking and action value differ; historical comparisons remain in Appendix~\ref{sec:historical-audit}.''')
    replace_paragraph('We train six Goop models',r'''For each of Goop and Sand, we train six faithful models from initialization to 100k: base-only and mixed-graph arms at three paired seeds. Both use width 128, ten message-passing blocks and all 1,000 training trajectories. Mix adds a uniformly random quarter of optional pairs with probability $1/2$; initial weights, frames and noise are paired. Native capped base graphs and self-messages are preserved. Random25, speed25, risk25 and RMS25 use the same quarter-annulus budget and radius ratio 1.267; base and dense control expansion. Each observed test uses 150 common histories per model, with risk scored on the preceding observed base history. Autonomous risk instead caches scores from its own preceding selected graph. Sand's distinct native-CUDA lineage is qualified in Appendix~\ref{sec:sand-graph-exposure}.''')
    replace_paragraph('Graph exposure improves full-rollout error',r'''On Goop, fixed-random25 H395 MSE falls from $0.301\pm0.169$ to $0.129\pm0.031$ after exposure, improving every seed. At matched histories, previous-observed risk remains worse than random in every mixed-model seed despite a smaller deficit (Table~\ref{tab:goop-exposure-placement}). Its correlation is $.214\pm.088$ with base error but $.001\pm.048$ with its own sparse-action benefit. Exposure improves this simulator without making residual rank a reliable allocation guide.''')
    replace_paragraph('We average frames within trajectories',r'''Frames average within trajectories, then trajectories within each seed; means and sample SDs retain all three seeds. Figure~\ref{fig:goop-exposure-placement} shows the Goop pairs. Follow-up designs are exploratory, informed by earlier analyses, with endpoints and policies fixed before evaluation.''')
    replace_paragraph('A separate WaterDrop follow-up',r'''WaterDrop pairs 10k continuations from three faithful 100k parents, inheriting optimizer state and matched frame/noise schedules. This estimates effects conditional on those inspected parents. Exposure changes the observed-test risk-minus-random gap from $(+1.62\pm1.07)\times10^{-10}$ to $(-0.99\pm0.67)\times10^{-10}$, reversing every seed. Validation's gap improves too, but mixed risk remains worse than random on average.

Sand also narrows the observed-test risk deficit in every seed, yet risk remains worse than random in both arms. Its gap changes from $(+4.76\pm3.80)\times10^{-9}$ to $(+1.87\pm1.04)\times10^{-9}$. A better interaction therefore need not mean better allocation than random.''')
    replace_paragraph('Among complete mixed-training policies',r'''Goop mixed speed25 and RMS25 have full-horizon MSE $.112\pm.027$ and $.125\pm.032$ and beat random25 in every seed. Mixed random, speed and RMS still place $22.16\%$, $19.55\%$ and $20.53\%$ of particles beyond the metadata box by $10^{-6}$, versus $0.0632\%$ for truth. Their mean trajectory-maximum excursions are $.401$, $.417$ and $.426$, versus $.000722$. These are geometric diagnostics, not conservation tests. All policies, failures and costs remain in Appendix~\ref{sec:goop-graph-exposure}; Figure~\ref{fig:qualitative-goop2d} gives a fixed metadata-selected example.''')
    replace_paragraph('For WaterDrop, all 810',r'''All 810 WaterDrop H995 outcomes complete. Mix improves random25 by $-.0060\pm.0056$, and dense, speed and cached risk also improve in every seed. Its autonomous risk-minus-random interaction is nevertheless $+.00284\pm.00466$, with mixed seed signs. Box excursions and increased recorded times remain (Appendix~\ref{sec:waterdrop-110k-exposure}).

All 1,080 Sand H314 outcomes complete, but exposure makes cached-risk error worse in every seed ($+.00917\pm.01201$), widening its risk-minus-random gap in every seed ($+.01975\pm.03062$). Random25 improves in two seeds, not three, and native base beats it in every seed of both arms. Dense exposure improves every seed, yet dense remains worse than native base and random. Thus fixed-policy exposure gains do not establish that expansion beats the base graph. All Sand policy means, signs, physical mismatches and costs remain in Appendix~\ref{sec:sand-graph-exposure}.''')
    figure=r'\\begin\{figure\*\}\[t\].*?\\end\{figure\*\}'
    assert re.findall(figure,original,re.S)==re.findall(figure,main,re.S)
    assert len(re.findall(r'\\begin\{table\}',main))==2
    write('conference_experiments_main_candidate.tex',main)
    write('experiments_prose_substitutions.json',substitutions)

    body=(ROOT/'work/manuscript_body.tex').read_text();body_sub=[]
    def body_paragraph(starts,new):
        old=next(p for p in body.split('\n\n') if p.startswith(starts))
        assert body.count(old)==1
        body_sub.append({'old':old,'new':new})
    body_paragraph('We investigate this connection',r'''We investigate this connection with paired graph-exposure studies. Goop and Sand compare base-only and mixed-graph training from initialization at a fixed optional-pair budget; WaterDrop pairs 10k continuations from three faithful 100k parents. Full rollouts test autonomous feedback. Earlier WaterDrop objective, action-benefit and graph-convention controls examine why prediction difficulty and intervention benefit can differ.''')
    body_paragraph('On Goop, mixed-graph training',r'''Exposure improves random-policy full rollouts in every paired Goop and WaterDrop seed. Goop risk still loses to random at matched test histories; WaterDrop reverses that observed-test ordering but has mixed autonomous interactions. Sand is adverse: exposure narrows the observed risk deficit while worsening cached-risk rollouts and the autonomous risk-minus-random gap in every seed. Native base also beats random expansion in every Sand seed of both arms. Training support, observed placement and autonomous allocation therefore require separate evidence.''')
    body_paragraph('The compact pilot uses few trajectories',r'''The compact pilot reuses few inspected trajectories. Full studies use three seeds and distinct interventions: Goop and Sand train from initialization for 100k updates; WaterDrop continues inspected 100k parents for 10k more. Follow-ups are exploratory; differing sources, horizons, backends and interventions do not isolate material effects or establish broad generalization. Sand's earlier strict CPU/CUDA gradient checks failed; its admitted native-CUDA lineage makes no CPU-equivalence claim. Observed risk uses a preceding observed base graph, whereas autonomous risk caches its own selected graph. One does not certify the other. Kinematic associations and box excursions neither identify material mechanisms nor test conservation. Peak memory and optimized runtime comparisons remain unmeasured.''')
    body_paragraph('The paired graph-exposure studies separate',r'''The paired studies separate training support from graph placement. Exposure improves fixed-random full rollouts in every Goop and WaterDrop seed, while their risk-placement conclusions remain conditional: Goop observed risk loses to random and failures leave its autonomous interaction undefined; WaterDrop reverses the observed-test ordering but has mixed autonomous interactions. Sand narrows its observed risk deficit yet worsens cached-risk rollout error and the autonomous risk-minus-random gap in every seed. Its native base graph also beats random expansion in every seed of both arms. These findings support explicit budget and failure accounting, not a general allocation or efficiency advantage.''')
    write('canonical_body_substitutions.json',body_sub)
    # Complete compact companion: every statistic from all three materials plus
    # all Sand accounting, runtime and independently audited scalar evidence.
    all_stats=json.loads((OUT/'all_scalar_statistic_map.json').read_text())
    companion={'schema':'cross_material_complete_presentation_companion_v1','frozen_presentation_plan_sha256':admission['presentation_plan_sha256'],
        'inputs':supplied,'statistics':all_stats,'sand_claims':claims,'sand_accounting':{'autonomous':full['coverage'],'diagnostics':{k:v['coverage'] for k,v in diagnostics.items()},'failed_prefixes':full['failed_accepted_prefixes'],'queue_accounting':sand['queue_accounting']},
        'sand_descriptive_runtime':runtime_companion,'sand_diagnostic_runtime':{k:v['runtime'] for k,v in diagnostics.items()},'scope':'Admitted saved scalar products only. All three seeds and every predeclared policy retained, no pooling, survivor means, model/array reruns, or new endpoints.'}
    raw=(json.dumps(companion,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    (OUT/'complete_companion.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    write('companion_manifest.json',{'compressed_sha256':digest((OUT/'complete_companion.json.gz').read_bytes()),'uncompressed_sha256':digest(raw),'uncompressed_bytes':len(raw),'statistic_objects_by_material':{m:len(v) for m,v in all_stats.items()},'total_statistic_objects':sum(len(v) for v in all_stats.values())})

    report=['# Sand: complete admitted interpretation','',
        'The principal new finding is adverse for cached-risk allocation. All 1,080 autonomous outcomes and all 3,648 mixed-stage cells complete, but graph exposure worsens cached-risk H314 error and its risk-minus-random gap in every paired seed. This does not contradict a conditional observed-history interaction; it demonstrates that the interaction does not transfer automatically through autonomous feedback.','',
        '## Exposure and absolute allocation','',
        '| Evaluation policy | Observed-test mix−base (×10⁻⁹) | Seed signs | H314 mix−base | Seed signs |','|---|---:|:---:|---:|:---:|']
    def md(r,scale=1):
        return 'undefined' if r['mean'] is None else f'{r["mean"]*scale:+.7g} ± {r["sample_sd"]*scale:.7g}'
    for policy in POLICIES:
        obs=claims['records']['same_state_test_training_'+policy]['statistic'];r=claims['records']['H314_training_'+policy]['statistic']
        report.append('| '+policy+' | '+md(obs,1e9)+' | '+' / '.join(signs(obs))+' | '+md(r)+' | '+' / '.join(signs(r))+' |')
    report += ['', 'Every expanded policy improves observed-test error in all three seeds. Base-policy observed error worsens in seeds 0 and 1. Under autonomous feedback, only dense exposure improves all three seeds; random, base and RMS improve two, and speed improves one. Cached risk improves none. Dense itself remains worse than base and random in all seeds and both arms. Native base beats random in all seeds of both arms. A training improvement under a fixed expanded policy therefore cannot be presented as evidence that expansion outperforms the native graph.', '',
        '## Risk placement at each endpoint','',
        '| Endpoint | Base risk−random | Mixed risk−random | Interaction | Interaction signs |','|---|---:|---:|---:|:---:|']
    for label,stage,scale in [('Observed validation (×10⁻⁹)','same_state_valid',1e9),('Observed test (×10⁻⁹)','same_state_test',1e9)]:
        d=diagnostics[stage];r=d['risk_minus_random_mix_minus_base_interaction'];report.append('| '+label+' | '+md(d['previous_observed_risk_minus_random_position_mse']['base'],scale)+' | '+md(d['previous_observed_risk_minus_random_position_mse']['mix'],scale)+' | '+md(r,scale)+' | '+' / '.join(signs(r))+' |')
    report.append('| Autonomous H314 | '+md(full['within_arm_policy_contrasts']['mean_rollout_mse']['base']['laggedrisk25_minus_random25'])+' | '+md(full['within_arm_policy_contrasts']['mean_rollout_mse']['mix']['laggedrisk25_minus_random25'])+' | '+md(full['risk_minus_random_mix_minus_base_interaction']['mean_rollout_mse'])+' | + / + / + |')
    report += ['', 'All observed and autonomous within-arm risk-minus-random values are positive in every seed. The observed interaction is negative in every seed; the autonomous interaction is positive in every seed. The complete companion retains the exact ordered seed values, including the small positive H314 interaction in seed 2 (3.9882560103739195e−5). Three training seeds do not support treating trajectories or frames as independent replicates.', '',
        '## Physical and computational qualifications','',
        'Across all six policies, mixed training lowers the across-seed mean outside-box fraction but increases trajectory-maximum excursion in every paired seed. Mixed random25 has 10.4701% outside (truth 0.0618251%) and mean trajectory-maximum excursion 0.324406 (truth 0.000795367). Thus a lower fraction outside is not a uniform physical improvement. All six mean committed-call times increase under mix; their seed signs are mixed. Synchronized calls include native parity and exclude separate setup/publication/recovery. Shared hosts, fixed policy order and changed rollout geometries preclude a causal speedup claim.', '',
        'Observed-test residual-risk/base-error correlations remain positive (base 0.251146±0.086003; mix 0.271585±0.025858), while risk/own-sparse-benefit correlations are negative (base −0.118949±0.044055; mix −0.062382±0.036330), with all per-model benefit correlations negative. Clean-validation MSE improves only in seed 2; NLL improves in seeds 1 and 2. These mixed residual-fit diagnostics do not establish action-value calibration.', '',
        '## Cross-material reading and provenance','',
        'The fixed tables retain Goop, WaterDrop and Sand in their predeclared order. Goop and WaterDrop improve random-policy autonomous error in every paired seed, while Sand improves only two. Goop retains three guard failures and undefined affected full-horizon statistics. WaterDrop is a 100k-parent-to-110k continuation with five policies, so RMS is not predeclared. Goop and Sand are fresh 100k training studies with six policies; their horizons are395 and314, versus WaterDrop995. These differences and Sand’s distinct CUDA lineage prevent a controlled material-effect interpretation. No title/abstract update or favorable-policy selection is made.', '',
        'The compact companion contains all4,216 three-seed statistic objects:1,107 Goop,2,002 WaterDrop and1,107 Sand, together with exact input pins and full Sand accounting/runtime provenance. Primary appendix tables retain every policy/arm and all primary test/full-horizon seed means. The retained paired audit contains25,606 checks. Saved-array/scalar-series support does not replay unsaved trajectories, source normalization, graph construction or model execution. Earlier failed source variants, numerical gates and transport attempts remain preserved.', '']
    write('interpretation.md','\n'.join(report))
    def prose_words(text):
        text=re.sub(r'\\begin\{(?:table|figure)\*?\}.*?\\end\{(?:table|figure)\*?\}','',text,flags=re.S)
        return len(text.split())
    before=body.split(r'\clearpage',1)[0]
    after=before
    for item in body_sub:after=after.replace(item['old'],item['new'])
    write('integration_receipt.json',{'status':'candidate_ready_root_integration_and_native_layout_pending',
        'main_existing_prose_words_excluding_floats':prose_words(original),'main_candidate_prose_words_excluding_floats':prose_words(main),
        'body_main_word_delta':len(after.split())-len(before.split()),'existing_main_tables':2,'candidate_main_tables':2,
        'existing_figure_byte_preserved':True,'title_abstract_edited':False,'canonical_files_edited':False,
        'no_scientific_program_rerun_or_raw_array_model_read':True,'source_bindings':supplied,
        'files_sha256':{p.name:digest(p.read_bytes()) for p in OUT.iterdir() if p.is_file()}})
    for material in supplied['materials'].values():
        for binding in material.values():
            assert digest(Path(binding['path']).read_bytes())==binding['sha256']
    print(json.dumps({'status':'Sand candidate completed','output':str(OUT),'prose_word_delta':prose_words(main)-prose_words(original),'body_main_word_delta':len(after.split())-len(before.split())}))


if __name__=='__main__':main()
