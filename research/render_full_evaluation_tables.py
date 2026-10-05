"""Generate manuscript result tables from the immutable six-model summaries."""
from pathlib import Path
import json, statistics, hashlib, argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reports-dir',type=Path,default=Path(__file__).resolve().parent/'results/full_evaluation_20261005/reports')
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
E=args.reports_dir; OUT=args.output_dir
OUT.mkdir(parents=True,exist_ok=True)
EXPECTED={'full_rollouts.json': '260bfafe1fd5a11c8bb84a229daee3e6de4ff841aa29a89574f6a096da9ae324', 'full_same_state.json': 'bef3c5958aa61a48cb3d1d294a6b458a0e8ca0069d4a631a227d356318b080b1'}
for name,digest in EXPECTED.items():
 if hashlib.sha256((E/name).read_bytes()).hexdigest()!=digest: raise ValueError('Summary hash differs: '+name)
r=json.loads((E/'full_rollouts.json').read_text());s=json.loads((E/'full_same_state.json').read_text())
assert r['state']=='complete' and s['state']=='complete'
assert r['complete_models']==s['complete_models']==6
policies=r['required_policies']; names={'base':'Base','dense':'Dense','random25':'Random25','speed25':'Speed25','laggedrisk25':'Cached risk25','previous-observed-base-risk25':'Previous observed risk25','base_shared_superset':'Base, shared superset'}
def pm(v,fmt='.4f',scale=1):
 if v['mean'] is None:return r'---'
 return f"${v['mean']*scale:{fmt}}\\pm{v['sample_sd']*scale:{fmt}}$"
def table(caption,label,columns,rows,layout,wide=False):
 t='table*' if wide else 'table'
 return '\n'.join([f'\\begin{{{t}}}[t]',r'\centering\small',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+layout+'}',r'\toprule',' & '.join(columns)+r'\\',r'\midrule']+[' & '.join(row)+r'\\' for row in rows]+[r'\bottomrule',r'\end{tabular}',f'\\end{{{t}}}'])+'\n'
rows=[]
for p in policies:
 gs=[r['policies'][o][p] for o in ('faithful','nll')]
 rows.append([names[p]]+[pm(g['metrics']['mean_rollout_mse']) for g in gs]+[' / '.join(str(sum(x['failed_trajectories'] for x in g['per_seed'])) for g in gs)])
main=r'''\subsection{Fixed full-architecture WaterDrop results}
All six width-128, ten-block models completed the fixed 100,000 updates on all
1,000 official training trajectories. Corrected NLL and faithful regression use
three paired seeds and noise augmentation. NLL's clean validation MSE is higher
in every pair despite smaller binned risk gaps. Final checkpoints were fixed
before the locked test evaluation; Appendix~\ref{sec:full-protocol} gives the protocol.
The locked evaluation covers 27 official test trajectories (indices 3--29),
995 forecasts, and all five policies. Indices 0--2 and historical aggregate
results were already inspected; this is not pristine independent confirmation.

Training uses base-only graphs with self-loops and a 128-neighbor cap; the locked
policy evaluation uses uncapped symmetric pairs without self-loops. Expanded
annulus edges were not exposed during this full-model training, unlike the
compact pilot. Thus these results jointly reflect allocation and graph-input
shift; they do not by themselves isolate a defect in the risk score.

'''+table('Full-architecture position MSE over 995 forecasts, mean $\\pm$ sample SD of three equal-trajectory seed means. Failures are counts out of 81 outcomes per objective/policy, shown faithful / NLL. A failed trajectory makes the full-horizon group mean undefined.','tab:full-rollouts',['Policy','Faithful','NLL','Failures'],rows,'lrrr',wide=True)+r'''
All 405 faithful outcomes completed; NLL had eight coordinate-guard failures,
all at seed 2 (Table~\ref{tab:full-rollouts}). Cached risk has higher faithful
full-rollout MSE than speed in every seed, with paired difference
$0.00301\pm0.00315$. Versus base and random, differences are
$0.00155\pm0.00263$ and $0.00083\pm0.00225$, respectively; both change sign
across seeds. It improves on dense in all three seeds. NLL's cached-risk
full-horizon mean is undefined, precluding a general advantage claim.
Computational completion also masks geometric error: faithful cached risk
places $13.11\pm6.10\%$ of particles outside metadata bounds by more than
$10^{-6}$, versus $2.68\%$ in the same ground-truth frames. Its mean trajectory-maximum
excursion is $.302$, compared with $.239$ for base and $.00782$ for truth.

On 297 fixed observed histories per model, mean frame correlations of
previous-base risk with base residuals are ($.357\pm.060$ faithful, $.411\pm.023$ NLL) and with signed dense benefit are ($-.022\pm.045$, $-.048\pm.012$). Random allocation
has lower one-step MSE than previous risk in every seed of both objectives.
The observed-history risk diagnostic requires an extra scoring pass and is
distinct from autonomous cached risk. Six rotated, synchronized timing repeats
per frame give natural-base costs of 10.74/10.50 ms and random costs of
11.80/11.55 ms (faithful/NLL); previous-observed risk costs 22.38/21.88 ms,
including scoring. Every policy executes the risk head. These local reference
measurements do not establish optimized deployment speedup. Full endpoint,
failure, boundary, placement, and timing results are in
Appendix~\ref{sec:full-results}.
'''
(OUT/'full_evaluation_main.tex').write_text(main)
a=[r'\section{Full-architecture evaluation and measured cost}',r'\label{sec:full-results}',r'''The queue completed on October 5, 2026 at 20:22 UTC. All six autonomous and six
same-state jobs completed; scientific guard failures remain outcomes rather
than retry candidates. Every table uses exactly three training seeds. Within
an autonomous seed, trajectories are weighted equally; same-state measurements
average 11 targets within each trajectory, then 27 trajectories. The targets
are fixed at '''+', '.join(map(str,s['required_target_frames']))+r'''. No particle or time-step independence is assumed. A missing or failed required outcome
makes its all-sample full-horizon summary undefined; accepted failure prefixes
are stored separately. Position MSE averages coordinates and particles.
''']
rows=[]
for o in ('faithful','nll'):
 for p in policies:
  g=r['policies'][o][p]
  rows.append([o,names[p]]+[pm(g['metrics'][m],'.5f') for m in ('mean_rollout_mse','mse_at_200','mse_at_995')]+[str(sum(x['failed_trajectories'] for x in g['per_seed']))+'/81'])
a.append(table('All locked autonomous endpoints. Secondary MSE at forecast 200 is defined for all outcomes because every failure occurred later. Undefined entries retain failures, rather than averaging survivors.','tab:full-endpoints',['Objective','Policy','Mean MSE','MSE@200','MSE@995','Failed'],rows,'llrrrr'))
rows=[]
for o in ('faithful','nll'):
 for p in policies:
  for seed in r['policies'][o][p]['per_seed']:
   for f in seed['failures']:
    fail=f['failure']; rows.append([o,str(seed['seed']),str(f['source_index']),names[p],str(f['completed_steps']),fail['category'].replace('_',r'\_')])
a.append(table('All eight scientific failures. The forward attempt exceeded the declared absolute-coordinate guard of 10; each completed-prefix count excludes that rejected attempt. No failed outcome was rerun.','tab:full-failures',['Objective','Seed','Index','Policy','Accepted steps','Reason'],rows,'lrllrl'))
a.append(r'''These coordinate thresholds are computational guards, not water-container
boundaries. They therefore do not replace geometric diagnostics. The earliest
failure accepted 876 forecasts and the latest accepted 918. No model or
checkpoint was selected from these outcomes.
''')
rows=[]
for o in ('faithful','nll'):
 for p in policies:
  g=r['policies'][o][p]
  rows.append([o,names[p]]+[('---' if x['metrics']['mean_rollout_mse'] is None else f"{x['metrics']['mean_rollout_mse']:.6f}")+f" ({x['failed_trajectories']})" for x in g['per_seed']])
a.append(table('Full-horizon position MSE for every seed, with failed-trajectory counts out of 27 in parentheses. An undefined seed is not replaced by a survivor mean.','tab:full-seeds',['Objective','Policy','Seed 0','Seed 1','Seed 2'],rows,'llrrr'))
rows=[]
for o in ('faithful','nll'):
 for p in ('base','dense','random25','speed25'):
  g=r['paired_comparisons'][f'{o}:laggedrisk25-minus-{p}']
  rows.append([o,'Cached risk minus '+names[p]]+[pm(g['metrics'][m],'.5f') for m in ('mean_rollout_mse','mse_at_200','failure_fraction')])
a.append(table('Paired cached-risk differences, negative favoring cached risk. Differences first pair identical trajectory IDs within each seed. Failure fractions remain defined even when full-horizon error does not. Three seeds warrant descriptive, not strong significance, conclusions.','tab:full-paired',['Objective','Comparison','Mean MSE','MSE@200','Failure fraction'],rows,'llrrr'))
rows=[]
for o in ('faithful','nll'):
 for p in policies:
  g=r['policies'][o][p]['metrics']
  rows.append([o,names[p],pm(g['predicted_boundary_mean_fraction_outside_gt_1e_minus6'],'.2f',100),pm(g['predicted_boundary_mean_step_maximum_excursion'],'.4f'),pm(g['predicted_boundary_trajectory_maximum_excursion'],'.4f')])
a.append(table('Predicted boundary diagnostics. Outside fraction (percent) averages particles then forecast steps. The last column averages each trajectory\'s worst excursion; it is not the single worst particle across the study. Failed groups are undefined.','tab:full-boundaries',['Objective','Policy','Outside (\\%)','Mean step max','Mean trajectory max'],rows,'llrrr'))
a.append(r'''For every complete group, the corresponding truth references are $2.6777\%$
outside, $0.005671$ mean step-maximum excursion, and $0.007820$ mean
trajectory-maximum excursion. The truth outside fraction is nonzero, so
metadata-bound violations alone are not an exact physical-failure classifier.
Nevertheless predicted excursions greatly exceed these references. Truth
references for failed groups are also undefined under the same full-sample
rule. Saved accepted-prefix diagnostics exclude the rejected attempted state.
''')
rows=[]
for p in s['policies']:
 gs=[s['objectives'][o]['metrics'] for o in ('faithful','nll')]
 rows.append([names[p]]+[pm(g[f'accuracy/{p}/normalized_coordinate_mse'],'.5f') for g in gs]+[pm(g[f'timing/{p}/end_to_end_seconds'],'.2f',1000) for g in gs])
a.append(table('Same observed histories: normalized acceleration coordinate MSE and measured end-to-end milliseconds. Previous observed risk includes its extra scoring pass; these timings are not amortized autonomous cached-risk costs.','tab:full-same-state',['Policy','Faithful MSE','NLL MSE','Faithful ms','NLL ms'],rows,'lrrrr'))
a.append(r'''There are 1,782 complete observed-frame outcomes (six models times 297),
with no failed frames and no undefined values among 7,128 correlation
coefficients. Every budgeted action preserves the same base pairs and
retains exactly $\lfloor.25|C_R|\rfloor$ optional pairs. Candidate pairs,
selected sets, overlaps and cutoff ties are recorded. The previous-risk score
is evaluated on the preceding observed base history; it is not the cache
from an autonomous trajectory. Dense benefit is the signed per-particle
base-minus-dense vector squared residual, not marginal single-edge value.
''')
rows=[]
for key,label in [('correlation/previous_risk_vs_base_error','Previous risk / base error'),('correlation/previous_risk_vs_dense_benefit','Previous risk / dense benefit'),('correlation/current_base_risk_vs_base_error','Current risk / base error'),('correlation/current_base_risk_vs_dense_benefit','Current risk / dense benefit'),('benefit/mean_normalized_vector_benefit','Mean signed normalized vector benefit'),('benefit/positive_fraction','Positive-benefit fraction'),('benefit/negative_fraction','Negative-benefit fraction'),('benefit/zero_fraction','Zero-benefit fraction')]:
 rows.append([label]+[pm(s['objectives'][o]['metrics'][key],'.6f' if key=='benefit/zero_fraction' else '.4f') for o in ('faithful','nll')])
a.append(table('Same-state residual ranking and signed dense interventions. Spearman correlations are computed within each frame; undefined correlations are retained, not silently omitted. Benefit signs refer to normalized vector error, whose coordinate scaling can differ from position-space signs.','tab:full-benefit',['Diagnostic','Faithful','NLL'],rows,'lrr'))
rows=[]
for o in ('faithful','nll'):
 g=s['objectives'][o]['metrics']
 rows.append([o,pm(g['timing/base/end_to_end_seconds'],'.3f',1000),pm(g['timing/base_shared_superset/end_to_end_seconds'],'.3f',1000),pm(g['timing/previous-observed-base-risk25/score_generation_seconds'],'.3f',1000)])
a.append(table('Timing controls, milliseconds: natural base searches radius $r$; the shared-superset control imposes the expanded candidate search while retaining the base graph. Score cost is the separate previous-history graph and network pass.','tab:full-cost-controls',['Objective','Natural base','Shared superset base','Previous-risk scoring'],rows,'lrrr'))
a.append(r'''One warmup per case precedes six rotated timing rounds on identical observed
histories, using synchronized native Metal and two PyTorch threads on the
shared local host. Timings include graph construction, selection, features,
transfers, network execution and decoding; residual analysis and file writing
are excluded. The risk head runs for every policy, including base; an optimized
mean-only baseline and peak memory are unmeasured. Different floating-point
radius classifications are also retained: same-state uses float64 distances,
whereas autonomous base classification uses float32 squared distances.
All 297 observed geometries produced identical pair sets under these two
radius classifiers in the saved audit; this does not guarantee equality on
other states. One retained repeated-call variation occurred for NLL seed 2,
source index 6, target 106: previous-observed-base risk produced two pair hashes
across six timed calls, with maximum recorded prediction difference
$2.45\times10^{-5}$. All other policy repeat differences were at most
$5.96\times10^{-8}$. Exact counts remained unchanged; no repeat was discarded
or rerun to improve the result. The reported accuracy uses the saved diagnostic
prediction, while timings retain every repeated call.
''')
rows=[]
for o in ('faithful','nll'):
 for p in policies:
  g=r['policies'][o][p]['metrics']
  rows.append([o,names[p],pm(g['mean_rollout_wall_seconds'],'.3f'),pm(g['mean_directed_edges'],'.1f')])
a.append(table('Autonomous elapsed seconds per trajectory and directed edges per accepted forecast, aggregated over three seed means. Wall time retains failed attempts and cached-risk warmup. Full-horizon edge means are undefined for failed groups.','tab:full-autonomous-cost',['Objective','Policy','Seconds','Directed edges'],rows,'llrr'))
a.append(r'''Autonomous policies run in fixed order and generate different states, so these
durations cannot isolate placement cost or certify a speedup. Failed outcomes
also have shorter horizons. Cached risk uses 996 network passes for a completed
trajectory (one warmup plus 995 forecasts); other policies use 995. Lower edge
counts on separate rollout geometries do not contradict base preservation on
identical positions. The largest recorded candidate set has 14,815 pairs; the largest retained
graph has 10,297 unordered pairs (20,594 directed edges). Accepted prefixes
and rejected attempts have distinct worst excursions of 9.09846 and 9.11111.
These are whole-study extrema, unlike the equal-trajectory means above.
All statuses, source/configuration/checkpoint/data hashes,
trajectory outcomes, boundary prefixes and timings are preserved with the
reproduction scripts. These results do not establish a general accuracy--cost
advantage, physical validity, convergence, or exact historical reproduction.
''')
(OUT/'full_evaluation_appendix.tex').write_text('\n\n'.join(a))
manifest={'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [E/'full_rollouts.json',E/'full_same_state.json']},'generated':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['full_evaluation_main.tex','full_evaluation_appendix.tex']}}
(OUT/'paper_results_generation.json').write_text(json.dumps(manifest,indent=2)+'\n')
