"""Make manuscript tables directly from the audited immutable native summary."""
from pathlib import Path
import hashlib,json,gzip,shutil
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
source=Path(__file__).with_name('native_graph_rollouts.json.gz')
audit=Path(__file__).with_name('independent_scalar_audit.json.gz')
source_bytes=gzip.decompress(source.read_bytes());audit_bytes=gzip.decompress(audit.read_bytes())
x=json.loads(source_bytes);a=json.loads(audit_bytes)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert x['audit']['passed'] and a['passed'] and a['source_summary_sha256']==hashlib.sha256(source_bytes).hexdigest()
assert x['audit']['checks']==19521125 and a['native_outcomes']==810
out=args.output_dir;out.mkdir(exist_ok=False)
labels={'base':'Base','dense':'Dense','random25':'Random','speed25':'Speed','laggedrisk25':'Cached risk'}
policies=tuple(labels)
def pm(m,digits=6,scale=1):
 if m['mean'] is None:return r'undefined'
 return f"${m['mean']*scale:.{digits}f}\\pm{m['sample_sd']*scale:.{digits}f}$"
def value(v):return 'undefined' if v is None else f'{v:.6f}'
parts=[r'''\section{Native-convention autonomous follow-up}
\label{sec:native-rollout-followup}
The observed-history graph control leaves open whether its conclusions persist under feedback. We therefore repeated all 810 trajectory--policy outcomes using the same six fixed 100k WaterDrop models and official sources 3--29, with 995 forecasts per trajectory. This exploratory follow-up preserves the native directed, capped base prefix and self-edge candidates, then appends uncapped optional annulus messages. It jointly changes graph conventions relative to the original locked evaluation; it is not an isolated causal estimate of the self-loop effect. The previously inspected test population and all original results remain distinct and retained.

All 405 faithful outcomes completed. Seven NLL outcomes, all at seed 2, reached the coordinate resource guard between forecasts 861 and 969. No failure was retried. Required failed trajectories make the corresponding seed's full-horizon mean undefined; therefore all five three-seed NLL full-horizon means are undefined. Table~\ref{tab:native-rollout-seeds} retains the other seed values without promoting their average to a complete-cohort result.

\begin{table*}[t]
\caption{Native-convention full-rollout coordinate MSE for every seed. Parentheses give failed outcomes out of 27 for each seed. Faithful models have zero failures throughout. All models remain fixed at 100k updates.}
\label{tab:native-rollout-seeds}
\centering\small
\begin{tabular}{lrrr|rrr}
\toprule
&\multicolumn{3}{c}{Faithful}&\multicolumn{3}{c}{NLL}\\
Policy&Seed 0&Seed 1&Seed 2&Seed 0&Seed 1&Seed 2\\
\midrule''']
for p in policies:
 vals=[]
 for obj in ('faithful','nll'):
  z=x['groups'][obj]['policies'][p]
  for i,t in enumerate(z['metrics']['mean_rollout_mse']['seed_values']):
   vals.append(value(t['value'])+(f" ({z['per_seed'][i]['failed_trajectories']})" if obj=='nll' else ''))
 parts.append(labels[p]+'&'+'&'.join(vals)+r'\\')
parts.append(r'''\bottomrule
\end{tabular}
\end{table*}

\begin{table}[t]
\caption{Paired faithful differences in native-convention full-rollout MSE, cached risk minus comparator. Negative favors risk. Mean and sample SD across all three seeds; these are descriptive effects, not significance claims.}
\label{tab:native-rollout-paired}
\centering\small
\begin{tabular}{lr}
\toprule Comparator&Paired difference\\\midrule''')
for p in ('base','random25','speed25','dense'):
 parts.append(labels[p]+'&'+pm(x['groups']['faithful']['native_policy_comparisons']['laggedrisk25_minus_'+p]['metrics']['mean_rollout_mse'])+r'\\')
parts.append(r'''\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\caption{Faithful native-convention boundary diagnostics, averaged within trajectory and then seed. Outside fraction uses the $10^{-6}$ tolerance; excursion is the mean trajectory-maximum coordinate excursion. These are geometric diagnostics, not certified physical failure labels.}
\label{tab:native-rollout-boundaries}
\centering\small
\begin{tabular}{lrr}
\toprule Policy&Outside (\%)&Excursion\\\midrule''')
for p in policies:
 m=x['groups']['faithful']['policies'][p]['metrics']
 parts.append(labels[p]+'&'+pm(m['predicted_boundary_mean_fraction_outside_gt_1e_minus6'],2,100)+'&'+pm(m['predicted_boundary_trajectory_maximum_excursion'],3)+r'\\')
m=x['groups']['faithful']['policies']['base']['metrics']
parts.append('Truth&'+pm(m['ground_truth_boundary_mean_fraction_outside_gt_1e_minus6'],2,100)+'&'+pm(m['ground_truth_boundary_trajectory_maximum_excursion'],5)+r'\\')
parts.append(r'''\bottomrule
\end{tabular}
\end{table}

Cached risk loses to base in all three faithful seeds, while its contrasts with random and speed change sign. Dense loses to base in all three seeds. Thus matching the trained base convention does not establish a consistent autonomous allocation advantage. In completed faithful trajectories the cap never binds. Their boundary excursions remain substantially larger than truth, even when the outside fraction is comparatively small.

Runtime records retain graph construction, network calls and diagnostics. They use a fixed execution order on policy-dependent geometries and do not identify a policy speedup. The NLL seed-2 process was suspended and resumed; its raw clocks are retained as interrupted elapsed time and excluded from uninterrupted runtime claims. We do not subtract a guessed pause duration or average only convenient trajectories. The original native summary preserves all measured counters; this qualification governs their interpretation.

Saved scalar results and prediction traces are linked to the source, configuration and checkpoint hashes. These consistency checks do not establish physical validity or independent statistical confirmation.
''')
(out/'appendix.tex').write_text('\n'.join(parts))
main=r'''A further native-convention rollout control retains all 810 outcomes (Appendix~\ref{sec:native-rollout-followup}). Faithful cached risk loses to base in all three seeds, with paired MSE difference $.004287\pm.003358$; its random comparison changes sign ($.000447\pm.000721$). Seven NLL guard failures leave all three-seed NLL full-horizon means undefined. Matching the trained base convention therefore does not resolve the allocation deficit.
'''
(out/'main.tex').write_text(main)
record={'source_summary_sha256':hashlib.sha256(source_bytes).hexdigest(),'independent_audit_sha256':hashlib.sha256(audit_bytes).hexdigest(),'generator_sha256':sha(Path(__file__)),'main_sha256':sha(out/'main.tex'),'appendix_sha256':sha(out/'appendix.tex'),'all_seeds_and_policies_retained':True,'runtime_qualification':'All NLLseed2 clocks are preserved as possibly interrupted; no uninterrupted three-seed NLL timing claim.'}
(out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
