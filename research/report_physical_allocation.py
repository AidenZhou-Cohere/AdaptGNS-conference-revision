"""Report the fixed exploratory compact-pilot physical allocation controls."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OBJECTIVES = ('faithful', 'nll', 'beta_nll')
SPLITS = ('valid', 'test')
CONTROLS = ('base', 'dense', 'random25', 'speed25', 'current_risk25', 'lagged_base_risk25')
PHYSICAL = ('inverse_count25', 'velocity_rms25')
POLICIES = CONTROLS + PHYSICAL


def stat(value, digits=4):
    if value['mean'] is None:
        return 'undefined'
    return f"{value['mean']:.{digits}f} ± {value['sample_seed_sd']:.{digits}f}"


def render(source, prefix):
    raw = source.read_bytes()
    data = json.loads(raw)
    if data['state'] != 'complete' or len(data['runs']) != 15:
        raise ValueError('A completed-population report requires all fifteen validated runs; retain failed attempts separately')
    groups = data['summaries']
    for run in data['runs'].values():
        if run['state'] != 'complete' or any(len(run['splits'][split]['frames']) != 36 for split in SPLITS):
            raise ValueError('Completed model/frame population differs')
    canonical = data['runs']['faithful_seed0']
    geometry_fields = ('id', 'trajectory', 'step', 'n_particles', 'base_pairs', 'available_extra_pairs', 'extra_pair_budget', 'isolated_particles')
    for split in SPLITS:
        for run in data['runs'].values():
            if [[row[key] for key in geometry_fields] for row in run['splits'][split]['frames']] != [
                    [row[key] for key in geometry_fields] for row in canonical['splits'][split]['frames']]:
                raise ValueError('Cross-model geometry identity differs')
    lines = ['# Cheap physical allocation controls on the compact pilot', '',
        '**Completed separate exploratory comparison: all 15 fixed models, both previously inspected splits, '
        '36 frames per split/model, six replayed controls and two new policies.** No model was trained or selected. '
        'The full-architecture training and its locked five-policy evaluation are unchanged.', '',
        'The new `inverse_count25` score is negative mandatory-base degree. `velocity_rms25` scores each particle '
        'by the RMS last-observed displacement difference over its base neighbors; isolated particles receive '
        'score zero by the declared convention and remain counted. The original pilot graph and pair ordering '
        'are retained, including its float32 base-radius comparison. This differs from the earlier physical '
        'correlation analysis’s strict float64 neighborhood. Every particle participates.', '',
        'Both methods retain all base pairs and exactly floor(0.25 × available annulus pairs). Optional-pair '
        'priority is the maximum endpoint score, with the original deterministic ID tie rule. No target is used '
        'to construct a score. All six original control errors and edge counts were replay-checked before '
        'accepting comparisons. Previous-risk scores use the preceding observed base graph, not an autonomous '
        'controller’s cached own-graph score.', '',
        '## One-step normalized-acceleration coordinate MSE', '',
        'Lower is better. Particles are averaged within each frame, then equally over twelve frames per '
        'trajectory, three trajectories per split, and five training seeds. ± denotes sample seed SD; it '
        'is not a confidence interval over the trajectory population. Missing required frames or seeds '
        'are never replaced with survivor means.', '']
    for split in SPLITS:
        lines += ['### '+split, '', '| Policy | Faithful | NLL | Beta-NLL |', '|---|---:|---:|---:|']
        for policy in POLICIES:
            lines.append('| '+policy+' | '+' | '.join(stat(groups[f'{obj}/{split}']['policies'][policy]) for obj in OBJECTIVES)+' |')
        lines.append('')
    lines += ['## Paired percent differences', '',
        'Each entry is the five-seed mean ± sample SD of 100 × (physical-policy MSE − control MSE) / '
        'control MSE, calculated separately within each seed. Negative favors the physical policy. '
        'This is not the percent difference between group means. All six controls, all objectives and '
        'both splits are retained; no p-values or best-case selection.', '']
    for obj in OBJECTIVES:
        for split in SPLITS:
            lines += ['### '+obj+' / '+split, '', '| Physical policy | '+' | '.join(CONTROLS)+' |', '|---|'+'---:|'*6]
            for policy in PHYSICAL:
                row=groups[f'{obj}/{split}']['physical_minus_control'][policy]
                lines.append('| '+policy+' | '+' | '.join((stat(row[c]['percent_difference'], 3)+'%' if row[c]['percent_difference']['mean'] is not None else 'undefined') for c in CONTROLS)+' |')
            lines.append('')
    lines += ['## Geometry, ties and replay', '',
        'The following counts describe 36 unique observed frames in each split, shared by all models. '
        'Split cutoff ties mean the budget includes some but not all optional pairs with the cutoff score. '
        'Such selection depends on particle IDs under the fixed tie rule.', '',
        '| Split | Isolated particle–frame pairs | inverse_count25 split-tie frames | velocity_rms25 split-tie frames |',
        '|---|---:|---:|---:|']
    for split in SPLITS:
        rows=canonical['splits'][split]['frames']
        lines.append(f"| {split} | {sum(row['isolated_particles'] for row in rows)} | "
                     f"{sum(row['cutoff_ties'][PHYSICAL[0]]['split_cutoff_tie'] for row in rows)} / 36 | "
                     f"{sum(row['cutoff_ties'][PHYSICAL[1]]['split_cutoff_tie'] for row in rows)} / 36 |")
    replays=[entry for run in data['runs'].values() for split in SPLITS
             for row in run['splits'][split]['frames'] for entry in row['control_replay'].values()]
    maximum=max(abs(row['mse_difference']) for row in replays)
    lines += ['', f'All {len(replays):,} frame-policy replay comparisons passed the fixed tolerance '
              f'(rtol 2e−5, atol 1e−7) and exact edge counts. Maximum absolute MSE difference from the '
              f'original controls was {maximum:.9g}. Pair overlaps, full cutoff-tie records, isolated '
              'counts, per-seed absolute differences and raw-array hashes remain in the scalar outputs.', '',
        '## Scope and reproducibility', '',
        'These are one-step interventions on observed histories from a small convenience sample. '
        'They were designed after inspecting earlier pilot and physical-correlation results. They do '
        'not establish long-horizon stability, independent confirmation, a full-architecture advantage '
        'or deployment speedup. The labels “physical” and “sparse” refer to the defined score heuristics, '
        'not a causal interpretation or detected free surface.', '',
        f"The complete single-CPU-thread attempt took {data['elapsed_seconds']:.3f} seconds. "
        'Timings include shared/reused work and concurrent full-model training; they are operational '
        'measurements, not a fair optimized policy-speed comparison. No new frame was allowed to '
        'start after the declared 300-second limit. A fresh output directory and preserved failure '
        'records prevent silent overwrites or automatic retries.', '',
        f"- Result SHA256: `{hashlib.sha256(raw).hexdigest()}`",
        f"- Protocol SHA256: `{data['protocol_sha256']}`",
        f"- Input-identity SHA256: `{data['input_identity_sha256']}`", '',
        '```sh', 'python -m research.physical_allocation_pilot \\',
        '  --data-dir data-pilot --output-dir work/physical-allocation-attempt',
        'python -m research.report_physical_allocation \\',
        '  --input work/physical-allocation-attempt/summary.json \\',
        '  --output-prefix research/results/physical_allocation_pilot', '```', '']
    prefix.with_suffix('.md').write_text('\n'.join(lines))
    fig, axes=plt.subplots(3,2,figsize=(12,10),layout='constrained',sharex=True)
    colors=('#236B8E','#B05B2E')
    for i,obj in enumerate(OBJECTIVES):
        for j,policy in enumerate(PHYSICAL):
            ax=axes[i,j]
            ax.axhline(0,color='#555555',lw=.8)
            for k,control in enumerate(CONTROLS):
                result=groups[f'{obj}/test']['physical_minus_control'][policy][control]['percent_difference']
                values=result['seed_values']
                offsets=np.linspace(-.12,.12,5)
                defined=[(index,value) for index,value in enumerate(values) if value is not None]
                ax.scatter([k+offsets[index] for index,value in defined],
                           [value for index,value in defined],color=colors[j],alpha=.6,s=22)
                if result['mean'] is None:
                    ax.text(k,.97,f'{len(defined)}/5; mean undefined',transform=ax.get_xaxis_transform(),
                            ha='center',va='top',rotation=90,fontsize=7)
                else:
                    ax.plot([k-.22,k+.22],[result['mean']]*2,color='#151515',lw=2)
            ax.set_title(obj+' / '+policy,fontsize=11)
            ax.set_ylabel('MSE difference versus control (%)')
            ax.grid(axis='y',alpha=.2)
            ax.set_xticks(range(6),['base','dense','random','speed','current\nrisk','previous\nrisk'],fontsize=9)
    fig.suptitle('Exploratory compact-pilot test comparison\nDots: all five paired seeds; black bars: mean; negative favors physical policy',fontsize=13)
    fig.savefig(prefix.with_suffix('.png'),dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-prefix',type=Path,required=True)
    args=parser.parse_args()
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    render(args.input,args.output_prefix)


if __name__=='__main__':
    main()
