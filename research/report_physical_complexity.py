"""Render every cached physical-descriptor association, preserving missingness."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from research.analyze_physical_complexity import METRICS, OBJECTIVES, SPLITS

LABELS = {
    'strain': 'Strain norm', 'vorticity': 'Absolute vorticity',
    'strain_partial': 'Strain, adjusted ranks', 'vorticity_partial': 'Vorticity, adjusted ranks',
    'divergence': 'Absolute divergence', 'velocity_dispersion': 'Neighbor velocity RMS difference',
    'error_on_fit': 'Base error (same fit mask)', 'benefit_on_fit': 'Dense benefit (same fit mask)',
    'neighbor_count_all': 'Neighbor count (all particles)', 'speed_all': 'Speed (all particles)',
    'wall_clearance_all': 'Signed wall clearance (all particles)',
    'observed_acceleration_all': 'Observed acceleration (all particles)',
    'error_all': 'Base error (all particles)', 'benefit_all': 'Dense benefit (all particles)'}


def format_stat(record, conditional=False):
    key, sd = ('defined_frame_mean', 'defined_frame_sample_seed_sd') if conditional else ('mean', 'sample_seed_sd')
    if record[key] is None:
        return 'undefined'
    return f'{record[key]:.3f} ± {record[sd]:.3f}'


def render(source, prefix):
    raw = source.read_bytes()
    data = json.loads(raw)
    groups = [f'{objective}/{split}' for objective in OBJECTIVES for split in SPLITS]
    table = data['summaries']
    matrix = np.array([[table[group][key]['mean'] if table[group][key]['mean'] is not None else np.nan
                        for group in groups] for key in METRICS])
    fig, ax = plt.subplots(figsize=(13.5, 9.5), layout='constrained')
    fig.get_layout_engine().set(rect=(0, .035, 1, .965))
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#dddddd')
    heat = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(groups)), [group.replace('/', '\n') for group in groups])
    ax.set_yticks(range(len(METRICS)), [LABELS[key] for key in METRICS])
    ax.tick_params(length=0, pad=8)
    ax.xaxis.tick_top()
    for i, key in enumerate(METRICS):
        for j, group in enumerate(groups):
            ax.text(j, i, format_stat(table[group][key]), ha='center', va='center', fontsize=9,
                    color='white' if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) > .65 else '#111111')
    ax.set_title('Exploratory compact pilot: residual risk versus observed descriptors\n'
                 'Within-frame rank correlations; equal trajectories; five-seed mean ± sample SD', pad=53, fontsize=13)
    fig.colorbar(heat, ax=ax, shrink=.75, label='Mean within-frame correlation')
    fig.text(.02, .003, 'Adjusted ranks control for neighbor count, speed and signed wall clearance. '
             'Undefined cells retain missing frames; no p-values or independent confirmation.', fontsize=9)
    image_path = prefix.with_suffix('.png')
    fig.savefig(image_path, dpi=170)
    plt.close(fig)
    lines = ['# Residual risk versus observed physical descriptors', '',
        '**Completed exploratory postprocess of all 15 compact-pilot models; no model calls or new training.** '
        'The same 36 validation and 36 previously inspected test frames were reused for every model. '
        'These are kinematic descriptors, not ground-truth physical complexity or epistemic uncertainty.', '',
        'Strain and vorticity come from a local affine velocity fit using only observed positions and '
        'displacements before the prediction target. The radius is 0.015, with at least three neighbors, '
        'spatial rank two and Gram condition at most 10⁶. Units are per stored frame; constant positive '
        'time rescaling leaves ranks unchanged. Adjusted coefficients correlate rank residuals after '
        'regressing on ranked neighbor count, speed and signed box-plane clearance. This descriptive '
        'adjustment does not establish causality.', '',
        'Coefficients are averaged within each trajectory, then equally across three trajectories and '
        'five training seeds. Values after ± are sample seed SD, not confidence intervals. '
        'No particle-level p-values are computed. A required undefined frame makes the unconditional '
        'aggregate undefined. Conditional defined-frame results below still require all three '
        'trajectories and all five seeds.', '', '## Geometry coverage', '',
        '| Split | Unique frames | Eligible particle–frame pairs / total | Fit coverage range across frames | Outside-box pairs | No-neighbor pairs |',
        '|---|---:|---:|---:|---:|---:|']
    for split in SPLITS:
        rows = data['geometry'][split]
        valid, total = sum(r['fit_valid_particles'] for r in rows), sum(r['n_particles'] for r in rows)
        fractions = [r['fit_valid_fraction'] for r in rows]
        lines.append(f"| {split} | {len(rows)} | {valid:,} / {total:,} ({valid/total:.2%}) | "
                     f"{min(fractions):.2%}–{max(fractions):.2%} | {sum(r['outside_box_particles'] for r in rows)} | "
                     f"{sum(r['no_neighbor_particles'] for r in rows)} |")
    lines += ['', 'Masks are identical across models. Per-frame counts and the mean/median risk of '
              'included and excluded particles are preserved in JSON. Particle–frame pairs are repeated '
              'observations, not independent samples. Box clearance is not a free-surface detector.', '']
    for conditional, title in ((False, 'Unconditional associations'), (True, 'Defined-frame conditional associations')):
        lines += ['', '## '+title, '', '| Descriptor | '+' | '.join(groups)+' |', '|---|'+'---:|'*len(groups)]
        for key in METRICS:
            lines.append('| '+LABELS[key]+' | '+' | '.join(format_stat(table[g][key], conditional) for g in groups)+' |')
        if conditional:
            lines += ['', 'Defined frame counts per seed (out of 36 scheduled frames):', '',
                      '| Descriptor | '+' | '.join(groups)+' |', '|---|'+'---|'*len(groups)]
            for key in METRICS:
                lines.append('| '+LABELS[key]+' | '+' | '.join(','.join(map(str,table[g][key]['defined_frames_per_seed'])) for g in groups)+' |')
    lines += ['', '## Interpretation limits', '',
        'Correlation with deformation is a spatial association. Useful edge allocation must be assessed '
        'by the actual matched-budget intervention and rollout accuracy, failures and measured cost. '
        'Positive whole-dense benefit is neither necessary nor sufficient for useful selected-edge '
        'allocation. Dense benefit is base error minus dense-graph error; it is not '
        'single-edge marginal value and shares the base error algebraically. Risk–error and '
        'risk–benefit comparisons on the identical fit mask are retained above, alongside all-particle '
        'comparisons. No proxy, objective, seed or split was selected by its outcome.', '',
        'These compact networks use a short training budget and a small convenience sample. '
        'The original pilot predictions and both splits had already been inspected. The running '
        'six-model 100,000-update experiment and its five locked policies are unchanged. Applying '
        'this postprocess to full-model same-state artifacts would be a separate exploratory '
        'diagnostic after the existing completion/test gate.', '', '## Reproduction and provenance', '',
        f"CPU postprocessing took {data['postprocessing_seconds']:.3f} seconds; this is not model or policy latency.",
        'The input identity manifest pins the summary from commit d061cef; the analyzer verifies '
        'its data/raw-array hashes and frame/particle joins before using them. No checkpoint '
        'or reserved full-test source was opened.', '',
        f"- Result SHA256: `{hashlib.sha256(raw).hexdigest()}`",
        f"- Analysis protocol SHA256: `{data['protocol_sha256']}`",
        f"- Geometry NPZ SHA256: `{data['geometry_arrays']['sha256']}` (retained locally under work/)",
        f"- Figure: `{image_path.name}`", '',
        '```sh', 'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m research.analyze_physical_complexity \\',
        '  --data-dir data-pilot --output-prefix work/physical_complexity/analysis',
        'python -m research.report_physical_complexity \\',
        '  --input work/physical_complexity/analysis.json --output-prefix research/results/physical_complexity_pilot',
        '```', '']
    prefix.with_suffix('.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    render(args.input, args.output_prefix)


if __name__ == '__main__':
    main()
