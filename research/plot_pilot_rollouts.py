"""Plot all-seed rollout accuracy and all-outcome failure counts."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=Path('research/results/rollout_summary.json'))
    parser.add_argument('--output', type=Path, default=Path('research/results/figures/pilot_rollouts'))
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    policies = ['base', 'dense', 'random25', 'speed25', 'laggedrisk25']
    labels = ['Base', 'Dense', 'Random', 'Speed', 'Lagged risk']
    objectives = ['faithful', 'nll', 'beta_nll']
    names = ['Faithful', 'NLL', r'$\beta$-NLL']
    colors = ['#31688e', '#35a18b', '#a86143']
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), gridspec_kw={'width_ratios': [1.5, 1]}, constrained_layout=True)
    ax = axes[0]
    for j, (kind, name, color) in enumerate(zip(objectives, names, colors)):
        offset = (j-1)*.22
        for i, policy in enumerate(policies):
            st = data['horizons']['200']['groups'][kind][policy]['metrics']['mse_at_200']
            values = np.asarray(st['seed_values'], dtype=float)
            if len(values) != 5 or not np.isfinite(values).all():
                raise ValueError('Accuracy panel requires every seed outcome')
            xs = i+offset+np.linspace(-.035,.035, len(values))
            ax.scatter(xs, values, s=11, alpha=.42, color=color, linewidths=0)
            ax.scatter(i+offset, st['mean'], marker='_', s=130, linewidths=2, color=color,
                       label=name if i == 0 else None)
    ax.set_xticks(range(5), labels)
    ax.set_ylabel('Position MSE at forecast step 200')
    ax.set_title('a  All 225 runs completed at 200 steps', loc='left', fontsize=10)
    ax.grid(axis='y', color='#e6e6e6', lw=.6)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    counts = np.array([[data['horizons']['995']['groups'][o][p]['failed'] for p in policies] for o in objectives])
    if data['horizons']['995']['attempted'] != 225:
        raise ValueError('Failure matrix requires all 225 outcomes')
    ax = axes[1]
    ax.imshow(counts, vmin=0, vmax=15, cmap='YlOrRd', aspect='auto')
    for row in range(3):
        for col in range(5):
            ax.text(col, row, f'{counts[row,col]}/15', ha='center', va='center', fontsize=10,
                    color='white' if counts[row,col] >= 10 else '#252525')
    ax.set_xticks(range(5), labels, rotation=25, ha='right')
    ax.set_yticks(range(3), names)
    ax.set_title('b  Failures before 995 forecast steps', loc='left', fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.supxlabel('Compact WaterDrop pilot · 5 seeds × 3 test trajectories per cell · exploratory', fontsize=8, color='#555555')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(args.output.with_suffix('.'+ext), dpi=190, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
