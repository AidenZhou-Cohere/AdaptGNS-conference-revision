"""Fixed cross-material table renderer; no experiment, file IO or model imports."""
import math

MATERIALS = ('Goop', 'WaterDrop', 'Sand')
POLICIES = ('base', 'dense', 'random25', 'speed25', 'laggedrisk25', 'relative-velocity-RMS25')
NAMES = dict(zip(POLICIES, ('Base', 'Dense', 'Random25', 'Speed25', 'Cached risk25', 'RMS25')))
OBSERVED = (
    ('mix_minus_base_at_base', r'Mix $-$ base (base)'),
    ('mix_minus_base_at_random25', r'Mix $-$ base (random)'),
    ('risk_minus_random_base_training', r'Risk $-$ random (base)'),
    ('risk_minus_random_mixed_training', r'Risk $-$ random (mix)'),
    ('risk_minus_random_training_interaction', 'Training interaction'),
)


def ordered(record):
    values = record['seed_values']
    if isinstance(values, dict):
        assert set(values) == {'0', '1', '2'}
        values = [values[str(seed)] for seed in range(3)]
    assert len(values) == 3
    assert all(v is None or type(v) in (float, int) and math.isfinite(v) for v in values)
    if any(v is None for v in values):
        assert record['mean'] is None and record['sample_sd'] is None
    else:
        mean = math.fsum(values) / 3
        sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / 2)
        assert math.isclose(mean, record['mean'], rel_tol=1e-10, abs_tol=1e-18)
        assert math.isclose(sd, record['sample_sd'], rel_tol=1e-10, abs_tol=1e-18)
    return values


def pm(record, exponent=0, precision=2, signed=False):
    ordered(record)
    if record['mean'] is None:
        return r'---'
    scale = 10 ** -exponent
    mean, sd = record['mean'] * scale, record['sample_sd'] * scale
    sign = '+' if signed else ''
    # General notation outside compact fixed-point range preserves nonzero signs.
    if abs(mean) >= 1000 or (mean != 0 and abs(mean) < 10 ** -precision):
        return rf'${mean:{sign}.2g}\pm{sd:.2g}$'
    return rf'${mean:{sign}.{precision}f}\pm{sd:.{precision}f}$'


def observed_table(data):
    rows = [r'\begin{table}[t]', r'\centering\scriptsize',
        r'\caption{Observed-test position-coordinate MSE contrasts. Each column uses its stated coordinate-squared scale. Entries are means $\pm$ sample SDs of three paired seeds. Negative contrasts favor the first term; the last row changes the risk-minus-random gap. Goop and Sand train from initialization to 100k; WaterDrop continues inspected 100k parents to 110k. Materials are not pooled.}',
        r'\label{tab:goop-exposure-placement}', r'\setlength{\tabcolsep}{2pt}',
        r'\begin{tabular}{lccc}', r'\toprule',
        'Contrast & ' + ' & '.join(r'\shortstack{' + material + r'\\$\times10^{' + str(data[material]['observed_display_exponent']) + r'}$}' for material in MATERIALS) + r'\\',
        r'\midrule']
    for key, label in OBSERVED:
        rows.append(label + ' & ' + ' & '.join(pm(data[m]['observed'][key], data[m]['observed_display_exponent'], 2, True) for m in MATERIALS) + r'\\')
    return '\n'.join(rows + [r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def full_table(data):
    rows = [r'\begin{table}[t]', r'\centering\scriptsize',
        r'\caption{Full-horizon position-MSE training effects (mixed minus base; three-seed mean $\pm$ sample SD). Goop H395 and Sand H314 require 90 outcomes per arm/policy; WaterDrop H995 requires 81. Each lower line reports base/mixed unsuccessful ($U$) and not-completed ($N$) counts. A dash preserves an undefined full-horizon effect; NP denotes a policy not predeclared.}',
        r'\label{tab:goop-rollouts}', r'\setlength{\tabcolsep}{2pt}',
        r'\begin{tabular}{lccc}', r'\toprule',
        r'Policy & Goop & WaterDrop & Sand\\', r'\midrule']
    for policy in POLICIES:
        cells = []
        for material in MATERIALS:
            item = data[material]['full'][policy]
            if item['state'] == 'not_predeclared':
                assert material == 'WaterDrop' and policy == 'relative-velocity-RMS25'
                cells.append('NP')
                continue
            assert item['state'] == 'predeclared'
            u, n = item['unsuccessful'], item['not_completed']
            assert set(u) == set(n) == {'base', 'mix'}
            assert all(type(v) is int and v >= 0 for d in (u, n) for v in d.values())
            counts = rf'$U:{u["base"]}/{u["mix"]},\ N:{n["base"]}/{n["mix"]}$'
            cells.append(r'\shortstack{' + pm(item['training_effect'], 0, 3, True) + r'\\' + counts + '}')
        rows.append(NAMES[policy] + ' & ' + ' & '.join(cells) + r'\\[2pt]')
    return '\n'.join(rows + [r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def replace_tables(canonical, observed, full):
    import re
    replacements = {'tab:goop-exposure-placement': observed, 'tab:goop-rollouts': full}
    matches = list(re.finditer(r'\\begin\{table\}\[t\].*?\\end\{table\}', canonical, re.S))
    assert len(matches) == 2
    for match in reversed(matches):
        labels = re.findall(r'\\label\{([^}]+)\}', match.group())
        assert len(labels) == 1 and labels[0] in replacements
        canonical = canonical[:match.start()] + replacements.pop(labels[0]).rstrip() + canonical[match.end():]
    assert not replacements
    return canonical
