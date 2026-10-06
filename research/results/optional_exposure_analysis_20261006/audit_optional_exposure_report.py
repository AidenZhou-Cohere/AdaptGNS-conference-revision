"""Read-only numeric review of report page 16; no production renderer imports."""
import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

from pypdf import PdfReader


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before_pdf = sha(args.pdf)
    before_result = sha(args.result)
    result = json.loads(args.result.read_text())
    assert before_result == '71c8924bd17f1cd85a69736611e66ea442bc265f12cb920ebcfb7ab1fd508c65'
    reader = PdfReader(args.pdf)
    assert len(reader.pages) == 16
    text = reader.pages[15].extract_text()
    assert '15. Where the risk-random error gap appears' in text
    table1 = text.split('Risk minus random, MSE x1,000\nFaithful\nNLL\n', 1)[1].split('The both-covered group', 1)[0]
    table2 = text.split('Whole-frame difference, x1,000\nFaithful\nNLL\n', 1)[1].split('Error difference equals', 1)[0]
    pattern = r'(-?\d+\.\d{3}) \+/- (\d+\.\d{3})'
    actual = re.findall(pattern, table1) + re.findall(pattern, table2)
    groups = [('whole', 'error'), ('neither', 'error'), ('left_only', 'error'), ('right_only', 'error'), ('both', 'error'), ('whole', 'cost'), ('whole', 'alignment')]
    expected = []
    cells = []
    for group, quantity in groups:
        suffix = quantity + ('_difference' if group == 'whole' else '_contribution')
        key = f'risk_minus_random/{group}/{suffix}'
        for objective in ['faithful', 'nll']:
            values = [result['runs'][f'{objective}_seed{seed}']['aggregate']['metrics'][key] for seed in range(3)]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            formatted = (f'{mean * 1000:.3f}', f'{sd * 1000:.3f}')
            expected.append(formatted)
            cells.append({'objective': objective, 'metric': key, 'seed_values': values, 'expected_display': list(formatted)})
    assert len(actual) == len(expected) == 14
    assert actual == expected, (actual, expected)
    claims = []
    for name, run in result['runs'].items():
        metrics = run['aggregate']['metrics']
        get = lambda g: metrics[f'risk_minus_random/{g}/error_contribution']
        both_largest = get('both') > 0 and all(get('both') > get(g) for g in ['neither', 'left_only', 'right_only'])
        risk_only_expected_sign = get('left_only') < 0 if run['objective'] == 'faithful' else get('left_only') > 0
        speed_both_positive = metrics['speed_minus_random/both/error_contribution'] > 0
        cost = metrics['risk_minus_random/whole/cost_difference']
        alignment = metrics['risk_minus_random/whole/alignment_difference']
        claims.append({'run': name, 'both_largest_positive': both_largest, 'risk_only_expected_sign': risk_only_expected_sign, 'speed_random_both_positive': speed_both_positive, 'cost_exceeds_positive_alignment': cost > alignment > 0})
        assert all(value for key, value in claims[-1].items() if key != 'run')
    for phrase in ['original observed-history', 'sample SDs', 'no-self-loop evaluation convention', 'do not establish a causal effect of concentration', '42.016 seconds', '34.107 seconds', '67,052,267', '1.78e-15']:
        assert phrase in text, phrase
    assert sha(args.pdf) == before_pdf
    assert sha(args.result) == before_result
    review = {'passed': True, 'pdf': str(args.pdf.resolve()), 'pdf_sha256': before_pdf, 'page_number': 16, 'results_sha256': before_result, 'mean_sd_cells': 14, 'claims_by_seed': claims, 'numeric_cells': cells, 'review_source_sha256': sha(Path(__file__)), 'scope': 'Read-only extraction and numeric/sign review. Means/sample SD independently recomputed from audited seed aggregates. No inference, training, production renderer imports or source modification.'}
    with args.output.open('x') as output:
        json.dump(review, output, indent=2)
        output.write('\n')
    print(json.dumps({'passed': True, 'mean_sd_cells': 14, 'seed_claim_checks': 24, 'pdf_sha256': before_pdf, 'review': str(args.output)}))


if __name__ == '__main__':
    main()
