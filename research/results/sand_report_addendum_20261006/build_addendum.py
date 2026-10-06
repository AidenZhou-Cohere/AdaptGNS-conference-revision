"""Append one author-facing Sand page from admitted saved scalar products.

The canonical report is read-only. No model, array, scientific summary worker,
network operation, or D3 product is used. Existing PDF pages are not regenerated.
"""
from pathlib import Path
import hashlib
import json

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT_SHA = '89fee4bad0dfbf5ebfea93c9ff5cdcad5c6d97ba03b8741272e3cdd4b9c7ce23'
POLICIES = ('base', 'dense', 'random25', 'speed25', 'laggedrisk25', 'relative-velocity-RMS25')
LABELS = dict(zip(POLICIES, ('Base', 'Dense', 'Random25', 'Speed25', 'Cached risk25', 'Relative-velocity RMS25')))
checks = []
shown = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, condition):
    if not condition:
        raise ValueError(name)
    checks.append(name)


def load(binding):
    path = Path(binding['path'])
    check('bound input: ' + str(path), path.is_file() and sha(path) == binding['sha256'])
    return json.loads(path.read_text())


def record(value, key):
    check(key + ': all three declared seeds', set(value['seed_values']) == {'0', '1', '2'})
    check(key + ': complete statistic', all(value['seed_values'][str(i)] is not None for i in range(3))
          and value['mean'] is not None and value['sample_sd'] is not None)
    shown[key] = value
    return value


def signs(value):
    return ['+' if value['seed_values'][str(i)] > 0 else '-' if value['seed_values'][str(i)] < 0 else '0'
            for i in range(3)]


def fmt(value, precision=5, scale=1, signed=False):
    prefix = '+' if signed else ''
    return f"{value['mean'] * scale:{prefix}.{precision}f} +/- {value['sample_sd'] * scale:.{precision}f}"


check('input manifest bytes', sha(HERE/'inputs.json') == INPUT_SHA)
inputs = json.loads((HERE/'inputs.json').read_text())
s = load(inputs['summary'])
a = load(inputs['paired_audit'])
admission = load(inputs['admission'])
runtime = load(inputs['runtime'])
check('admitted interpretation', admission['status'] == 'admitted_fixed_complete_products_for_scientific_interpretation')
check('paired scalar audit bound and passed', a['summary_sha256'] == inputs['summary']['sha256']
      and a['status'] == 'passed_supported_checks')
f, d = s['full_rollout'], s['diagnostics']
check('1080 complete autonomous outcomes', f['coverage'] == {'completed_required_outcome': 1080})
check('3648 complete mixed-stage cells',
      sum(sum(v.values()) for stage in d.values() for v in stage['coverage'].values()) + 1080 == 3648)
check('all declared policies', set(f['absolute']['mean_rollout_mse']['base']) == set(POLICIES))
original_path = Path(inputs['original_report']['path'])
check('canonical report unchanged before authoring', sha(original_path) == inputs['original_report']['sha256'])
original = PdfReader(original_path)
check('20 original pages', len(original.pages) == 20)
backup = HERE/'revision_report_before.pdf'
with backup.open('xb') as stream:
    stream.write(original_path.read_bytes())

styles = {
    'head': ParagraphStyle('head', fontName='Helvetica-Bold', fontSize=18, leading=22,
                           textColor=colors.HexColor('#16334a'), spaceAfter=9),
    'sub': ParagraphStyle('sub', fontName='Helvetica-Bold', fontSize=11, leading=15,
                          textColor=colors.HexColor('#16334a'), spaceBefore=6, spaceAfter=5),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=10, leading=14, spaceAfter=8),
    'small': ParagraphStyle('small', fontName='Helvetica', fontSize=8.5, leading=11,
                            textColor=colors.HexColor('#455767'), spaceAfter=6),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=8, leading=11),
}
story = []
prose = []


def para(text, style='body'):
    check('ASCII report prose', text.isascii())
    prose.append({'style': style, 'text': text})
    story.append(Paragraph(text, styles[style]))


para('19. Graph exposure: Sand', 'head')
para('Completed 100k study | six faithful models, three paired seeds | all 1,080 full-H314 rollouts complete', 'small')
para('Base-only and mixed-graph models train from initialization on all 1,000 training trajectories. Paired arms share initialization, sampled frames and noise. Mix expands half of training examples with a random quarter of optional annulus pairs; the native capped base graph remains intact.')
rows = [['Inference policy', 'Base-only<br/>mean +/- SD', 'Mixed graph<br/>mean +/- SD',
         'Mix minus base<br/>mean +/- SD [seed signs]']]
for policy in POLICIES:
    base = record(f['absolute']['mean_rollout_mse']['base'][policy], 'H314/base/' + policy)
    mix = record(f['absolute']['mean_rollout_mse']['mix'][policy], 'H314/mix/' + policy)
    delta = record(f['mix_minus_base']['mean_rollout_mse'][policy], 'H314/mix_minus_base/' + policy)
    rows.append([LABELS[policy], fmt(base), fmt(mix), fmt(delta, signed=True) + ' [' + '/'.join(signs(delta)) + ']'])
table = Table([[Paragraph(str(value), styles['cell']) for value in row] for row in rows],
              colWidths=[112, 126, 126, 160], repeatRows=1)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9eff3')),
    ('LINEBELOW', (0, 0), (-1, 0), .8, colors.HexColor('#708291')),
    ('LINEBELOW', (0, -1), (-1, -1), .5, colors.HexColor('#708291')),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
]))
story.extend([table, Spacer(1, 8)])
para('Full-horizon position MSE; each seed equally averages all 30 trajectory means. Means and sample SDs describe three seeds. Signs are ordered seeds 0/1/2; a minus favors mixed training. All six policies and all required outcomes are retained.', 'small')

for policy in POLICIES[1:]:
    observed_policy = 'previous-observed-base-risk25' if policy == 'laggedrisk25' else policy
    value = record(d['same_state_test']['mix_minus_base']['accuracy/' + observed_policy + '/position_coordinate_mse'],
                   'observed_test/mix_minus_base/' + policy)
    check(policy + ': observed error improves in every seed', signs(value) == ['-'] * 3)
check('dense full-rollout exposure improves every seed', signs(shown['H314/mix_minus_base/dense']) == ['-'] * 3)
for arm in ('base', 'mix'):
    for policy in ('random25', 'dense'):
        value = record(f['within_arm_policy_contrasts']['mean_rollout_mse'][arm][policy + '_minus_base'],
                       'H314/' + arm + '/' + policy + '_minus_base')
        check(arm + '/' + policy + ': native base better every seed', signs(value) == ['+'] * 3)
para('<b>Useful training effect, conditional expansion.</b> Exposure lowers observed-test error under all five expanded policies in every seed; dense full-rollout error also improves in every seed. Yet native base beats both random25 and dense in every seed of both arms. Learning to use an expanded graph does not imply that expansion outperforms the native graph.', 'small')

para('Residual ranking does not transfer through feedback', 'sub')
gaps = {arm: record(d['same_state_test']['previous_observed_risk_minus_random_position_mse'][arm],
                    'observed_test/risk_minus_random/' + arm) for arm in ('base', 'mix')}
for arm in ('base', 'mix'):
    check('observed risk loses every seed: ' + arm, signs(gaps[arm]) == ['+'] * 3)
interaction = record(f['risk_minus_random_mix_minus_base_interaction']['mean_rollout_mse'], 'H314/risk_interaction')
check('autonomous interaction adverse every seed', signs(interaction) == ['+'] * 3)
check('cached risk exposure adverse every seed', signs(shown['H314/mix_minus_base/laggedrisk25']) == ['+'] * 3)
para('On matched observed test histories, the risk-minus-random deficit narrows from '
     + fmt(gaps['base'], 2, 1e9, True) + ' to ' + fmt(gaps['mix'], 2, 1e9, True)
     + ' (position-coordinate MSE x 10<super>-9</super>), but risk remains worse in every seed of both arms. '
       'In autonomous rollouts, exposure worsens cached-risk error in every seed and widens its risk-minus-random gap by '
     + fmt(interaction, 5, 1, True) + ', also adverse in every seed. Observed risk uses the preceding observed base graph; autonomous risk caches its own selected-graph scores.', 'small')

para('Completion still leaves physical and cost limitations', 'sub')
outside_key = 'predicted_boundary_mean_fraction_particles_outside_by_more_than_1e-6'
excursion_key = 'predicted_boundary_trajectory_maximum_excursion'
outside = record(f['absolute'][outside_key]['mix']['random25'], 'boundary/mix/random25/outside')
truth_outside = record(f['absolute']['ground_truth_boundary_mean_fraction_particles_outside_by_more_than_1e-6']['mix']['random25'], 'boundary/truth/outside')
excursion = record(f['absolute'][excursion_key]['mix']['random25'], 'boundary/mix/random25/excursion')
truth_excursion = record(f['absolute']['ground_truth_boundary_trajectory_maximum_excursion']['mix']['random25'], 'boundary/truth/excursion')
for policy in POLICIES:
    check(policy + ': lower mean outside fraction',
          f['absolute'][outside_key]['mix'][policy]['mean'] < f['absolute'][outside_key]['base'][policy]['mean'])
    check(policy + ': higher excursion every seed', all(
        f['absolute'][excursion_key]['mix'][policy]['seed_values'][str(seed)] >
        f['absolute'][excursion_key]['base'][policy]['seed_values'][str(seed)] for seed in range(3)))
    check(policy + ': higher mean committed time',
          runtime['statistics']['mix'][policy]['mean'] > runtime['statistics']['base'][policy]['mean'])
para(f"Mixed random25 places {100 * outside['mean']:.2f}% of particles outside the metadata box by more than 1e-6 "
     f"(truth: {100 * truth_outside['mean']:.4f}%), with mean trajectory-maximum excursion {excursion['mean']:.4f} "
     f"(truth: {truth_excursion['mean']:.7f}). Across all policies, mean outside fractions fall after exposure, "
     'but maximum excursions increase in every paired seed. These are geometric diagnostics, not conservation tests.', 'small')
times = {arm: record(runtime['statistics'][arm]['random25'], 'committed_seconds_per_case/' + arm + '/random25')
         for arm in ('base', 'mix')}
para('Mean committed-call time increases for all six policies; random25 changes from '
     + fmt(times['base'], 2) + ' to ' + fmt(times['mix'], 2) + ' seconds/case. Calls include native parity. '
       'Fixed policy order, shared hosts and evolving geometries prevent an isolated speedup claim.', 'small')

para('Coverage and scope', 'sub')
para('All 3,648 mixed-stage cells complete: 1,080 autonomous outcomes, 900 observed validation histories, 900 observed test histories and 768 clean-validation frames. None failed or remained uncompleted. Sand is an exploratory, distinct native-CUDA lineage; earlier strict CPU/CUDA gradient checks failed, so CPU equivalence and a controlled material effect are not claimed.', 'small')
para('Source: completed Sand100k paired summary and audited saved-array/scalar evidence; manuscript appendix, "Sand: graph exposure and placement." The preceding 20 report pages preserve their dated snapshots. This page contains no Goop3D numerical claim.', 'small')


def footer(canvas, doc):
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#617383'))
    canvas.drawString(44, 25, 'Adaptive Interaction Graphs | completed graph-exposure studies')
    canvas.drawRightString(568, 25, str(20 + doc.page))


addendum_path = HERE/'sand_evidence_addendum.pdf'
check('fresh addendum path', not addendum_path.exists())
SimpleDocTemplate(str(addendum_path), pagesize=(612, 792), leftMargin=44, rightMargin=44,
                  topMargin=38, bottomMargin=43).build(story, onFirstPage=footer, onLaterPages=footer)
check('one-page addendum', len(PdfReader(addendum_path).pages) == 1)
candidate_path = HERE/'revision_report_candidate.pdf'
writer = PdfWriter()
writer.append(PdfReader(backup))
writer.append(PdfReader(addendum_path))
with candidate_path.open('xb') as stream:
    writer.write(stream)
candidate = PdfReader(candidate_path)
check('21-page combined candidate', len(candidate.pages) == 21)
content_preservation = []
for index, page in enumerate(original.pages):
    before, after = page.get_contents().get_data(), candidate.pages[index].get_contents().get_data()
    check(f'original page {index + 1}: content-stream bytes preserved', before == after)
    check(f'original page {index + 1}: extracted text preserved', page.extract_text() == candidate.pages[index].extract_text())
    content_preservation.append({'page': index + 1, 'decoded_content_sha256': hashlib.sha256(before).hexdigest()})
check('canonical report unchanged after authoring', sha(original_path) == inputs['original_report']['sha256'])
for binding in ('summary', 'paired_audit', 'admission', 'runtime'):
    check('scientific scalar source unchanged: ' + binding, sha(Path(inputs[binding]['path'])) == inputs[binding]['sha256'])
for name, value in (
    ('displayed_statistic_objects.json', shown),
    ('addendum_prose.json', prose),
    ('report_checks.json', {
        'status': 'scalar_binding_and_pdf_structure_passed_pending_visual_review',
        'inputs': inputs, 'original_pages': 20, 'addendum_pages': 1, 'candidate_pages': 21,
        'checks': checks, 'statistic_objects_recorded': len(shown),
        'original_content_preservation': content_preservation,
        'original_pdf_sha256': sha(original_path), 'candidate_pdf_sha256': sha(candidate_path),
        'addendum_pdf_sha256': sha(addendum_path), 'source_sha256': sha(Path(__file__)),
        'scientific_reruns_or_raw_array_model_reads': False, 'd3_numeric_product_reads': False,
        'canonical_report_written': False,
    }),
):
    with (HERE/name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
print(json.dumps({'candidate': str(candidate_path), 'candidate_sha256': sha(candidate_path),
                  'addendum': str(addendum_path), 'addendum_sha256': sha(addendum_path),
                  'checks': len(checks), 'displayed_statistic_objects': len(shown)}))
