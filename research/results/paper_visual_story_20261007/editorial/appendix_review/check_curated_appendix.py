"""Source-only integrity checks for the editorial appendix candidate.
No experiment, aggregation, rendering, compilation, or canonical-file mutation.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
p = HERE / 'curated_appendix.tex'
text = p.read_text()
receipt = json.loads((HERE / 'table_transcription.json').read_text())
checks = {}
checks['candidate_hash_matches_receipt'] = hashlib.sha256(text.encode()).hexdigest() == receipt['candidate_sha256']
# Strip genuine comments, including the sole future insertion marker.
source = '\n'.join(re.split(r'(?<!\\)%', line, maxsplit=1)[0] for line in text.splitlines())
depth = 0
for token in re.finditer(r'(?<!\\)[{}]', source):
    depth += 1 if token.group() == '{' else -1
    assert depth >= 0, ('unmatched closing brace', token.start())
checks['balanced_braces'] = depth == 0
stack = []
for m in re.finditer(r'\\(begin|end)\{([^}]+)\}', source):
    if m[1] == 'begin':
        stack.append(m[2])
    else:
        assert stack and stack.pop() == m[2], ('environment mismatch', m.group())
checks['balanced_environments'] = not stack
checks['even_unescaped_math_delimiters'] = len(re.findall(r'(?<!\\)\$', source)) % 2 == 0
labels = re.findall(r'\\label\{([^}]+)\}', source)
references = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', source)
checks['labels_unique'] = len(labels) == len(set(labels))
checks['local_references_resolve'] = set(references) <= set(labels)
checks['one_future_d3_marker'] = text.count('% GOOP3D_AUTONOMOUS_H295_SEPARATE_INSERT') == 1
checks['no_old_insertion_markers'] = len(re.findall(r'^% .*INSERT', text, re.M)) == 1
checks['no_document_end_in_fragment'] = r'\end{document}' not in text
checks['no_external_file_dependencies'] = not re.search(r'\\(?:input|include|includegraphics)\b', source)

# Independently locate the selected candidate row in its labeled table and
# verify the ordered cells recorded by the extraction receipt. Source cells
# were copied verbatim; no precision-changing numeric parser is used here.
def table_at(label):
    pos = text.index('\\label{' + label + '}')
    lo = text.rfind(r'\begin{table}', 0, pos)
    hi = text.index(r'\end{table}', pos) + len(r'\end{table}')
    return text[lo:hi]

def policy(s):
    return s.lower().replace('previous ', '').replace('cached ', '').replace(' ', '')

verified = 0
for cell in receipt['selected_table_cells']:
    original = cell['source_row']
    arm = {'base': 'Base-only', 'mix': 'Mixed'}[original[0]]
    name = policy(original[1])
    rows = [line for line in table_at(cell['source_label']).splitlines() if line.startswith(arm + ' &')]
    found = []
    for row in rows:
        cols = [c.strip() for c in row.removesuffix('\\\\').split('&')]
        if policy(cols[1]) == name:
            found.append(cols)
    assert len(found) == 1, (cell['source_label'], original)
    cols = found[0]
    # Cells from combined observed/rollout source tables occupy distinct
    # contiguous positions. Verify sequence, not merely set membership.
    wanted = cell['selected_cells']
    matches = [cols[j:j+len(wanted)] for j in range(2, len(cols)-len(wanted)+1)]
    assert wanted in matches, (cell['source_label'], original, cols, wanted)
    verified += len(wanted)
checks['all_selected_source_cells_present_in_correct_rows'] = True
checks['selected_source_rows_checked'] = len(receipt['selected_table_cells'])
checks['selected_source_cells_checked'] = verified
checks['primary_absolute_rows'] = sum(len(re.findall(r'^(?:Base-only|Mixed) &', table_at(label), re.M)) for label in ['tab:goop-all-rollouts','tab:waterdrop110k-rollout','tab:sand-all-rollouts'])
checks['primary_cost_rows'] = sum(len(re.findall(r'^(?:Base-only|Mixed) &', table_at(label), re.M)) for label in ['tab:goop-all-cost','tab:waterdrop110k-cost','tab:sand-all-cost'])
checks['d3_prepared_cells_checked'] = len(receipt['d3_prepared_transcriptions'])
checks['d3_prepared_cells_present'] = all(c['tex'] in text for c in receipt['d3_prepared_transcriptions'])

# These three central equations must remain byte-identical to the source.
snapshot = json.loads((HERE / 'source_table_snapshot.json').read_text())
failure_source = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', snapshot['table_blocks']['tab:goop-all-failures'], re.S).group()
failure_candidate = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', table_at('tab:goop-all-failures'), re.S).group()
checks['goop_failure_tabular_exact'] = failure_source == failure_candidate
def align_block(t, label):
    pos = t.index('\\label{' + label + '}')
    lo = t.rfind(r'\begin{align}', 0, pos)
    hi = t.index(r'\end{align}', pos) + len(r'\end{align}')
    return t[lo:hi]
checks['preserved_math_blocks'] = {label: snapshot['math_blocks'][label] == align_block(text,label) for label in ['eq:nll','eq:faithful','eq:decomposition']}
checks['all_math_blocks_exact'] = all(checks['preserved_math_blocks'].values())
checks['table_count'] = len(re.findall(r'\\begin\{table\}', source))
checks['section_count'] = len(re.findall(r'\\section\{', source))
checks['approximate_whitespace_words_including_tables_math'] = len(source.split())
checks['candidate_sha256'] = receipt['candidate_sha256']
checks['rendered_or_compiled'] = False
checks['layout_status'] = 'UNVERIFIED: parent will compile integrated candidate in the same native editor; no alternate PDF or tab created.'
checks['unexpected_unresolved_refs'] = sorted(set(references) - set(labels))
checks['duplicate_labels'] = sorted(k for k,v in Counter(labels).items() if v>1)
checks['status'] = 'PASS' if all(v for k,v in checks.items() if isinstance(v,bool) and k not in ['rendered_or_compiled']) and checks['primary_absolute_rows'] == 34 and checks['primary_cost_rows'] == 34 and checks['d3_prepared_cells_checked'] == 18 else 'FAIL'
(HERE / 'source_integrity_check.json').write_text(json.dumps(checks, indent=2, sort_keys=True) + '\n')
print(json.dumps(checks, indent=2, sort_keys=True))
assert checks['status'] == 'PASS'
