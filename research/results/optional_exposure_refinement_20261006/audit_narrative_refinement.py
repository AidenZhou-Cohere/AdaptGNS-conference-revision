"""Read-only correspondence checks for the accepted writing-only refinement."""
import difflib
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def tables(text):
    result = {}
    for match in re.finditer(r'\\begin\{(table\*?)\}.*?\\end\{\1\}', text, re.S):
        block = match.group(0)
        labels = re.findall(r'\\label\{([^{}]*)\}', block)
        assert len(labels) == 1 and labels[0] not in result
        result[labels[0]] = block
    return result


def keys(text, command):
    return re.findall(r'\\' + command + r'\{([^{}]*)\}', text)


def main():
    before_path = ROOT / 'narrative_refinement_before/revised_manuscript.tex'
    after_path = REPO / 'outputs/revised_manuscript.tex'
    accepted_path = ROOT / 'scientific_narrative_review_20261006.md'
    accepted_record_path = accepted_path.with_suffix('.json')
    render_root = ROOT / 'optional_exposure_render_v4'
    numeric_review_path = ROOT / 'optional_exposure_renderer_review_v4.json'
    input_paths = [before_path, after_path, accepted_path, accepted_record_path, numeric_review_path]
    input_paths += sorted(render_root.iterdir())
    initial = {str(path): sha(path) for path in input_paths if path.is_file()}
    before, after = before_path.read_text(), after_path.read_text()
    assert keys(before, 'aistatstitle') == keys(after, 'aistatstitle') and len(keys(before, 'aistatstitle')) == 2
    abstract_pattern = r'\\begin\{abstract\}(.*?)\\end\{abstract\}'
    abstracts = re.findall(abstract_pattern, before, re.S)
    assert len(abstracts) == 1 and abstracts == re.findall(abstract_pattern, after, re.S)
    old_tables, new_tables = tables(before), tables(after)
    assert len(old_tables) == len(new_tables) == 22 and old_tables == new_tables
    labels = keys(after, 'label')
    assert keys(before, 'label') == labels and len(labels) == len(set(labels))
    references = re.findall(r'\\(?:eqref|ref|pageref|autoref)\{([^{}]*)\}', after)
    assert references == re.findall(r'\\(?:eqref|ref|pageref|autoref)\{([^{}]*)\}', before)
    assert set(references) <= set(labels), set(references) - set(labels)
    citations = re.findall(r'\\cite(?:p|t)?(?:\[[^\]]*\]){0,2}\{([^{}]*)\}', after)
    assert citations == re.findall(r'\\cite(?:p|t)?(?:\[[^\]]*\]){0,2}\{([^{}]*)\}', before)
    bibitems = re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^{}]*)\}', after)
    assert bibitems == re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^{}]*)\}', before)
    cited_keys = {key.strip() for group in citations for key in group.split(',')}
    assert cited_keys <= set(bibitems), cited_keys - set(bibitems)
    math_pattern = r'\\begin\{(equation\*?|align\*?)\}.*?\\end\{\1\}'
    old_math = [m.group(0) for m in re.finditer(math_pattern, before, re.S)]
    new_math = [m.group(0) for m in re.finditer(math_pattern, after, re.S)]
    assert old_math == new_math
    main_insert = (render_root / 'optional_exposure_main.tex').read_text()
    appendix = (render_root / 'optional_exposure_appendix.tex').read_text()
    assert main_insert in after and appendix in after
    accepted_record = json.loads(accepted_record_path.read_text())
    assert accepted_record['review_sha256'] == sha(accepted_path)
    expected_fragments = [
        'Our empirical contribution is a controlled test of residual-guided graph allocation.',
        'the estimation error below must also cover that change of graph input; ordinary calibration or rank correlation does not bound this transfer.',
        r'$\Delta E=\Delta C-\Delta A$ makes $\Delta C>\Delta A$ equivalent to the',
        r'positive gap; the additional observation is $\Delta A>0$ in every seed.',
        r'Here $C$ is squared prediction change, not computational cost.',
        r'$\ell_{ia}=\tfrac12\|(p_{ia}-y_i)\oslash s\|^2$.',
        r'$N^{-1}\sum_i I_{ig}(\ell_{iL}-\ell_{iR})$.',
        r'For $r_i=(y_i-b_i)\oslash s$ and $\delta_{ia}=(p_{ia}-b_i)\oslash s$,',
        r'$A_{ia}=2r_i^\top\delta_{ia}/2$ and',
        r'$C_{ia}=\|\delta_{ia}\|^2/2$.',
        r'$\ell_{iL}-\ell_{iR}=(C_{iL}-C_{iR})-(A_{iL}-A_{iR})$.',
    ]
    assert all(fragment in after for fragment in expected_fragments)
    assert 'equal-degree subgroup' not in main_insert
    stable = {}
    for filename in ['optional_exposure_decomposition.png', 'plot_coordinates.json', 'mechanism_findings.json']:
        prior = ROOT / 'optional_exposure_render_v3' / filename
        current = render_root / filename
        assert prior.read_bytes() == current.read_bytes()
        stable[filename] = sha(current)
    numeric_review = json.loads(numeric_review_path.read_text())
    assert numeric_review['passed'] and sum(numeric_review['cells'].values()) == 92
    assert numeric_review['render_manifest_sha256'] == sha(render_root / 'render_manifest.json')
    changes = [
        {'kind': tag, 'before_lines': [a + 1, b], 'after_lines': [c + 1, d]}
        for tag, a, b, c, d in difflib.SequenceMatcher(None, before.splitlines(), after.splitlines(), autojunk=False).get_opcodes()
        if tag != 'equal'
    ]
    for path_string, initial_hash in initial.items():
        assert sha(Path(path_string)) == initial_hash
    report = {
        'schema': 'narrative-refinement-independent-review-v1', 'passed': True,
        'scope': 'Read-only writing/math/units/correspondence review. No manuscript changes, raw-outcome recomputation, inference or independent native compilation.',
        'inputs_sha256': initial,
        'title_fields_unchanged': 2, 'abstract_unchanged': True, 'abstract_sha256': digest(abstracts[0]),
        'preexisting_numeric_tables_unchanged': 22,
        'table_sha256': {label: digest(value) for label, value in old_tables.items()},
        'labels_unchanged_unique': len(labels), 'references_unchanged_resolved': len(references),
        'citation_groups_unchanged_resolved': len(citations), 'bibliography_entries_unchanged': len(bibitems),
        'display_math_environments_unchanged': len(old_math),
        'final_v4_inserts_present_exactly': True, 'numeric_renderer_cells_passed': 92,
        'numeric_seed_coordinates_passed': numeric_review['seed_coordinates'],
        'stable_v3_v4_artifacts_sha256': stable,
        'accepted_changes': {
            'empirical_contribution_first': True,
            'graph_transfer_in_cache_error': True,
            'positive_gap_identity_separate_from_positive_alignment_observation': True,
            'scalar_loss_and_elementwise_anisotropic_scale_defined': True,
            'optional_equal_degree_main_sentence_omitted_as_requested': True,
        },
        'manual_math_review': {
            'decomposition': 'With r=(y-b) elementwise-divided by s and delta=(p-b) elementwise-divided by s, ell=||delta-r||^2/2=||r||^2/2+C-A. Subtracting two actions cancels the base term, giving Delta ell=Delta C-Delta A. A=r dot delta and C=||delta||^2/2 match the saved normalized coordinate metric. All s coordinates are the saved acceleration standard deviations.',
            'cache_bound': 'The new sentence places graph-input mismatch inside the existing epsilon assumption. Triangle inequality with temporal drift and the Lipschitz max-endpoint map gives the unchanged a(epsilon+D_t)+zeta score bound and 2B factor. It introduces no measured calibration guarantee.',
            'interpretation': 'Delta C>Delta A is equivalent to the positive gap; Delta A>0 is the additional observed statement. Group contributions are unconditional, fractions retained, and noncausal/multihop caveats remain.',
        },
        'changed_ranges': changes, 'issues': [], 'review_source_sha256': sha(Path(__file__)),
    }
    output = ROOT / 'narrative_refinement_review.json'
    with output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: report[key] for key in ['passed', 'preexisting_numeric_tables_unchanged', 'labels_unchanged_unique', 'references_unchanged_resolved', 'citation_groups_unchanged_resolved', 'bibliography_entries_unchanged', 'display_math_environments_unchanged', 'numeric_renderer_cells_passed']}))
    print(str(output))


if __name__ == '__main__':
    main()
