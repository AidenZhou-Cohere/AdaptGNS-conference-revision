#!/usr/bin/env python3
"""Read-only integrated-source check; intercept the canonical builder's write."""
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import os
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPECTED = '81aa5098e9ddca1bb435e7980aa76bcad444ebd5424ff7f66190be7705df0a8d'
sha = lambda b: hashlib.sha256(b).hexdigest()


def main():
    os.chdir(ROOT)
    manuscript = ROOT / 'outputs/revised_manuscript.tex'
    body_path = ROOT / 'work/manuscript_body.tex'
    main_path = ROOT / 'work/conference_experiments_main.tex'
    original = HERE / 'applied_integration_v1'
    before = (original / 'revised_manuscript.tex.before').read_bytes()
    current = manuscript.read_bytes()
    assert sha(current) == EXPECTED
    figure = (HERE / 'goop_fixed_source12_inline_v1.tex').read_bytes()
    note = (HERE / 'integration_note_v1.tex').read_bytes()
    insertion = note + b'\n' + figure + b'\n'
    body = body_path.read_bytes()
    before_body = (original / 'manuscript_body.tex.before').read_bytes()
    assert before_body.count(b'\\end{document}') == 1
    assert body == before_body.replace(b'\\end{document}', insertion + b'\\end{document}')
    main_bytes = main_path.read_bytes()
    before_main = (original / 'conference_experiments_main.tex.before').read_bytes()
    sentence = b' A fixed metadata-selected rollout illustrates the remaining physical mismatch in Figure~\\ref{fig:qualitative-goop2d}.'
    assert main_bytes.count(sentence) == 1
    assert main_bytes.replace(sentence, b'') == before_main
    assert current.count(figure) == body.count(figure) == 1
    assert current.count(note) == body.count(note) == 1
    assert current.count(insertion) == 1
    # These are the only changes to the predecessor's standalone source.
    assert current.replace(insertion, b'').replace(sentence, b'') == before
    sec = current.index(b'\\label{sec:goop-graph-exposure}')
    cost = current.index(b'\\label{tab:goop-all-cost}')
    paired = current.index(b'\\label{fig:goop-exposure-placement}')
    note_at = current.index(note)
    figure_at = current.index(figure)
    assert sec < cost < paired < note_at < figure_at < current.index(b'\\end{document}')
    assert b'\\section{' not in current[sec:figure_at]
    labels = re.findall(rb'\\label\{([^}]+)\}', current)
    old_labels = re.findall(rb'\\label\{([^}]+)\}', before)
    assert len(labels) == len(set(labels)) == 66
    assert len(old_labels) == len(set(old_labels)) == 65
    assert set(labels) - set(old_labels) == {b'fig:qualitative-goop2d'}
    refs = re.findall(rb'\\(?:ref|eqref|pageref)\{([^}]+)\}', current)
    assert not set(refs) - set(labels)
    for pattern in [rb'\\aistatstitle\{[^\n]+', rb'\\begin\{abstract\}.*?\\end\{abstract\}']:
        assert re.findall(pattern, current, re.S) == re.findall(pattern, before, re.S)

    # Execute the reviewed builder with reads recorded and writes intercepted.
    # This verifies the canonical inputs without touching the open manuscript.
    original_read_text = Path.read_text
    read_inputs = {}
    captures = {}

    def record_read(path, *args, **kwargs):
        value = original_read_text(path, *args, **kwargs)
        absolute = path.resolve()
        assert absolute.is_relative_to(ROOT)
        read_inputs[str(absolute.relative_to(ROOT))] = sha(absolute.read_bytes())
        return value

    def intercept_write(path, text, *args, **kwargs):
        assert path.resolve() == manuscript
        assert not captures
        captures['bytes'] = text.encode('utf-8')
        return len(text)

    builder = ROOT / 'work/build_manuscript.py'
    builder_bytes = builder.read_bytes()
    with patch.object(Path, 'read_text', record_read), patch.object(Path, 'write_text', intercept_write):
        exec(compile(builder_bytes, str(builder), 'exec'), {'__name__': '__main__', '__file__': str(builder)})
    assert captures['bytes'] == current
    assert manuscript.read_bytes() == current
    assert builder.read_bytes() == builder_bytes
    for name, expected in read_inputs.items():
        assert sha((ROOT / name).read_bytes()) == expected
    receipt = {
        'schema': 'goop_qualitative_applied_independent_check_v1',
        'passed': True,
        'manuscript_sha256': sha(current),
        'predecessor_manuscript_sha256': sha(before),
        'canonical_body_sha256': sha(body),
        'canonical_main_sha256': sha(main_bytes),
        'canonical_builder_sha256': sha(builder_bytes),
        'canonical_inputs_sha256': read_inputs,
        'canonical_build_matches_current_exact_bytes': True,
        'canonical_build_writes_intercepted': True,
        'only_changes_are_exact_note_figure_and_main_reference': True,
        'exact_figure_once': True,
        'placement': 'Goop appendix, after cost table and paired-seed figure, before end of document',
        'title_and_abstract_unchanged': True,
        'unique_labels': 66,
        'original_labels_retained': 65,
        'undefined_references': [],
        'native_compilation': 'Not rerun. Root integration receipt records success; final native visual inspection remains separate.',
        'source_unchanged_after_read_only_verification': True,
        'canonical_body_reconstruction': 'before.replace(end_document, note + newline + figure + newline + end_document)',
        'manuscript_figure_byte_range_half_open': [figure_at, figure_at + len(figure)],
        'manuscript_note_byte_range_half_open': [note_at, note_at + len(note)],
        'figure_sha256': sha(figure),
        'note_sha256': sha(note),
    }
    (HERE / 'applied_integration_independent_check_v1.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
