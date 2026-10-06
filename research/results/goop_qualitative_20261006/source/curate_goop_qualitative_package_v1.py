#!/usr/bin/env python3
"""Allowlist the fixed Goop figure into a new package; no existing package edits."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPO = ROOT / 'outputs/AdaptGNS'
DEST = REPO / 'research/results/goop_qualitative_20261006'
PARENT_REL = 'research/results/goop2d_graph_exposure_100k_20261006'
PARENT_COMMIT = '26645fd180cbc54f1d008e13fc733a1643c003c9'
sha = lambda data: hashlib.sha256(data).hexdigest()


def tree_hashes(path):
    return {str(p.relative_to(path)): sha(p.read_bytes()) for p in sorted(path.rglob('*')) if p.is_file()}


def main():
    assert not DEST.exists(), 'Use a new version; never replace a package silently.'
    check = json.loads((HERE / 'applied_integration_independent_check_v1.json').read_text())
    assert check['passed']
    assert sha((ROOT / 'outputs/revised_manuscript.tex').read_bytes()) == check['manuscript_sha256']
    parent_before = tree_hashes(REPO / PARENT_REL)
    subprocess.run(['git', 'diff', '--exit-code', PARENT_COMMIT, '--', PARENT_REL], cwd=REPO, check=True, capture_output=True)
    parent_git_tree = subprocess.check_output(['git', 'rev-parse', f'{PARENT_COMMIT}:{PARENT_REL}'], cwd=REPO, text=True).strip()
    DEST.mkdir(parents=True)
    copies = []
    snippets = []

    def save(name, data):
        out = DEST / name
        assert not out.exists()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out

    def copy(source, name):
        data = source.read_bytes()
        save(name, data)
        copies.append({'source_path': str(source.relative_to(ROOT)), 'source_sha256': sha(data),
                       'source_bytes': len(data), 'package_path': name})

    def jsave(name, value):
        save(name, (json.dumps(value, indent=2) + '\n').encode())

    for name in ['goop_fixed_source12_inline_v1.tex', 'goop_fixed_source12_inline_direct_preview_v1.png',
                 'integration_note_v1.tex']:
        copy(HERE / name, 'presentation/' + name)
    copy(ROOT / 'outputs/revised_manuscript.tex', 'presentation/revised_manuscript.tex')
    copy(HERE / 'INTEGRATION_README_v1.md', 'notes/INTEGRATION_README_v1.md')
    for name in ['goop_fixed_source12_inline_v1.json', 'selected_inline_verification_v1.json',
                 'selected_inline_visual_review_v1.json', 'applied_integration_independent_check_v1.json']:
        copy(HERE / name, 'verification/' + name)
    copy(HERE / 'applied_integration_v1/integration_receipt.json', 'verification/root_native_integration_receipt.json')
    for name in ['render_request_v1.json', 'fetch_manifest_v1.json', 'fetch_receipt_v1.json', 'selection_receipt_v1.json']:
        copy(HERE / name, 'inputs/' + name)
    for name in ['inputs/test_manifest.json', 'inputs/jobs/base_seed0/full_rollout_test/protocol.json',
                 'inputs/jobs/mix_seed0/full_rollout_test/protocol.json']:
        copy(HERE / name, name)
    for name in ['export_qualitative_picture_v1.py', 'render_qualitative_v1.py', 'qualitative_selection_plan_v2.md']:
        copy(HERE.parent / name, 'source/' + name)
    for name in ['verify_and_preview_selected_inline_v1.py', 'verify_applied_integration_v1.py',
                 'verify_curated_package_v1.py', 'curate_goop_qualitative_package_v1.py']:
        copy(HERE / name, 'source/' + name)
    copy(HERE / 'applied_integration_v1/manuscript_body.tex.before', 'canonical/manuscript_body.before.tex')
    copy(ROOT / 'work/build_manuscript.py', 'canonical/work/build_manuscript.py')
    for name, expected in check['canonical_inputs_sha256'].items():
        assert sha((ROOT / name).read_bytes()) == expected
        if name != 'work/manuscript_body.tex':
            copy(ROOT / name, 'canonical/' + name)

    # Exact readable excerpts, with byte offsets into the copied source.
    full = (DEST / 'presentation/revised_manuscript.tex').read_bytes()
    section_start = full.index(b'\\section{Goop: graph exposure and placement}')
    note_start = check['manuscript_note_byte_range_half_open'][0]
    selected = [
        ('presentation/goop_appendix_before_qualitative.tex', 'presentation/revised_manuscript.tex', section_start, note_start),
    ]
    main_bytes = (DEST / 'canonical/work/conference_experiments_main.tex').read_bytes()
    paragraph_start = main_bytes.index(b'Among complete mixed-training policies,')
    paragraph_stop = main_bytes.index(b'\n', paragraph_start) + 1
    selected.append(('presentation/main_physical_diagnostics.tex', 'canonical/work/conference_experiments_main.tex', paragraph_start, paragraph_stop))
    for name, source, start, stop in selected:
        data = (DEST / source).read_bytes()[start:stop]
        save(name, data)
        snippets.append({'package_path': name, 'package_source': source,
                         'source_byte_range_half_open': [start, stop], 'sha256': sha(data), 'bytes': len(data)})

    raw = []
    fetch = json.loads((HERE / 'fetch_manifest_v1.json').read_text())
    for item in json.loads((HERE / 'fetch_receipt_v1.json').read_text())['files']:
        source = Path(item['path'])
        if source.suffix == '.npz' or source.name.startswith('trajectory_'):
            assert sha(source.read_bytes()) == item['sha256']
            raw.append({'source_path': str(source.relative_to(ROOT)), 'sha256': item['sha256'],
                        'bytes': item['bytes'], 'reason': 'Raw trace array or large per-step row; omitted from the compact qualitative package.'})
    jsave('source_pointers.json', {
        'parent_commit': PARENT_COMMIT, 'parent_package': PARENT_REL,
        'figure_input_hash_authority': fetch['source_audit'],
        'input_request': 'inputs/render_request_v1.json',
        'fetch_locations_and_hashes': 'inputs/fetch_manifest_v1.json',
        'omitted_raw_inputs_required_for_reexport': raw,
        'reconstruction_scope': 'The TeX source and canonical manuscript rebuild are self-contained. Re-exporting particle glyphs from arrays additionally requires the exact omitted files; no substitute run or selection is authorized.'})
    jsave('omissions.json', {
        'raw_trace_arrays_and_rows': raw,
        'archive_omitted': {'name': 'exact_selected_remote_bytes_v1.tar',
                            'sha256': json.loads((HERE / 'fetch_receipt_v1.json').read_text())['archive_sha256']},
        'excluded_categories': ['credentials', 'process inventories', 'raw trace NPZs', 'tar archives', 'new model checkpoints'],
        'outcomes_scope': 'The fixed three displayed outcomes are complete. All study failures remain in the unchanged parent package and manuscript; this package does not replace their accounting.'})
    jsave('runtime_versions.json', {'python': platform.python_version(), 'numpy': importlib.metadata.version('numpy'),
                                    'matplotlib': importlib.metadata.version('matplotlib'),
                                    'scope': 'Local numerical check and direct-PNG preview environment; not experiment timing.'})
    jsave('parent_package.json', {'commit': PARENT_COMMIT, 'path': PARENT_REL, 'git_tree': parent_git_tree,
                                  'files_sha256': parent_before, 'unchanged_during_curation': True})
    jsave('copy_provenance.json', {'schema': 'exact_goop_qualitative_copies_v1', 'exact_copies': copies, 'exact_snippets': snippets})
    save('README.md', README.encode())
    assert tree_hashes(REPO / PARENT_REL) == parent_before
    subprocess.run(['git', 'diff', '--exit-code', PARENT_COMMIT, '--', PARENT_REL], cwd=REPO, check=True, capture_output=True)
    for item in copies:
        assert sha((ROOT / item['source_path']).read_bytes()) == item['source_sha256']
    files = [{'path': str(p.relative_to(DEST)), 'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size}
             for p in sorted(DEST.rglob('*')) if p.is_file()]
    jsave('manifest.json', {'schema': 'goop_qualitative_package_manifest_v1', 'files': files,
                            'file_count_excluding_manifest': len(files), 'payload_bytes': sum(f['bytes'] for f in files),
                            'preserved_parent_commit': PARENT_COMMIT, 'manuscript_sha256': check['manuscript_sha256']})
    print(json.dumps({'package': str(DEST), 'files_excluding_manifest': len(files),
                      'payload_bytes': sum(f['bytes'] for f in files),
                      'manifest_sha256': sha((DEST / 'manifest.json').read_bytes()), 'parent_package_unchanged': True}, indent=2))


README = r'''# Fixed Goop2D qualitative rollout

This package adds one metadata-selected physical illustration to the completed
Goop2D study in `../goop2d_graph_exposure_100k_20261006` at commit
`26645fd180cbc54f1d008e13fc733a1643c003c9`. The parent package is unchanged.
It is not a new experiment or a selection by error. Seed 0, source 12
(`test:000012`, 1,083 particles) is the lower median of the complete 30-source
census sorted by initial particle count and source index. The full 1/200/395
forecast grid and all particles are retained.

## Interpretation and limits

Ground truth settles near the lower boundary while both mixed-training
forecasts retain elevated groups. The example illustrates substantial
remaining physical mismatch. Its base-trained/base and mixed-policy columns
change both training and evaluation graph; training-effect inference rests on
the quantitative paired comparison under the same random25 policy. It does
not estimate population performance or support a speedup claim.

Counts of strict metadata-box crossings at forecasts 1/200/395 are truth
0/0/0, base/base 0/553/465, mix/random25 0/0/11, and mix/cached-risk25 0/0/14.
These counts use any crossing, whereas the quantitative table uses excursions
greater than 1e-6. Crosses are clamped to the box and do not encode excursion
magnitude. Cached risk here is autonomous laggedrisk25, distinct from the
previous-observed same-state diagnostic. All three selected outcomes complete;
the parent package and manuscript preserve every study failure.

## Contents and validation

- `presentation/goop_fixed_source12_inline_v1.tex`: unchanged inline figure,
  SHA256 `2a5b34584dda3196f3070c5ba4eb072140e1999943edfc10ae611bd8a03ba6b4`.
- `presentation/goop_fixed_source12_inline_direct_preview_v1.png`: direct
  drawing of emitted glyphs with approximate fonts; it is not a native TeX proof.
- `presentation/integration_note_v1.tex`: exact surrounding interpretation.
- `presentation/revised_manuscript.tex`: exact integrated standalone source,
  SHA256 `81aa5098e9ddca1bb435e7980aa76bcad444ebd5424ff7f66190be7705df0a8d`.
- `presentation/*diagnostics.tex` and `*before_qualitative.tex`: exact source
  excerpts with offsets in `copy_provenance.json`.
- `canonical/`: exact original body before insertion, current canonical
  snippets and all builder inputs. The current body is reconstructed from
  the preserved body, exact note and exact figure without storing another
  duplicate of the large particle glyph block.
- `inputs/`, `verification/`, `source/`: exact selection, fetch, numerical,
  visual, integration and source receipts plus the unchanged exporter/renderer.

Every one of 12,996 glyphs was independently checked against the selected saved
arrays, including coordinates, rounding, inside/outside class and counts.
Source verification establishes exact insertion after the Goop paired-seed
figure, 66 unique labels, all 65 original labels retained, unchanged title and
abstract, and byte-identical canonical reconstruction. Root's copied native
integration receipt records compilation success; final native PDF visual
inspection was still pending when this package was made. No new native
compilation, model inference or remote action occurred during curation.

## Verify and reproduce

From the research-fork root:

```sh
python research/results/goop_qualitative_20261006/source/verify_curated_package_v1.py research/results/goop_qualitative_20261006
```

This checks the exact manifest file set, all hashes, copied-source provenance,
source excerpts and byte-identical canonical build entirely in memory. It
does not alter files or require particle arrays, NumPy, Matplotlib or LaTeX.

To re-export particle glyphs, first restore the exact omitted arrays and
per-step row JSONs listed in `source_pointers.json`, using the original
relative locations recorded in `copy_provenance.json` and the fetch manifest.
Restore the unchanged exporter, renderer, selection plan, request and input
protocols to their recorded relative locations as well. Verify every SHA256;
then run the command in `notes/INTEGRATION_README_v1.md` with a new output prefix.
The exporter requires NumPy; direct-PNG verification also requires Matplotlib.
Local versions are recorded in `runtime_versions.json`. Do not substitute
another source, seed, policy, checkpoint or error-selected frame.

Raw NPZs, large per-step row JSONs and the original tar are deliberately
omitted; their exact hashes and source pointers remain. Credentials and
process inventories are excluded. The package preserves paper evidence;
it does not imply conference readiness or author verification.
'''


if __name__ == '__main__':
    main()
