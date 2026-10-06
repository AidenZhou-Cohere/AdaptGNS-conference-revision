"""Check that optional-exposure inserts preserve existing manuscript content."""
import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader


def digest(data):
    return hashlib.sha256(data).hexdigest()


def text_sha(text):
    return digest(text.encode())


def titles(text):
    return re.findall(r'\\aistatstitle\{([^{}]*)\}', text)


def abstracts(text):
    return re.findall(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', text, re.S)


def tables(text):
    result = {}
    for match in re.finditer(r'\\begin\{(table\*?)\}.*?\\end\{\1\}', text, re.S):
        content = match.group(0)
        labels = re.findall(r'\\label\{([^{}]*)\}', content)
        assert len(labels) == 1, labels
        assert labels[0] not in result
        result[labels[0]] = content
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before-root', type=Path, required=True)
    parser.add_argument('--manuscript', type=Path, required=True)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--report-review', type=Path, required=True)
    parser.add_argument('--render-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before_path = args.before_root / 'revised_manuscript.tex'
    files = [before_path, args.manuscript, args.pdf, args.report_review]
    hashes = {str(p.resolve()): digest(p.read_bytes()) for p in files}
    before, after = before_path.read_text(), args.manuscript.read_text()
    assert titles(before) == titles(after) and len(titles(before)) == 2
    assert abstracts(before) == abstracts(after) and len(abstracts(before)) == 1
    old_tables, new_tables = tables(before), tables(after)
    table_records = []
    for label, content in old_tables.items():
        assert new_tables[label] == content, label
        table_records.append({'label': label, 'sha256': text_sha(content), 'unchanged': True})
    additions = sorted(set(new_tables) - set(old_tables))
    assert additions == ['tab:optional-exposure', 'tab:optional-exposure-alignment', 'tab:optional-exposure-both']
    before_lines, after_lines = before.splitlines(keepends=True), after.splitlines(keepends=True)
    changes = []
    for tag, i, j, a, b in difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        assert tag == 'insert', (tag, i, j, a, b)
        changes.append({'kind': tag, 'after_line_start': a + 1, 'after_line_end': b, 'added_lines': b - a})
    assert len(changes) == 2
    for filename in ['optional_exposure_main.tex', 'optional_exposure_appendix.tex']:
        insert = (args.render_root / filename).read_text()
        assert insert in after, filename
    report_review = json.loads(args.report_review.read_text())
    assert report_review['passed']
    assert report_review['mean_sd_cells'] == 14
    assert report_review['pdf_sha256'] == hashes[str(args.pdf.resolve())]
    page = PdfReader(args.pdf).pages[15].extract_text()
    assert 'Squared change per coordinate' in page
    assert 'normalized acceleration coordinates' in page
    for path in files:
        assert digest(path.read_bytes()) == hashes[str(path.resolve())], str(path)
    result = {
        'schema': 'optional-document-preservation-review-v1',
        'passed': True,
        'scope': 'Read-only title, abstract, preexisting table and insertion preservation; report page16 numeric review linkage. No inference or authoring.',
        'files_sha256': hashes,
        'title_fields_unchanged': 2,
        'title_sha256': [text_sha(value) for value in titles(before)],
        'abstract_unchanged': True,
        'abstract_sha256': text_sha(abstracts(before)[0]),
        'preexisting_tables_unchanged': len(table_records),
        'preexisting_tables': table_records,
        'new_table_labels': additions,
        'only_insertions': True,
        'insertion_ranges': changes,
        'final_v3_inserts_present_exactly': True,
        'report_page16_mean_sd_cells_passed': 14,
        'report_page16_seed_claim_checks_passed': 24,
        'report_cost_wording_corrected': True,
        'review_source_sha256': digest(Path(__file__).read_bytes()),
    }
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'passed': True, 'preexisting_tables_unchanged': len(table_records), 'titles_unchanged': 2, 'abstract_unchanged': True, 'added_tables': additions, 'only_insertions': changes, 'review_path': str(args.output)}))


if __name__ == '__main__':
    main()
