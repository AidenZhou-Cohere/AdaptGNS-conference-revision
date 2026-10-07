#!/usr/bin/env python3
"""Create an explicit successor plan from preserved v1 plus sealed integration entries.

No directory discovery, science, copying to fork, Git or network operation occurs.
The seal provides every new path/hash/size and any deliberately replaced entry.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'sha256': digest, 'bytes': path.stat().st_size}


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--seal-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    out = args.output_dir.resolve()
    need(out.is_relative_to(workspace), 'output outside workspace')
    need(pin(args.seal)['sha256'] == args.seal_sha256, 'seal pin differs')
    seal = json.loads(args.seal.read_bytes())
    need(seal['schema'] == 'goop3d_completion_sealed_publication_successor_v1', 'seal schema differs')
    prior_path = workspace / seal['prior_plan']['source']
    need(pin(prior_path) == {k: seal['prior_plan'][k] for k in ('bytes', 'sha256')}, 'prior plan changed')
    plan = json.loads(prior_path.read_bytes())
    entries = {e['destination']: e for e in plan['files']}
    # Explicit replacements preserve the old entry as a history destination.
    for replacement in seal['replace']:
        old = entries.pop(replacement['destination'])
        history = replacement['preserve_old_as']
        need(history not in entries, 'duplicate preserved destination')
        entries[history] = dict(old, destination=history, role='superseded_curation_history')
    for entry in seal['add']:
        need(entry['destination'] not in entries, 'new entry duplicates existing destination')
        need(pin(workspace / entry['source']) == {k: entry[k] for k in ('bytes', 'sha256')}, 'sealed new input changed: ' + entry['source'])
        entries[entry['destination']] = entry
    seal_destination = 'publication_tools/input_seal.json'
    need(seal_destination not in entries, 'seal destination already exists')
    entries[seal_destination] = {'source': str(args.seal.resolve().relative_to(workspace)),
        'destination': seal_destination, 'role': 'explicit_successor_input_seal', **pin(args.seal)}
    for entry in entries.values():
        need(pin(workspace / entry['source']) == {k: entry[k] for k in ('bytes', 'sha256')}, 'preserved input changed: ' + entry['source'])
    def binding(name):
        entry = entries[name]
        return {'destination': name, 'sha256': entry['sha256']}
    def included_json(name):
        return json.loads((workspace / entries[name]['source']).read_bytes())
    final_presentation = seal['final_presentation']
    review = included_json(final_presentation['review'])
    need(review['status'] == 'passed_narrow_v4_scope_review', 'v4 presentation review absent')
    need(review['renderer_sha256'] == entries[final_presentation['renderer']]['sha256'], 'v4 renderer/review mismatch')
    need(review['receipt_sha256'] == entries[final_presentation['receipt']]['sha256'], 'v4 receipt/review mismatch')
    need(review['combined_appendix_sha256'] == entries[final_presentation['combined_appendix']]['sha256'], 'v4 appendix/review mismatch')
    # Follow the reviewed presentation lineage back to admitted actual products.
    prior_editorial = included_json(final_presentation['prior_editorial_review'])
    original_presentation = included_json(final_presentation['original_presentation_review'])
    need(review['prior_editorial_review_sha256'] == entries[final_presentation['prior_editorial_review']]['sha256'], 'v3/v4 review lineage differs')
    need(prior_editorial['prior_presentation_review_sha256'] == entries[final_presentation['original_presentation_review']]['sha256'], 'v2/v3 review lineage differs')
    need(original_presentation['actual_product_review_sha256'] == plan['numerical_product_gate']['review']['sha256'], 'actual-product/presentation lineage differs')
    receipt = included_json(final_presentation['receipt'])
    need(receipt['source_summary_sha256'] == plan['numerical_product_gate']['products']['summary.json']['sha256'], 'presentation summary differs')
    for name in final_presentation['published_receipt_members']:
        destination = str(Path(final_presentation['receipt']).parent / name)
        need(receipt['files'][name] == {k: entries[destination][k] for k in ('bytes', 'sha256')}, 'presentation receipt member differs: ' + name)
    need(len(set(final_presentation['published_receipt_members'])) == len(final_presentation['published_receipt_members']), 'duplicate presentation member')
    integration = seal['integration']
    integration_review = included_json(integration['review'])
    compile_receipt = included_json(integration['compile_receipt'])
    integration_manifest = included_json(integration['manifest'])
    manuscript_entry = entries[integration['manuscript']]
    need(integration_review['status'] == 'passed_focused_observed_manuscript_integration_review', 'focused manuscript review absent')
    need(integration_review['manuscript_sha256'] == manuscript_entry['sha256'], 'review/manuscript pin differs')
    need(integration_review['integration_manifest_sha256'] == entries[integration['manifest']]['sha256'], 'review/integration manifest pin differs')
    need(integration_review['root_compile_transcription_sha256'] == entries[integration['compile_receipt']]['sha256'], 'review/native compile receipt pin differs')
    need(integration_review['prior_editorial_v4_review_sha256'] == entries[final_presentation['review']]['sha256'], 'manuscript/v4 lineage differs')
    need(integration_review['actual_product_review_sha256'] == plan['numerical_product_gate']['review']['sha256'], 'manuscript/product lineage differs')
    need(compile_receipt['source_sha256'] == manuscript_entry['sha256'] and compile_receipt['native_compile_result']['kind'] == 'success', 'native compilation not successful for this manuscript')
    need(compile_receipt['pdf_export_performed'] is False, 'unexpected PDF export scope')
    for original_path, destination in integration['canonical_sources'].items():
        need(integration_manifest['outputs'][original_path] == {k: entries[destination][k] for k in ('bytes', 'sha256')}, 'canonical integration source differs: ' + original_path)
    need(integration['status_documents'], 'final status-document selection is still pending')
    for destination in integration['status_documents']:
        need(destination in entries, 'status document absent from sealed selection')
    plan.update(created_utc=datetime.now(timezone.utc).isoformat(),
                prior_plan_sha256=seal['prior_plan']['sha256'],
                files=sorted(entries.values(), key=lambda e: e['destination']),
                presentation_lineage={name: binding(destination) for name, destination in final_presentation.items() if name != 'published_receipt_members'},
                integration_seal_sha256=args.seal_sha256,
                integration=seal['integration'], pending=seal['pending'], scope=seal['scope'])
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'plan.json'
    with target.open('xb') as stream:
        stream.write(encode(plan))
    print(json.dumps({'status': 'explicit_successor_plan_prepared', 'plan': str(target), **pin(target),
                      'files': len(entries), 'bytes_selected': sum(e['bytes'] for e in entries.values())}, sort_keys=True))


if __name__ == '__main__':
    main()
