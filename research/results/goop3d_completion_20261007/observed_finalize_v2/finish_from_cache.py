#!/usr/bin/env python3
"""Finish the observed-history audit from all2568 immutable passing checkpoints.

Loads the original pinned runner's metadata preparation and merge functions,
then the unchanged summarizer and a separately versioned exact-variance checker.
Never calls a row-array audit or launches workers. Final NPZ checks hash opaque
bytes only. Original cache, failed check, old checker and pins remain unchanged.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RUNNER_SHA = '8bcdb32f7d36a9a4c6f3bcfdb96eecbe36c5b419a855b35473448627e7e7f76b'
SOURCE_PINS_SHA = '365c8071abe5016a9092813d57177705386f6568efe03ce1893b640d4b8e7c92'
PROTOCOL_SHA = 'd2c2815cfee9b1f0eb4aff854eb4acb6c00b0ec0b6f232e308bde9487078f167'
CACHE_INDEX_SHA = 'f5346b964bd71e4060137e729fe17de84c1facb5d246274b14449d1acceb5b65'
PRIOR_FAILURE_SHA = '1ee80845881059ead8a3b8702db353200b49ae4e0c11817530f54351fbd2a993'
OLD_CHECKER_SHA = 'd266da3d5a437586c407aba20898374f62d459b2b9c2980232d241e4a56bf5ad'


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ordinary(path):
    path = Path(path)
    need(path.is_file() and all(not p.is_symlink() for p in (path, *path.parents)), 'ordinary canonical input required: ' + str(path))
    return path


def load(path, pin, name):
    need(sha(ordinary(path)) == pin, 'source bytes differ: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ReadOnlyCache:
    """Read committed records without mkdir, index changes, or fallback auditing."""
    def __init__(self, root, index, identity, runner):
        self.root, self.index, self.identity, self.runner = Path(root), index, identity, runner
        self.bindings = {str(self.root / 'checkpoint_index.json'): CACHE_INDEX_SHA}
        need(index['schema'] == 'goop3d_observed_row_checkpoints_v1' and index['identity'] == identity,
             'original cache source/input/runtime identity differs')
        need(not index['failed'] and len(index['completed']) == 2568, 'all2568 original passing row checkpoints required')

    def read(self, key, task_sha):
        need('/' not in key and key not in ('', '.', '..'), 'invalid row key')
        entry = self.index['completed'].get(key)
        need(entry is not None and entry['task_sha256'] == task_sha, 'exact expected cached task missing or changed')
        path = ordinary(self.root / 'rows' / (key + '.json'))
        need(path.stat().st_size <= 16 << 20, 'bounded cache row required')
        raw = path.read_bytes()
        need(hashlib.sha256(raw).hexdigest() == entry['sha256'], 'cached row bytes differ')
        record = self.runner.strict(raw)
        need(record['key'] == key and record['task_sha256'] == task_sha
             and record['identity_sha256'] == self.runner.digest(self.runner.encode(self.identity)), 'cached row binding differs')
        self.bindings[str(path)] = entry['sha256']
        return record['result']


def execute(args):
    runner_path = args.original_runner_dir / 'run_observed_resumable.py'
    runner = load(runner_path, RUNNER_SHA, '_original_observed_cache_runner')
    pins_path, protocol_path = args.original_runner_dir / 'source_pins.json', args.original_runner_dir / 'protocol.json'
    need(sha(ordinary(pins_path)) == SOURCE_PINS_SHA and sha(ordinary(protocol_path)) == PROTOCOL_SHA, 'original source/protocol pins differ')
    pins = runner.strict(pins_path.read_bytes())
    need(pins['runner_sha256'] == RUNNER_SHA
         and pins['frozen_files_sha256']['check_goop3d_observed_history_summary_v1.py'] == OLD_CHECKER_SHA, 'original failed checker/runner binding differs')
    common, audited, arithmetic, summarizer, _ = runner.load_science(args.frozen_dir, pins)
    checker_path = HERE / 'check_observed_exact_variance_v2.py'
    checker = load(checker_path, args.checker_sha256, '_observed_exact_variance_checker_v2')
    source_pin = sha(__file__)
    need(source_pin == args.finisher_sha256, 'exact reviewed finisher pin required')
    cache = args.cache.resolve()
    need(cache == args.cache and cache.is_dir() and not cache.is_symlink(), 'canonical original cache required')
    index_path, prior_failure_path = cache / 'checkpoint_index.json', cache / 'attempts/000001/failure.json'
    need(sha(ordinary(index_path)) == CACHE_INDEX_SHA and sha(ordinary(prior_failure_path)) == PRIOR_FAILURE_SHA,
         'exact original passing cache and failed check required')
    index = runner.strict(index_path.read_bytes())
    prior_failure = runner.strict(prior_failure_path.read_bytes())
    need(prior_failure['error_type'] == 'ValueError' and prior_failure['error'] == 'independent arithmetic differs', 'expected original checker failure differs')
    # The previous driver owns this same kernel lock; no lock-file deletion or cache writes.
    lock_path = ordinary(cache / '.lock')
    output = args.output.resolve()
    protected = [cache, args.frozen_dir.resolve(), args.original_runner_dir.resolve(), HERE,
                 Path(common.QUEUE).resolve(), Path(common.COLLECTION).parent.resolve(), Path(common.OUTPUT).resolve()]
    need(args.output == output and all(output != p and not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected),
         'fresh finalization output must be separate from every original input')
    with lock_path.open('rb') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        try:
            checks, collection, full_accounting, schedules, ref, sources, allowed, prepared, tasks = runner.prepare(
                common, audited, arithmetic, args.frozen_dir, pins)
            identity = {'source_pins_sha256': SOURCE_PINS_SHA, 'protocol_sha256': PROTOCOL_SHA,
                        'collection_sha256': common.COLLECTION_SHA,
                        'original_source_manifest_sha256': collection['source_manifest_sha256'],
                        'python_version': platform.python_version(), 'python_executable_sha256': sha(Path(sys.executable).resolve()),
                        'numpy_version': audited.np.__version__, 'frozen_files_sha256': pins['frozen_files_sha256']}
            store = ReadOnlyCache(cache, index, identity, runner)
            expected = {key: pin for key, pin, _ in tasks}
            need(len(tasks) == len(expected) == 2568 and set(expected) == set(index['completed']), 'complete exact observed task grid differs')
            need(all(index['completed'][key]['task_sha256'] == pin for key, pin in expected.items()), 'original cached task input binding differs')
            need({p.stem for p in (cache / 'rows').glob('*.json')} == set(expected), 'cache row file set differs from committed grid')
            print(json.dumps({'state': 'verified_complete_cache_metadata', 'cached_rows': len(tasks), 'array_audit_calls': 0}), flush=True)
            result = runner.merge(common, arithmetic, checks, full_accounting, schedules, ref, sources,
                                  allowed, prepared, store, identity, pins)
            need(len(store.bindings) == 2569, 'merge must consume every cached row exactly within the full grid')
            summary = summarizer.summarize(result)
            checked = checker.verify(result, summary)
            print(json.dumps({'state': 'exact_variance_check_passed', 'checked_model_metric_aggregates': checked['checked_model_metric_aggregates'],
                              'array_audit_calls': 0, 'opaque_rehash_files': len(allowed),
                              'opaque_rehash_bytes': sum(v['bytes'] for v in allowed.values())}), flush=True)
            # The archived NPZ files are byte-hashed only; never decoded or scientifically recalculated.
            for path, entry in allowed.items():
                common.read_bound(path, entry['sha256'], common.CAP_ARCHIVE if path.endswith('.npz') else common.CAP_JSON,
                                  expected_bytes=entry['bytes'], retain=False)
            common.read_bound(common.COLLECTION, common.COLLECTION_SHA, common.CAP_COLLECTION,
                              expected_bytes=common.COLLECTION_BYTES, retain=False)
            for path, pin in sources.items():
                common.read_bound(path, pin, common.CAP_JSON, retain=False)
            for path, pin in store.bindings.items():
                need(sha(ordinary(path)) == pin, 'original cached input changed during finalization')
            need(sha(prior_failure_path) == PRIOR_FAILURE_SHA, 'original failure changed')
            for name, pin in pins['frozen_files_sha256'].items():
                need(sha(args.frozen_dir / name) == pin, 'frozen source changed')
            for path, pin in ((runner_path, RUNNER_SHA), (pins_path, SOURCE_PINS_SHA), (protocol_path, PROTOCOL_SHA),
                              (checker_path, args.checker_sha256), (Path(__file__), source_pin)):
                need(sha(path) == pin, 'finalizer/original source changed')
            lineage = {'finalizer_revision': 2, 'original_cache_index_sha256': CACHE_INDEX_SHA,
                       'original_failed_check_sha256': PRIOR_FAILURE_SHA, 'old_checker_source_sha256': OLD_CHECKER_SHA,
                       'corrected_checker_source_sha256': args.checker_sha256, 'finisher_source_sha256': source_pin,
                       'saved_row_array_audit_calls': 0, 'comparison_tolerance_unchanged': True,
                       'correction': 'Exact rational sample variance avoids false nonzero SD for identical large seed means.'}
            for product in (result, summary, checked):
                product.update(lineage)
            for product in (summary, checked):
                product.update(successor_protocol_sha256=PROTOCOL_SHA, successor_source_pins_sha256=SOURCE_PINS_SHA,
                               prior_failed_phase_sha256=runner.PRIOR_FAILED_PHASE)
            audit_pin = runner.atomic_write(output / 'audit.json', result)
            summary['audit_sha256'] = audit_pin
            summary_pin = runner.atomic_write(output / 'summary.json', summary)
            checked.update(audit_sha256=audit_pin, summary_sha256=summary_pin)
            check_pin = runner.atomic_write(output / 'arithmetic_check.json', checked)
            failure_copy_pin = runner.atomic_write(output / 'retained_original_failure.json', prior_failure)
            need(failure_copy_pin == PRIOR_FAILURE_SHA, 'retained original failure must be byte-identical')
            receipt = {'schema': 'goop3d_observed_cache_finalization_v2',
                       'status': 'all2568_cached_rows_merged_and_exact_variance_check_passed',
                       'required_observed_cells': 2568, 'required_all_cells': 4728, 'cache_rows_consumed': len(tasks),
                       **lineage, 'original_runner_sha256': RUNNER_SHA, 'original_source_pins_sha256': SOURCE_PINS_SHA,
                       'original_protocol_sha256': PROTOCOL_SHA, 'collection_sha256': common.COLLECTION_SHA,
                       'cache_row_sha256': store.bindings, 'original_arrays_decoded': 0,
                       'opaque_original_files_rehashed': len(allowed), 'opaque_original_bytes_rehashed': sum(v['bytes'] for v in allowed.values()),
                       'products_sha256': {'audit.json': audit_pin, 'summary.json': summary_pin,
                                           'arithmetic_check.json': check_pin, 'retained_original_failure.json': failure_copy_pin},
                       'elapsed_seconds': time.monotonic() - started, 'arbitrary_elapsed_cutoff': False,
                       'original_cache_and_failed_attempt_unchanged': True, 'scientific_admission_requires_independent_review': True}
            receipt_pin = runner.atomic_write(output / 'completion.json', receipt)
            print(json.dumps({'state': receipt['status'], 'completion_sha256': receipt_pin,
                              'products_sha256': receipt['products_sha256']}), flush=True)
        except BaseException as error:
            runner.atomic_write(output / 'failure.json', {'error_type': type(error).__name__, 'error': str(error),
                'traceback': traceback.format_exc(), 'saved_row_array_audit_calls': 0,
                'original_cache_and_failed_attempt_preserved': True})
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true')
    for name in ('original-runner-dir', 'frozen-dir', 'cache', 'output'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--finisher-sha256')
    p.add_argument('--checker-sha256')
    args = p.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'inert_cache_only_finalizer', 'expected_cached_rows': 2568,
                          'array_audit_calls': 0, 'arbitrary_elapsed_cutoff': False}))
        return
    need(all(getattr(args, name) is not None for name in ('original_runner_dir', 'frozen_dir', 'cache', 'output',
                                                        'finisher_sha256', 'checker_sha256')), 'explicit inputs and reviewed source hashes required')
    execute(args)


if __name__ == '__main__':
    main()
