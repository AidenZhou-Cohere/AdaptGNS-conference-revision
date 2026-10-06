"""Verify curated source/review bytes only; never import or execute research."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_file(root, name):
    relative = PurePosixPath(name)
    require(bool(name) and not relative.is_absolute() and
            all(part not in ('.', '..') for part in relative.parts) and
            str(relative) == name, 'Non-canonical relative path: ' + name)
    path = root.joinpath(*relative.parts)
    require(all(not item.is_symlink() for item in [path, *path.parents]
                if item != root.parent and root.parent in item.parents),
            'Symlink in selected path: ' + name)
    require(path.is_file(), 'Missing selected file: ' + str(path))
    require(path.resolve().is_relative_to(root.resolve()), 'Path escapes root: ' + name)
    return path


def check_record(path, record, label):
    require(path.stat().st_size == record['bytes'], 'Size mismatch: ' + label)
    require(digest(path) == record['sha256'], 'SHA256 mismatch: ' + label)


def verify(bundle, original_workspace=None):
    manifest_path = relative_file(bundle, 'manifest.json')
    manifest = json.loads(manifest_path.read_text())
    require(manifest['schema'] == 'adaptgns_scoped_preparation_manifest_v1',
            'Unexpected manifest schema')
    actual = set()
    for path in bundle.rglob('*'):
        require(not path.is_symlink(), 'Symlink in bundle: ' + str(path))
        if path.is_file():
            actual.add(path.relative_to(bundle).as_posix())
    require(actual == set(manifest['files']) | {'manifest.json'},
            'Manifest file set differs from package')
    for name, record in manifest['files'].items():
        check_record(relative_file(bundle, name), record, name)
    require((bundle / 'FILELIST.txt').read_text().splitlines() == sorted(actual),
            'FILELIST differs from package')
    require(manifest['file_count_excluding_manifest'] == len(manifest['files']),
            'Manifest file count mismatch')
    require(manifest['bytes_excluding_manifest'] ==
            sum(record['bytes'] for record in manifest['files'].values()),
            'Manifest byte count mismatch')

    inventory = json.loads((bundle / 'source_inventory.json').read_text())
    source_roots = ('workspace/', 'reviews/', 'review_history/')
    require(set(inventory) == {name for name in actual if name.startswith(source_roots)},
            'Copied-source inventory differs from selected package files')
    for name, record in inventory.items():
        check_record(relative_file(bundle, name), record, name)
        if original_workspace is not None:
            check_record(relative_file(original_workspace, record['source']), record,
                         'original:' + record['source'])

    prefix = 'workspace/work/deadline_research_20261005/cuda_preparation/'
    pins = json.loads((bundle / 'current_approved_pins.json').read_text())
    for name, expected in pins.items():
        require(prefix + name in inventory, 'Pin absent from inventory: ' + name)
        require(digest(relative_file(bundle, prefix + name)) == expected,
                'Current approved pin mismatch: ' + name)
    scope = json.loads((bundle / 'scope.json').read_text())
    require(set(scope['current_primary_python_entries']) ==
            {name for name in pins if name.endswith('.py')},
            'Primary entry list differs from current pins')
    require(scope['science_execution_or_test_admission_granted'] is False,
            'Preparation bundle must not grant execution admission')

    dependencies = json.loads((bundle / 'dependency_paths.json').read_text())
    predecessor_name = 'goop3d_vectorized_preparation_20261006'
    require(dependencies['predecessor_manifest'] == '../' + predecessor_name + '/manifest.json',
            'Unexpected predecessor manifest location')
    predecessor = bundle.parent / predecessor_name
    old_manifest_path = relative_file(predecessor, 'manifest.json')
    require(digest(old_manifest_path) == dependencies['predecessor_manifest_sha256'],
            'Predecessor manifest SHA256 mismatch')
    old_manifest = json.loads(old_manifest_path.read_text())
    require(set(dependencies['files']) ==
            {name.removeprefix('workspace/') for name in old_manifest['files']
             if name.startswith('workspace/')},
            'Dependency list differs from predecessor workspace closure')
    for name, record in dependencies['files'].items():
        old_name = 'workspace/' + name
        require(record['published_path'] == '../' + predecessor_name + '/' + old_name,
                'Unexpected dependency path: ' + name)
        old_record = old_manifest['files'][old_name]
        require(all(record[key] == old_record[key] for key in ('bytes', 'sha256')),
                'Dependency record differs from predecessor manifest: ' + name)
        check_record(relative_file(predecessor, old_name), record, 'predecessor:' + name)
        require(record['overlay'] == (old_name in inventory),
                'Dependency overlay flag mismatch: ' + name)

    return {
        'passed': True,
        'manifest_sha256': digest(manifest_path),
        'package_files_including_manifest': len(actual),
        'package_bytes_including_manifest': sum((bundle / name).stat().st_size for name in actual),
        'copied_source_files': len(inventory),
        'current_primary_entries': len(scope['current_primary_python_entries']),
        'current_pins': len(pins),
        'predecessor_workspace_dependencies': len(dependencies['files']),
        'original_selected_sources_rechecked': original_workspace is not None,
        'scope': 'Read-only package, selected-source and predecessor hashes; no research execution.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--original-workspace', type=Path,
                        help='Optionally rehash only exact source_inventory paths in the original workspace')
    args = parser.parse_args()
    try:
        result = verify(args.bundle.absolute(),
                        args.original_workspace.absolute() if args.original_workspace else None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({'passed': False, 'error': str(error)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
