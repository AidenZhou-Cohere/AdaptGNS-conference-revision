"""Read-only audit of preserved Goop context arrays; no semantic admission."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


root = Path(sys.argv[1]).resolve()
report = {'schema': 'adaptgns_goop_auxiliary_census_v1', 'status': 'running',
          'scientific_training_admitted': False, 'test_accessed': False,
          'source_sha256': sha(Path(__file__)), 'splits': {}}
pins = {'train': '5ef43daf9bac961a69bd460a539891624b79c8a13f80ca851e85802197982256'}
for split in ('train', 'valid'):
    manifest_path = root / (split + '.json')
    manifest = json.loads(manifest_path.read_text())
    if split in pins:
        require(sha(manifest_path) == pins[split], 'Pinned train manifest differs')
    require(manifest['dataset'] == 'Goop' and manifest['split'] == split, 'Split identity differs')
    require('context_mean' not in manifest['metadata'] and 'context_std' not in manifest['metadata'], 'Context normalization exists')
    rows = []
    for record in manifest['records']:
        desc = record['step_context']
        path = (root / desc['path']).resolve()
        require(path.is_relative_to(root), 'Context path escapes numeric root')
        require(sha(path) == desc['sha256'] and path.stat().st_size == desc['size_bytes'], 'Context bytes differ')
        values = np.load(path, allow_pickle=False)
        require(list(values.shape) == desc['shape'] == [401, 1] and values.dtype.str == desc['dtype'] == '<f4', 'Context representation differs')
        rows.append({'id': record['id'], 'source_index': record['source_index'],
                     'sha256': desc['sha256'], 'shape': list(values.shape), 'dtype': values.dtype.str,
                     'elements': int(values.size), 'finite_count': int(np.isfinite(values).sum()),
                     'nan_count': int(np.isnan(values).sum()),
                     'positive_infinity_count': int(np.isposinf(values).sum()),
                     'negative_infinity_count': int(np.isneginf(values).sum()),
                     'unique_float32_bits_hex': [f'{int(x):08x}' for x in np.unique(values.view('<u4'))]})
    report['splits'][split] = {'manifest_sha256': sha(manifest_path), 'record_count': len(rows),
        'context_mean_absent': True, 'context_std_absent': True,
        'all_context_descriptors_and_bytes_verified': True,
        'all_context_values_nan': all(row['nan_count'] == row['elements'] for row in rows), 'records': rows}
report['status'] = 'all_preserved_auxiliary_bytes_verified'
report['semantics'] = 'No inference from NaN values; omission from model inputs requires the separate official-source semantics review.'
print(json.dumps(report, indent=2, allow_nan=False))
