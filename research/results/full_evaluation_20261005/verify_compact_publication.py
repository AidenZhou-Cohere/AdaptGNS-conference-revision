"""Verify this compact evidence package without inference or third-party modules."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
manifest_path=root/'PUBLICATION_MANIFEST.json'
manifest=json.loads(manifest_path.read_text())
errors=[]
expected=set()
for item in manifest['files']:
    relative=Path(item['path']);path=(root/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        errors.append('Unsafe path: '+str(relative));continue
    if item['path'] in expected:errors.append('Repeated path: '+item['path'])
    expected.add(item['path'])
    if not path.is_file() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
        errors.append('Missing or changed: '+item['path'])
actual={path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
extra=actual-expected-{'PUBLICATION_MANIFEST.json','PUBLICATION_AUDIT.json'}
if extra:errors.append('Unexpected files: '+str(sorted(extra)))
audit_path=root/'PUBLICATION_AUDIT.json'
if audit_path.exists():
    audit=json.loads(audit_path.read_text())
    if audit['manifest_sha256']!=sha(manifest_path):errors.append('Manifest hash differs from publication audit')
else:errors.append('Publication audit missing')
print(json.dumps({'passed':not errors,'verified_files':len(expected),'errors':errors},indent=2))
raise SystemExit(bool(errors))
