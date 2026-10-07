"""Detached job accounting; no scientific phase or elapsed-time cutoff."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def identity(pid):
    raw=Path(f'/proc/{pid}/stat').read_text()
    fields=raw[raw.rfind(')')+2:].split()
    return {'pid':pid,'start_ticks':int(fields[19]),'pgid':os.getpgid(pid)}

def write(path,value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)

job=Path(sys.argv[1]);raw=(job/'spec.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==sys.argv[2]
spec=json.loads(raw)
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==spec['boot_id']
record={'status':'starting','owner':identity(os.getpid()),'spec_sha256':sys.argv[2],'started_utc_seconds':time.time()}
write(job/'status.json',record)
env=os.environ.copy();env.update(spec['environment']);env.pop('CUDA_VISIBLE_DEVICES',None)
with (job/'worker.stdout').open('xb') as stdout,(job/'worker.stderr').open('xb') as stderr:
    child=subprocess.Popen(spec['argv'],cwd=spec['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
    try:
        child_identity=identity(child.pid)
    except (FileNotFoundError,ProcessLookupError) as error:
        child_identity={'pid':child.pid,'identity_sample_error':str(error)}
    record.update(status='running',child=child_identity);write(job/'status.json',record)
    code=child.wait()
record.update(status='exited',returncode=code,finished_utc_seconds=time.time())
write(job/'status.json',record)
raise SystemExit(code)
