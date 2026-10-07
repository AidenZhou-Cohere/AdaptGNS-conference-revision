"""Read-only closure observation of the two exact previous metadata observers."""
import datetime as D, hashlib, json, os, socket, stat, sys, time
HOST='aidenzhou-teal-rat-80-5c78ffbffc-g8djt'
BOOT='4419f0f5-8f54-4cfd-9306-f4a9fc0dd5e6'
def need(ok, message):
    if not ok: raise ValueError(message)

def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key'); result[key] = value
        return result
    def reject(value): raise ValueError('nonfinite JSON')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)

def exact(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)

def sha(raw): return hashlib.sha256(raw).hexdigest()

def parse_stat(raw, expected_pid):
    text = raw.decode('ascii'); at = text.rfind(') ')
    need(at > 0 and text[:text.index(' (')].isdigit() and int(text[:text.index(' (')]) == expected_pid, 'malformed proc PID')
    fields = text[at+2:].split(); need(len(fields) >= 20, 'short proc stat')
    need(fields[0] in ('R','S','D','Z','T','t','X','x','K','W','P','I') and all(x.isascii() and x.isdigit() for x in (fields[1],fields[2],fields[3],fields[19])), 'malformed native state or integer')
    need(all(0 <= int(x) < 2**31 for x in fields[1:4]) and 0 < int(fields[19]) < 2**64, 'invalid native numeric range')
    need(str(int(fields[19])) == fields[19], 'noncanonical native start ID')
    return {'pid':expected_pid,'state':fields[0],'ppid':int(fields[1]),'pgid':int(fields[2]),'sid':int(fields[3]),'start_id':fields[19]}

def sample_native(rt, expected):
    pid = expected['pid']
    try: first = parse_stat(rt.proc(pid,'stat'),pid)
    except FileNotFoundError: return {'pid':pid,'status':'absent'}
    except Exception as exc: return {'pid':pid,'status':'cannot_observe','error':type(exc).__name__+': '+str(exc)}
    try:
        if first['start_id'] != expected['start_id']:
            second = parse_stat(rt.proc(pid,'stat'),pid)
            need(second['start_id'] == first['start_id'], 'PID changed during reuse observation')
            return {'pid':pid,'status':'pid_reused','expected_start_id':expected['start_id'],'observed_start_id':second['start_id'],'unrelated_argv_not_read':True}
        raw = rt.proc(pid,'cmdline'); second = parse_stat(rt.proc(pid,'stat'),pid)
        need(second['start_id'] == first['start_id'] and all(second[k] == first[k] for k in ('ppid','pgid','sid')), 'native identity changed during read')
        argv = [x.decode('utf-8',errors='surrogateescape') for x in raw.rstrip(b'\0').split(b'\0')] if raw else []
        second['argv'] = argv
        mismatches = [k for k in ('ppid','pgid','sid','argv') if k in expected and not exact(second[k],expected[k])]
        return {'pid':pid,'status':'present','identity':second,'recorded_field_mismatches':mismatches,'zombie':second['state'] == 'Z'}
    except FileNotFoundError: return {'pid':pid,'status':'changed_during_observation'}
    except Exception as exc: return {'pid':pid,'status':'cannot_observe','error':type(exc).__name__+': '+str(exc)}
class Runtime:
    def __init__(self,c):self.c=c;self.tick=time.monotonic();self.wall=D.datetime.now(D.timezone.utc)
    def check(self):
        mono=time.monotonic();now=D.datetime.now(D.timezone.utc);elapsed=mono-self.tick;wall=(now-self.wall).total_seconds();phase_wall=(now-D.datetime.fromisoformat(self.c['phase_started_utc'])).total_seconds();phase_mono=mono-self.c['phase_host_monotonic']
        need(0<=elapsed<5 and 0<=wall<5 and abs(elapsed-wall)<=5 and 0<=phase_wall<3540 and 0<=phase_mono<3540 and abs(phase_wall-phase_mono)<=5,'existing phase or five-second observation budget expired')
    def read(self,path):
        self.check();need(path=='/proc/sys/kernel/random/boot_id' or path in {'/proc/'+str(p)+'/'+n for p in (79394,79395) for n in ('stat','cmdline')},'exact two-PID metadata allowlist')
        with open(path,'rb') as f:raw=f.read(65537)
        need(len(raw)<=65536,'bounded native metadata');self.check();return raw
    def proc(self,pid,name):return self.read('/proc/'+str(pid)+'/'+name)
    def stamp(self):self.check();return {'utc':D.datetime.now(D.timezone.utc).isoformat(),'monotonic_seconds':time.monotonic()}
def main():
    need(sys.argv[1:2]==['--observe'] and len(sys.argv)==4 and sys.argv[2]=='--contract-sha256','exact explicit observation entry')
    need(sys.platform.startswith('linux') and sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode and not sys.flags.optimize,'isolated original Python flags')
    raw=sys.stdin.buffer.read(262145);need(len(raw)<=262144 and sha(raw)==sys.argv[3],'exact bounded contract');c=strict(raw)
    need(c['schema']=='goop3d_failed_metadata_observer_closure_contract_v1' and c['original_capture_session']==51372 and c['original_capture_transport_sha256']=='c202135c685f2fbda594bc843e47cab1ad8594c53db9070ba5eebceff87b16bc','exact prior capture')
    need(c['phase_sha256']=='fb5e6c098f83ebf344ce00a9bb7e31a8b135788d9850ad6d713a9283a83358f2' and c['phase_started_utc']=='2026-10-06T23:31:18.235220+00:00' and c['phase_host_monotonic']==8316075.273839823,'original phase origin')
    expected=c['expected'];need([x['pid'] for x in expected]==[79394,79395] and all(x['start_id']=='831917341' for x in expected),'exact two original native identities')
    need(expected[0]['ppid']==39 and expected[1]['ppid']==79394 and all(x['pgid']==x['sid']==79394 for x in expected),'original recorded ancestry')
    rt=Runtime(c);started=rt.stamp();need(socket.gethostname()==HOST and rt.read('/proc/sys/kernel/random/boot_id').decode().strip()==BOOT,'same original host/boot')
    first={str(x['pid']):sample_native(rt,x) for x in expected};second={str(x['pid']):sample_native(rt,x) for x in expected};boot=rt.read('/proc/sys/kernel/random/boot_id').decode().strip();finished=rt.stamp()
    absent=boot==BOOT and all(first[str(x['pid'])]['status']==second[str(x['pid'])]['status']=='absent' for x in expected)
    result={'schema':'goop3d_failed_metadata_observer_closure_v1','original_capture_session':51372,'original_capture_transport_sha256':c['original_capture_transport_sha256'],'contract_sha256':sha(raw),'host':HOST,'boot_id':BOOT,'final_boot_id':boot,'phase_sha256':c['phase_sha256'],'started':started,'finished':finished,'first':first,'second':second,'both_exact_prior_metadata_identities_absent_twice':absent,'scope':'only previous metadata observer79395 and timeout79394; original45 not reread','scientific_admission':False,'new_analysis_clock_or_release':False,'termination_cause_established':False,'target_processes_signaled':False,'logs_products_or_arrays_opened':False,'remote_files_written':False}
    encoded=json.dumps(result,sort_keys=True,allow_nan=False).encode();need(len(encoded)<=131072,'bounded report');rt.check();sys.stdout.buffer.write(encoded+b'\n');sys.stdout.buffer.flush()
if __name__=='__main__':main()
