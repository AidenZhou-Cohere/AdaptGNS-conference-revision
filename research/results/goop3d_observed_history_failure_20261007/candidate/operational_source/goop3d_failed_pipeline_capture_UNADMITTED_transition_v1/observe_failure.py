"""Bounded failure/native metadata only; no numerical products parsed or processes signaled."""
import base64
import datetime as D
import hashlib
import json
import os
import socket
import stat
import sys
import time

HOST='aidenzhou-teal-rat-80-5c78ffbffc-g8djt'
BOOT='4419f0f5-8f54-4cfd-9306-f4a9fc0dd5e6'
BASE='/root/repos/AdaptGNS-cuda-20261006'
ROOT=BASE+'/goop3d_observed_history_analysis_20261006_v1'
OWNER=ROOT+'/owners/d3_observed_history_pipeline'
NAMES=('owner_started.json','child_registered.json','owner_terminal.json')
HISTORICAL=(42898,42899,42911,42912,42913,42914,43568,43632,43633,43760,46156,46220,46221,46351,48776,48851,48979,49043,51739,51807,51946,52010,54713,54714,55094,55158,57372,57438,59710,59776,62415,62481,65470,65471,65472,65473,65892,65893,65928,66518,66519,66558)
PRODUCTS=tuple(ROOT+'/'+n for n in ('audit.json','summary.json','arithmetic_check.json','pipeline_receipt.json'))
LOGS=tuple(OWNER+'/'+n for n in ('child.stdout','child.stderr','publication_failed.json'))+tuple(ROOT+'/'+n for n in ('pipeline.failure.json','audit.failure.json','audit.progress.jsonl'))
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

def identity(row):
    need(type(row) is dict, 'native identity missing')
    need(all(type(row.get(k)) is int and 0 < row[k] < 2**31 for k in ('pid','ppid','pgid','sid')), 'invalid native IDs')
    need(type(row.get('start_id')) is str and row['start_id'].isascii() and row['start_id'].isdigit() and 0 < int(row['start_id']) < 2**64, 'invalid start ID')
    need(str(int(row['start_id'])) == row['start_id'], 'noncanonical start ID')
    need(type(row.get('argv')) is list and row['argv'] and all(type(x) is str for x in row['argv']), 'invalid argv')
    return {k:row[k] for k in ('pid','ppid','pgid','sid','start_id','argv')}

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

def registry_identities(records, c):
    """Only projected registry identity fields are used; no product semantics."""
    roles = {}; claims = {}; errors = []
    def add(role, row):
        row = identity(row)
        need(role not in roles or exact(roles[role], row), 'conflicting '+role+' identity')
        roles[role] = row
    try:
        for name, d in records.items():
            need(type(d) is dict, 'registry root must be an object')
            if name in ('owner_started.json','owner_terminal.json'):
                need(d.get('release_sha256') == c['release_sha256'], 'original release differs')
                guard = d['native_hard_guard']; timing = d['native_outer_timing']
                need(guard['absolute_deadline_ns'] == c['hard_deadline_monotonic_ns'] and guard['clock'] == 'CLOCK_MONOTONIC' and guard['signal'] == 'SIGKILL', 'original guard differs')
                hard = c['hard_deadline_monotonic_ns']; computed = timing['computed_monotonic_ns']; seconds = timing['term_seconds']
                need(type(computed) is int and type(seconds) is int and seconds > 0 and seconds == (hard-computed)//10**9-20, 'original timeout arithmetic differs')
                argv = ['/usr/bin/timeout','--signal=TERM','--kill-after=15s',str(seconds)+'s',BASE+'/.venv/bin/python','-I','-S','-B',c['owner_source']['path'],'--execute','--hard-deadline-monotonic-ns',str(hard),'--outer-term-seconds',str(seconds),'--outer-computed-monotonic-ns',str(computed),'--release',ROOT+'/controls/pipeline.cpu_release.json','--release-sha256',c['release_sha256']]
                need(exact(d['outer_timeout_identity']['argv'],argv) and exact(d['owner_identity']['argv'],argv[4:]), 'original native owner argv differs')
                add('outer',d['outer_timeout_identity']); add('owner',d['owner_identity'])
                if name == 'owner_started.json': need(exact(d['original_clock_sample'],c['clock_sample']), 'original clock sample differs')
                else:
                    child = d.get('child')
                    claims = {k:d.get(k) for k in ('status','failure','child_native_absent','publication_utc','publication_monotonic')}
                    claims['unverified_worker_exit_code'] = child.get('exit_code') if type(child) is dict else None
                    if child is not None:
                        need(type(child) is dict and child['registered'] is True and type(child['pid']) is int and exact(child['owner_identity'],d['owner_identity']) and exact(child['command'],c['worker_argv']) and child['pid'] == child['identity']['pid'], 'terminal original child differs')
                        add('worker',child['identity'])
            else:
                need(name == 'child_registered.json' and d['registered'] is True and type(d['pid']) is int and d['pid'] == d['identity']['pid'] and exact(d['command'],c['worker_argv']), 'original registration differs')
                add('owner',d['owner_identity']); add('worker',d['identity'])
        if 'worker' in roles: need(exact(roles['worker']['argv'],c['worker_argv']), 'original worker argv differs')
        if set(roles) == {'outer','owner','worker'}:
            a,b,d = (roles[x] for x in ('outer','owner','worker'))
            need(len({a['pid'],b['pid'],d['pid']}) == 3 and not ({a['pid'],b['pid'],d['pid']} & set(HISTORICAL)), 'original roles overlap')
            need(b['ppid'] == a['pid'] and d['ppid'] == b['pid'] and a['pgid'] == b['pgid'] == d['pgid'] and a['sid'] == b['sid'] == d['sid'], 'original recorded ancestry differs')
    except (KeyError,TypeError,ValueError,OverflowError) as exc:
        errors.append(type(exc).__name__+': '+str(exc))
    return roles, claims, errors

class Runtime:
    def __init__(self,c):
        self.start=time.monotonic();self.start_utc=D.datetime.now(D.timezone.utc);self.c=c
        self.allowed_pids=set(HISTORICAL)|{os.getpid(),os.getppid()}
        self.allowed=set(c['remote_source_runtime_bindings'])|{ROOT+'/controls/analysis_phase.json',ROOT+'/controls/pipeline.cpu_release.json','/proc/sys/kernel/random/boot_id'}|set(PRODUCTS)|set(LOGS)|{OWNER+'/'+n for n in NAMES}
    def check(self):
        elapsed=time.monotonic()-self.start;wall=(D.datetime.now(D.timezone.utc)-self.start_utc).total_seconds()
        phase_wall=(D.datetime.now(D.timezone.utc)-D.datetime.fromisoformat(self.c['phase_started_utc'])).total_seconds();phase_mono=time.monotonic()-self.c['phase_host_monotonic']
        need(0<=elapsed<12 and 0<=wall<12 and abs(elapsed-wall)<=5 and 0<=phase_wall<3540 and 0<=phase_mono<3540 and abs(phase_wall-phase_mono)<=5,'existing phase or12second metadata budget expired')
    def clocks(self):self.check();return {'utc':D.datetime.now(D.timezone.utc).isoformat(),'monotonic_seconds':time.monotonic()}
    def open(self,path):
        self.check();need(path in self.allowed or any(path=='/proc/'+str(p)+'/'+n for p in self.allowed_pids for n in ('stat','cmdline')),'exact metadata/hash allowlist only')
        directory=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
        try:
            bits=path.strip('/').split('/')
            for bit in bits[:-1]:
                nxt=os.open(bit,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory);os.close(directory);directory=nxt
            fd=os.open(bits[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
        finally:os.close(directory)
        return fd
    def read(self,path,cap,retain=True):
        fd=self.open(path)
        try:
            before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and before.st_size<=cap,'bounded regular metadata or opaque product')
            parts=[];hasher=hashlib.sha256();size=0
            while True:
                self.check();block=os.read(fd,min(65536,cap+1-size));self.check()
                if not block:break
                size+=len(block);need(size<=cap,'read size cap');hasher.update(block)
                if retain:parts.append(block)
            after=os.fstat(fd);keys=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            need(keys(before)==keys(after),'file changed during observation')
            return b''.join(parts) if retain else {'status':'opaque_hash_only','bytes':size,'sha256':hasher.hexdigest(),'numeric_content_parsed':False}
        finally:os.close(fd)
    def proc(self,pid,name):return self.read('/proc/'+str(pid)+'/'+name,65536)
    def metadata(self,path,cap,opaque=False):
        try:
            if opaque:return self.read(path,cap,False)
            raw=self.read(path,cap);return {'status':'read','bytes':len(raw),'sha256':sha(raw),'raw_base64':base64.b64encode(raw).decode()}
        except FileNotFoundError:return {'status':'missing'}
        except Exception as e:return {'status':'cannot_observe','error':type(e).__name__+': '+str(e)}

def validate_contract(c):
    need(c['schema']=='goop3d_failed_pipeline_metadata_contract_v1' and c['original_pipeline_session']==33822 and c['original_pipeline_exit_code']==1,'actual failed original pipeline contract')
    need(c['phase_sha256']=='fb5e6c098f83ebf344ce00a9bb7e31a8b135788d9850ad6d713a9283a83358f2' and c['release_sha256']=='5c40c1d082a500883c16f0fa4922d4c0ed6578db87583031d1bae4c5af5c4004','unchanged original phase/release')
    need(tuple(x['pid'] for x in c['historical'])==HISTORICAL and c['host']==HOST and c['boot_id']==BOOT,'exact42historical contract and host')
    need(c['no_original_success_collector_dispatched'] is True and c['scientific_admission'] is False and c['new_analysis_clock_or_release'] is False,'failure observation only')

def observe(c,rt):
    validate_contract(c);started=rt.clocks();need(socket.gethostname()==HOST and rt.read('/proc/sys/kernel/random/boot_id',128).decode().strip()==BOOT,'exact original host/boot')
    for path,pin in c['remote_source_runtime_bindings'].items():need(rt.read(path,16<<20,False)['sha256']==pin,'exact source/runtime binding '+path)
    phase=rt.read(ROOT+'/controls/analysis_phase.json',1<<20);release=rt.read(ROOT+'/controls/pipeline.cpu_release.json',1<<20)
    need(sha(phase)==c['phase_sha256'] and sha(release)==c['release_sha256'],'exact original issued control bytes')
    pid,parent=os.getpid(),os.getppid();selfstat=parse_stat(rt.proc(pid,'stat'),pid);parentstat=parse_stat(rt.proc(parent,'stat'),parent)
    argv=lambda p:[v.decode('utf-8',errors='surrogateescape') for v in rt.proc(p,'cmdline').rstrip(b'\0').split(b'\0')]
    ownargv,parentargv=argv(pid),argv(parent)
    need(os.readlink('/proc/'+str(pid)+'/exe')=='/usr/bin/python3.12' and os.readlink('/proc/'+str(parent)+'/exe')=='/usr/bin/timeout' and os.path.realpath(BASE+'/.venv/bin/python')=='/usr/bin/python3.12','original verified runtime and own timeout executable')
    need(selfstat['ppid']==parent and selfstat['pgid']==parentstat['pgid'] and selfstat['sid']==parentstat['sid'],'own new observer timeout ancestry')
    need(parentargv[:3]==['/usr/bin/timeout','--signal=KILL','20s'] and parentargv[3:]==ownargv and ownargv[:5]==[BASE+'/.venv/bin/python','-I','-S','-B','-c'] and sha(ownargv[5].encode())==c['observer_source_sha256'] and ownargv[6:]==['--observe','--contract-sha256',sha(json.dumps(c,indent=2,sort_keys=True).encode()+b'\n')],'exact own observer/timeout argv and contract')
    records={};metadata={}
    for n in NAMES:
        item=rt.metadata(OWNER+'/'+n,256<<10);metadata[n]=item
        if item['status']=='read':
            try:records[n]=strict(base64.b64decode(item['raw_base64']))
            except Exception as e:item['status']='invalid_json';item['parse_error']=type(e).__name__+': '+str(e)
    roles,claims,errors=registry_identities(records,c);complete=set(roles)=={'outer','owner','worker'} and not errors and all(item['status'] in ('read','missing') for item in metadata.values())
    expected=c['historical']+(list(roles.values()) if complete else []);rt.allowed_pids.update(x['pid'] for x in expected)
    first={str(x['pid']):sample_native(rt,x) for x in expected}
    logs={path:rt.metadata(path,256<<10) for path in LOGS}
    products={path:rt.metadata(path,256<<20,True) for path in PRODUCTS}
    second={str(x['pid']):sample_native(rt,x) for x in expected}
    stable=True
    for n,item in metadata.items():
        again=rt.metadata(OWNER+'/'+n,256<<10)
        stable=stable and again==item
    boot=rt.read('/proc/sys/kernel/random/boot_id',128).decode().strip();finished=rt.clocks()
    closure=complete and stable and boot==BOOT and all(first[k]['status']==second[k]['status']=='absent' for k in first)
    result={'schema':'goop3d_failed_pipeline_metadata_observation_v1','phase_sha256':c['phase_sha256'],'release_sha256':c['release_sha256'],'original_pipeline_session':33822,'original_pipeline_exit_code':1,'started':started,'finished':finished,'host':HOST,'boot_id':BOOT,'final_boot_id':boot,'registry_metadata':metadata,'unverified_terminal_claims':claims,'registry_errors':errors,'registry_identity_coverage_complete':complete,'registry_bytes_unchanged':stable,'original_new_owner_identities':roles if complete else {},'native_first':first,'native_second':second,'observed_pid_count':len(expected),'all42_plus_actual_new_owner_pids_absent_twice':closure,'failure_logs_raw':logs,'product_sizes_opaque_hashes_only':products,'new_observer_identity':{**selfstat,'argv':ownargv},'new_observer_timeout_identity':{**parentstat,'argv':parentargv},'original_success_collector_dispatched':False,'collector_absence_basis':'root-bound failed original pipeline and local absence of original product_collection intent; no collector identity invented','termination_cause_established':False,'scientific_admission':False,'successful_product_numeric_content_parsed':False,'original_collection_or_arrays_opened':False,'new_analysis_clock_or_release':False,'target_processes_signaled':False,'remote_files_written':False}
    return result

def main():
    need(sys.argv[1:2]==['--observe'] and len(sys.argv)==4 and sys.argv[2]=='--contract-sha256','explicit observer and contract')
    need(sys.platform.startswith('linux') and sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode and not sys.flags.optimize,'isolated original Python flags')
    raw=sys.stdin.buffer.read(512*1024+1);need(len(raw)<=512*1024 and sha(raw)==sys.argv[3],'bounded original contract bytes');c=strict(raw);rt=Runtime(c)
    result=observe(c,rt);encoded=json.dumps(result,sort_keys=True,allow_nan=False).encode();need(len(encoded)<=(8<<20),'bounded metadata output');rt.check();sys.stdout.buffer.write(encoded+b'\n');sys.stdout.buffer.flush()

if __name__=='__main__':main()
