#!/usr/bin/env python3
"""Inert until --observe: exact D3 metadata/native observation, never science.

Must run under the reviewed external GNU timeout. No target process signals,
subprocesses, directory scans, model/array/product reads, or remote writes.
"""
import datetime as D
import hashlib
import json
import os
import socket
import stat
import sys
import time

HOST = 'aidenzhou-teal-rat-80-5c78ffbffc-g8djt'
BOOT = '4419f0f5-8f54-4cfd-9306-f4a9fc0dd5e6'
BASE = '/root/repos/AdaptGNS-cuda-20261006'
OWNER = BASE + '/goop3d_final_analysis_20261006_v1/owners/d3_saved_array_audit'
NAMES = ('owner_started.json', 'child_registered.json', 'owner_terminal.json')
HISTORICAL = (42898,42899,42911,42912,42913,42914,43568,43632,43633,43760,46156,46220,46221,46351,48776,48851,48979,49043,51739,51807,51946,52010,54713,54714,55094,55158,57372,57438,59710,59776,62415,62481,65470,65471,65472,65473,65892,65893,65928)
MAX_JSON = 16 * 1024 * 1024

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
                argv = ['/usr/bin/timeout','--signal=TERM','--kill-after=15s',str(seconds)+'s',BASE+'/.venv/bin/python','-I','-S','-B',c['owner_source']['path'],'--execute','--hard-deadline-monotonic-ns',str(hard),'--outer-term-seconds',str(seconds),'--outer-computed-monotonic-ns',str(computed),'--release',BASE+'/goop3d_final_analysis_20261006_v1/controls/d3_saved_array_audit.cpu_release.json','--release-sha256',c['release_sha256']]
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

def parse_stat(raw, expected_pid):
    text = raw.decode('ascii'); at = text.rfind(') ')
    need(at > 0 and text[:text.index(' (')].isdigit() and int(text[:text.index(' (')]) == expected_pid, 'malformed proc PID')
    fields = text[at+2:].split(); need(len(fields) >= 20, 'short proc stat')
    need(fields[0] in ('R','S','D','Z','T','t','X','x','K','W','P','I') and all(x.isascii() and x.isdigit() for x in (fields[1],fields[2],fields[3],fields[19])), 'malformed native state or integer')
    need(all(0 <= int(x) < 2**31 for x in fields[1:4]) and 0 < int(fields[19]) < 2**64, 'invalid native numeric range')
    need(str(int(fields[19])) == fields[19], 'noncanonical native start ID')
    return {'pid':expected_pid,'state':fields[0],'ppid':int(fields[1]),'pgid':int(fields[2]),'sid':int(fields[3]),'start_id':fields[19]}

class Runtime:
    """Read-only allowlist. The caller never follows registry-mentioned paths."""
    def __init__(self):
        self.start = time.monotonic(); self.start_utc = D.datetime.now(D.timezone.utc)
        self.allowed_pids = set(HISTORICAL)
    def check(self):
        elapsed = time.monotonic()-self.start
        wall = (D.datetime.now(D.timezone.utc)-self.start_utc).total_seconds()
        need(0 <= elapsed < 12 and 0 <= wall < 12 and abs(wall-elapsed) <= 5, 'observation bound/clock disagreement')
    def clocks(self):
        self.check(); return {'utc':D.datetime.now(D.timezone.utc).isoformat(),'monotonic_seconds':time.monotonic()}
    def host(self): return socket.gethostname()
    def read_raw(self, path, cap):
        self.check()
        allowed = path == '/proc/sys/kernel/random/boot_id' or path in {OWNER+'/'+name for name in NAMES} or any(path == '/proc/'+str(pid)+'/'+name for pid in self.allowed_pids for name in ('stat','cmdline'))
        need(allowed, 'outside exact metadata allowlist')
        fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
        try:
            bits = path.strip('/').split('/')
            for part in bits[:-1]:
                new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd); fd = new; self.check()
            item = os.open(bits[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            try:
                before = os.fstat(item); need(stat.S_ISREG(before.st_mode) and before.st_size <= cap, 'nonregular/oversized metadata')
                raw = bytearray()
                while True:
                    self.check(); block = os.read(item,min(65536,cap+1-len(raw)))
                    if not block: break
                    raw.extend(block); need(len(raw) <= cap, 'metadata read cap exceeded')
                after = os.fstat(item)
                need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'metadata changed while read')
                self.check(); return bytes(raw)
            finally: os.close(item)
        finally: os.close(fd)
    def boot(self): return self.read_raw('/proc/sys/kernel/random/boot_id',128).decode().strip()
    def registry(self,name): return self.read_raw(OWNER+'/'+name,MAX_JSON)
    def proc(self,pid,name): return self.read_raw('/proc/'+str(pid)+'/'+name,65536)

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

def observe(c,rt):
    need(c['schema'] == 'goop3d_post_expiry_metadata_observation_contract_v1' and c['operation'] == 'd3_saved_array_audit' and c['original_session'] == 29448, 'exact original operation required')
    need(c['host'] == HOST and c['boot_id'] == BOOT and c['owner_dir'] == OWNER and tuple(x['pid'] for x in c['historical']) == HISTORICAL, 'exact original scope required')
    start = rt.clocks(); host = rt.host(); boot = rt.boot()
    result = {'schema':'goop3d_post_expiry_metadata_observation_v1','host':host,'boot_id':boot,'started':start,'original_phase_sha256':c['original_phase_sha256'],'original_stop_utc':c['original_stop_utc'],'original_session':29448,'scientific_admission':False,'remote_execution_success_established':False,'new_analysis_clock_or_release':False,'native_closure_observed':False}
    if host != HOST or boot != BOOT:
        result.update(status='wrong_host_or_boot_cannot_establish_original_closure',finished=rt.clocks()); return result
    records = {}; metadata = {}; issues = []
    for name in NAMES:
        try:
            raw = rt.registry(name); records[name] = strict(raw)
            metadata[name] = {'status':'read','sha256':sha(raw),'bytes':len(raw)}
        except FileNotFoundError: metadata[name] = {'status':'missing'}
        except Exception as exc:
            metadata[name] = {'status':'cannot_observe','error':type(exc).__name__+': '+str(exc)}; issues.append(name)
    roles, claims, errors = registry_identities(records,c)
    complete = set(roles) == {'outer','owner','worker'} and not errors and not issues
    # Unvalidated/contradictory records cannot add arbitrary PIDs to the scope.
    saved = list(roles.values()) if complete else []
    rt.allowed_pids.update(x['pid'] for x in saved)
    expected = c['historical'] + saved
    first = {str(x['pid']):sample_native(rt,x) for x in expected}
    second = {str(x['pid']):sample_native(rt,x) for x in expected}
    stable = True
    for name, meta in metadata.items():
        try:
            raw = rt.registry(name)
            if meta['status'] != 'read' or sha(raw) != meta['sha256']: stable = False
        except FileNotFoundError:
            if meta['status'] != 'missing': stable = False
        except Exception: stable = False
    boot_final = rt.boot(); finished = rt.clocks()
    absent = complete and stable and boot_final == BOOT and all(first[k]['status'] == second[k]['status'] == 'absent' for k in first)
    identity_absence = complete and stable and boot_final == BOOT and all(first[k]['status'] in ('absent','pid_reused') and exact(first[k],second[k]) for k in first)
    result.update(status='observed_all_scoped_pids_absent' if absent else 'closure_not_established',registry_metadata=metadata,registry_identity_coverage_complete=complete,registry_errors=errors,registry_bytes_unchanged=stable,original_saved_identities=roles if complete else {},unverified_terminal_metadata_claims=claims,native_first=first,native_second=second,observed_pid_count=len(expected),all_recorded_identity_absence_observed=identity_absence,native_closure_observed=absent,final_boot_id=boot_final,finished=finished)
    return result

def main():
    need(sys.argv[1:2] == ['--observe'] and len(sys.argv) == 4 and sys.argv[2] == '--contract-sha256', 'explicit observation and contract pin required')
    need(sys.platform.startswith('linux') and sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode and not sys.flags.optimize, 'original isolated nonoptimized Linux Python required')
    rt = Runtime(); raw = sys.stdin.buffer.read(512*1024+1)
    need(len(raw) <= 512*1024 and sha(raw) == sys.argv[3], 'bounded exact contract required')
    result = observe(strict(raw),rt); rt.check()
    encoded = json.dumps(result,sort_keys=True,allow_nan=False).encode()
    need(len(encoded) <= 1024*1024, 'observation output cap'); rt.check()
    sys.stdout.buffer.write(encoded+b'\n'); sys.stdout.buffer.flush()

if __name__ == '__main__': main()
