#!/usr/bin/env python3
"""Minimal isolated bootstrap: literal timing arithmetic, then exec GNU timeout.

No JSON, source hashing, target imports, output publication or remote operations.
The explicit5-second bootstrap allowance is part of the existing remaining time;
ordinary OS scheduling is assumed, not guaranteed against kernel/runtime failure.
The timed owner arms a native absolute SIGKILL timer before substantive preflight.
"""
import os
import sys
import time

PYTHON='/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python'
OWNER='/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/supervise_sand_B_preflight_amendment_cpu_v1.py'


def command(deadline_ns, release, release_sha, now_ns):
    if not (type(deadline_ns) is int and 0<deadline_ns<2**63
            and type(now_ns) is int and 0<deadline_ns-now_ns<=600*10**9):
        raise ValueError('Positive amendment remaining deadline<=600seconds required')
    if not (release.startswith('/') and len(release_sha)==64 and all(c in '0123456789abcdef' for c in release_sha)):
        raise ValueError('Exact absolute release path/hash required')
    # Fifteen seconds of TERM/KILL grace plus the declared5-second bootstrap
    # allowance are subtracted from the SAME absolute remaining clock.
    seconds=(deadline_ns-now_ns)//10**9-15-5
    if seconds<=0:raise ValueError('No remaining whole-invocation time')
    return ['/usr/bin/timeout','--signal=TERM','--kill-after=15s',str(seconds)+'s',PYTHON,'-I','-S','-B',OWNER,
        '--execute','--hard-deadline-monotonic-ns',str(deadline_ns),'--outer-term-seconds',str(seconds),
        '--outer-computed-monotonic-ns',str(now_ns),
        '--release',release,'--release-sha256',release_sha]


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if not argv:return 0
    if len(argv)!=6 or argv[::2]!=['--hard-deadline-monotonic-ns','--release','--release-sha256']:
        raise ValueError('Exact literal bootstrap argv required')
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise ValueError('Bootstrap must use -I -S -B')
    args=command(int(argv[1]),argv[3],argv[5],time.monotonic_ns())
    os.execv(args[0],args)


if __name__=='__main__':raise SystemExit(main())
