# Post-expiry D3 metadata observation

This package prepares one root-owned observation. It does not issue analysis
time, admit scientific products, signal original processes, or repair an
experiment. Nothing has been executed remotely during preparation.

The original phase remains 20:18:58.570367–21:18:58.570367 UTC on 2026-10-06
(SHA256 `46a0e4c5dd7e35e1233735171eaf3df094da1e2f6236a9928c8ee911d21768df`).
Original local saved-audit session 29448 exited 130 after scoped local cleanup.
The accepted collection and its independent review remain unchanged. A local
SSH exit, DNS failure, terminal claim, or timeout does not establish remote
completion or closure.

## Exact observation

The remote helper permits only:

- Exact teal hostname and boot; UTC and monotonic observation timestamps.
- Existing `owner_started.json`, `child_registered.json`, and
  `owner_terminal.json` below the original `owners/d3_saved_array_audit` path.
- `/proc/PID/stat` and `/proc/PID/cmdline` for the 39 accepted historical PIDs
  and the three saved-operation identities, when those identities can be
  established consistently from the permitted registries.

It does not enumerate processes or directories, open any scientific product,
follow metadata-mentioned file paths, hash arrays/models, read child logs,
read GPU state, signal target processes, or invoke workers. PID reuse is checked
before reading a new process's command line. A reused PID's unrelated argv is
not read. Safe reads use no-follow directory traversal, regular-file checks,
bounded lengths, and mutation checks. Registry JSON is capped at 16 MiB because
the accepted collection's owner terminal was approximately 1.4 MiB.

The contract pins the original saved-audit release, command, deadline, and
clock sample. Historical identities come from the accepted evaluation,
qualification, and collection registries; the collection terminal is tied to
the accepted independent review. The source/runtime paths in the command are
the preserved qualified paths. This observation does not freshly hash runtime
binaries or recertify numerical execution.

Registry identity coverage and terminal presence are separate. A missing
terminal can coexist with complete identities from start and registration.
Missing, malformed, or contradictory identity evidence prevents a full closure
claim. A terminal `complete` or worker exit zero is reported only as unverified
metadata, never as accepted scientific success.

## Root command

After independent review and final manifest freeze, run the local wrapper once
with that manifest hash and a fresh absolute output directory:

```text
work/venv/bin/python -B <absolute-package>/root_observe.py
  --root-observe --package-sha256 <reviewed-manifest-sha256>
  --output <absolute-fresh-observation-directory>
  [--https-proxy http://127.0.0.1:PORT]
```

The optional proxy must already be running under root's separate authorization
and review. The wrapper neither starts nor stops it. Only the new SSH
subprocess receives the same loopback value in `HTTPS_PROXY` and `https_proxy`,
with `NO_PROXY` and `no_proxy` removed so inherited exceptions cannot bypass
the reviewed route. The parent environment is unchanged and proxy environment
values are not logged. URLs, TLS checks,
credentials, SSH configuration, and frozen scientific controls are unchanged.
Only an uncredentialed `http://127.0.0.1:PORT` value is accepted.

The wrapper invokes the original SSH alias/config directly, with one connection
attempt, an eight-second connection timeout, and SSH server-alive limits. It
passes the pinned contract on nonblocking stdin. The remote command is:

```text
/usr/bin/timeout --signal=KILL 20s
  /root/repos/AdaptGNS-cuda-20261006/.venv/bin/python -I -S -B
  -c <exact-reviewed-observer-source>
  --observe --contract-sha256 <exact-contract-sha256>
```

The helper's read budget is 12 seconds with both UTC and monotonic checks. The
local whole-observation budget is 35 seconds, including source checks,
connection, capped transport, cleanup and publication. Transport work leaves
five seconds for cleanup/publication. Cleanup may signal only this wrapper's
new local SSH process group; GNU timeout may kill only its new remote observer
group. No signal is sent to an original evaluation, owner, or worker.

These bounds assume normal OS scheduling and functioning local filesystem.
Suspension, process unresponsiveness, or a failed final write can leave no
complete observation. Root must retain the actual tool exit and files that
exist. A late or incomplete observation must not be upgraded into closure.
Every attempt must use a fresh output directory; repeating the observation
requires root's separate decision and preserves prior failed attempts.

## Reading the outcome

The wrapper writes `intent.json`, capped raw `stdout`/`stderr`, and
`transport.json`. A transport failure or missing/invalid report means
`remote_observation_unavailable`. Root must check actual tool completion,
transport leader reaping, absence of unexpected cleanup/response errors, whole-budget
preservation, and the pinned report before accepting an operational observation.
The receipt's budget field is explicitly measured before publication; the
wrapper checks the bound again after writing it. Root must preserve the actual
wrapper exit and elapsed time. Leader reaping alone does not establish closure
of a ProxyCommand descendant. If the leader already exited while a descendant
held pipes open, the wrapper closes its pipes and reports that descendant closure
is unestablished; it does not signal a possibly reused leader/group ID. Root must
handle any needed scoped transport cleanup separately. No such condition is
relabeled a successful bounded observation.

The helper performs two native observations and rechecks registry bytes and
boot. `native_closure_observed: true` requires complete saved identities,
unchanged metadata, exact host/boot and all 42 scoped PIDs absent in both
observations. Reuse has a distinct outcome: it can establish that a recorded
identity is absent, but it does not set the stricter all-PIDs-absent closure
flag. Present processes, zombies, changes during observation, permissions,
truncation, and malformed metadata remain distinct unresolved states.

Even a valid closure observation says nothing about whether the remote audit
finished successfully or whether a scientific product is acceptable. The
wrapper always leaves scientific admission and remote execution success false.

## Recovery prerequisites and remaining allocation

Before considering any further work, root needs restored authenticated routing,
an actual bounded observation on the original host/boot, complete original
identity coverage and native closure, and an explicit decision about any
preserved terminal metadata. This package has no recovery, retry, product
retrieval, or scientific-admission action.

The existing allocation cannot resume or be reanchored. The human's standing
authorization covers routine operational adjustments; this handoff introduces
no additional human approval requirement. Root must determine whether a separate
allocation fits that authorized scope and independently review its contract,
retaining this expired attempt and its censoring. It must reuse the accepted collection. If
metadata indicates that the saved audit already completed, it must not be
rerun merely to obtain a new in-budget result; later review of any preserved
artifact would need its own explicit scope and provenance rules. Summary and
paired audit were not launched. This observer cannot establish product
existence, integrity or recoverability because those paths are deliberately
outside its read scope.

The frozen D3 global cutoff is 2026-10-07 01:00:00 UTC. A separately reviewed
full 3600-second allocation could fit only if all prerequisites and its actual
start occur no later than 00:00:00 UTC. No current clock was queried during
preparation. Root's fresh clock, restored connectivity, closure evidence, and
source review must determine present fit; stale incident timestamps must not
decide admission or allocation. Scope, duration and feasibility remain for root.
No new phase, reservation, or allocation is created here.
