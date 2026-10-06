# Bounded Goop-3D validation timing, version 1

This is an infrastructure experiment on the original complete Goop-3D
validation source. It admits neither scientific training nor a scientific
endpoint, checkpoint promotion, reserved test access or automated retry.
The six 512-update capacity models must first finish and pass their reviewed
complete checkpoint, source and pairing audit. Root must verify that the
capacity supervisor and all workers are stopped before timing begins.

Use the separately reviewed D3 evaluator without changes: 0.025 base radius,
1.267 radius factor, native cap128/self directed prefix, exact symmetric
quarter-budget optional suffix, three coordinates, dt0.0025, original bounds,
2M candidate-pair and5M directed-edge guards. Preserve the original data,
metadata, context, auxiliary and all numerical source bindings.

The checkpoint sample is fixed prospectively to base seed0 and mix seed0 from
the distinct 512-update capacity schema. It is not selected by loss or rollout
quality. Use GPU0 and GPU1 respectively on the root-declared dedicated host.
Run three consecutive two-process waves: full-rollout, same-state,
clean-validation. Both processes must stop and be reaped before the next wave.
The deadline worksheet receives no evaluation-overlap credit.

Full rollout evaluates all six policies at H295 on validation indices37,51,64,
in source order. These are the smallest, lower-median and largest particle
counts within the fixed30-source validation grid. There are18 planned outcomes
per arm. No successful prefix substitutes for the full horizon. Same-state
uses target7 on those same three sources, six policies plus the natural-base
timing reference, one warmup and seven cyclic timing rounds. Clean validation
uses128 evenly spaced six-frame histories within those three sources. Targets,
graphs, variance interpretation, D3 normalization and all diagnostic definitions
remain exactly those of the reviewed evaluator.

Root prospectively chose inner/whole-child limits of3600/3900 seconds for full
rollout and600/900 seconds for each other mode. The exact limits must be
declared in the root spec and all six derived releases before any timing
outcome is inspected. Root can choose different bounded limits prospectively;
the existing implementation restricts inner limits to1..7200 seconds and
larger outer limits to at most7500 seconds. A later limit change requires a
new separately identified release and preservation/review of the prior attempt.

The supervisor measures child lifetime from before process creation until
wait4 observes/reaps exit. It records raw wall time, observer-lag disclosure,
command, PID/start identity, exit code, CPU time, Linux peak RSS and every
signal. It enforces each outer limit and the2026-10-07 01:00UTC cutoff, with
identity-checked SIGINT/SIGTERM/SIGKILL at cleanup offsets0/5/10 seconds and a
15-second cleanup window. Root additionally enforces a whole-supervisor
watchdog, including preflight and artifact collection. With the proposed
limits, use6300 seconds with a30-second kill-after grace.

Only owned child groups with matching PID/start/command identity may be
signaled. Existing competing research or GPU processes prevent a wave from
starting. The wrapper's actual ancestor processes are excluded from duplicate
detection so a root timeout wrapper is not mistaken for another experiment.
An external GPU process or wrong physical assignment during timing stops the
attempt. An unreaped child keeps the lock and its exact identity for root review.

Guard-complete child invocations preserve all planned attempts and can be
followed by the remaining predeclared diagnostic waves. They are not admitted
as successful full-H timing. Nonzero exits, implementation/parity failures,
input changes, timeout, interrupt or contamination stop later waves. The
inventory always retains all six planned identities; unattempted calls remain
explicitly missing. All partial files, logs, failures and unsuccessful attempts
are retained. No invocation is automatically restarted.

Complete collection is only a receipt of stopped processes and preserved
artifacts. The scalar deadline worksheet separately requires every full-H
outcome and diagnostic frame to be complete, verifies each scalar/numeric
artifact binding, and then produces engineering estimates. It does not infer
scientific usefulness, select a training endpoint or claim conference readiness.
