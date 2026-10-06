# Goop 2D: device-scoped evaluation supervision

Prospective operational amendment, October 6, 2026. No new scientific outcome
has selected this amendment. The author requested using aquamarine's spare GPUs
for Sand. This document changes resource ownership and its reporting, not the
scientific Goop training or evaluation question.

The six Goop 100k models, original paired initialization/noise/frame schedules,
training source and configuration, final-checkpoint rule, data splits, all six
policies, full H395 horizon, same-state and clean-validation schedules, numerical
and physical diagnostics, and paired estimands remain unchanged. The original
v2 scientific protocol and all old wrappers/records remain preserved byte-for-byte.
The original four evaluation stages retain whole-invocation quotas of 7200,
1800, 1800 and 900 seconds, 15-second owned cleanup per stage, and the original
October 7 01:00 UTC computation/analysis deadline. A shared machine may complete
fewer items within these fixed allowances; no quota or outcome denominator is
silently expanded or reduced in response.

The running training supervisors have host-wide foreign-GPU guards. They will
not be modified, suspended, replaced, or given a filtered view of GPU processes.
Sand may start only after the relevant Goop training children have all exited
and been reaped, with the frozen supervisor's persisted endpoint-audit state
and native identities checked. Existing training remains uninterrupted.

The separate `supervise_goop_evaluation_gpu_scoped_v3.py` will run Goop evaluation
under the same A4/B2 model-to-device mapping as v2. Its released owned GPU indices
are 0–3 on yellow and 0–1 on aquamarine. GPUs2/3 on aquamarine may be used by the
separately released Sand study. The supervisor records complete host GPU
inventories; it refuses foreign work on its owned devices and refuses any owned
child appearing on the wrong device. Work on unassigned devices remains visible
but is never adopted, signalled, or terminated. Cleanup remains limited to exact
owned child identities. There is no live parent handoff or device remapping.
Each actual release binds this amendment, its supervisor, the unchanged evaluator
and inputs, all physical GPU UUIDs and the original commands. The existing v2
release preparation may supply the original command skeleton; a separately
reviewed v3 release must explicitly bind the changed supervisory source/schema
and GPU scope before execution.

All raw GPU observations are saved before scope validation, including rejected
observations. Polling is not a continuous activity trace. Runtime is labelled
shared-host operational measurement: additional Sand work can contend for CPU,
I/O, memory bandwidth or other host resources despite distinct GPUs. Raw durations,
network calls, graph work, failure prefixes and quota events remain reportable,
but neither these durations nor historical isolated/preparation timings establish
an isolated speed advantage. Any later isolated timing experiment needs its own
fixed schedule and measurements; it must not overwrite these records.

The v3 collector retains the original complete population and paired statistics.
Each timeout, never-started item, uncommitted current item, numerical/resource
failure and successful full-horizon result retains its distinct status. Missing
required observations leave affected full-population means undefined. The
amendment and scope history are supporting reproducibility material, not a
proposed main-paper contribution.
