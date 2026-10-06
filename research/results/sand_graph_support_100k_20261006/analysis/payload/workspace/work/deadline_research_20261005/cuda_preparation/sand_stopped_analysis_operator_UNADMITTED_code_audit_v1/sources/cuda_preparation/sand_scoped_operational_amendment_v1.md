# Sand operational amendment: fixed six-model study on scoped GPUs

This is a separate prospective operational amendment to the immutable
`sand_graph_support_100k_protocol_v1.md` (SHA256
`e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d`).
The original protocol remains preserved. Only host/seed/GPU mapping, host GPU
exclusivity, scheduling admission and deadline allocations are superseded here.
The fixed plan is `sand_scoped_schedule_fixed_spec_v2.json` (SHA256
`403a96c1304341ea1899c994e0ab7474280add44ea5986ec41570e08d16c337b`).
This document and the fixed plan alone do not admit execution.

The scientific comparison remains six fresh faithful models: base and mix at
seeds0,1,2, each100000 updates. The trainer, graph policy, optimizer, schedule,
noise, architecture, checkpoints, pairing, frozen numerical source and data
are unchanged. Timing probes are never promoted. No selected interim endpoint,
seed replacement, automatic retry/resume or favorable-outcome selection is
allowed. Whole-cohort freeze still precedes test acquisition/admission and
inference. The complete policy, horizon, source grid, estimands, boundary
diagnostics, failure/missing accounting and measurement obligations remain.

HostB (aquamarine) runs base0/GPU2 and mix0/GPU3. HostA (yellow) runs base1/GPU0,
mix1/GPU1, base2/GPU2 and mix2/GPU3. OnB, root must first establish that the
existing Goop training children are reaped and the entire original Goop training
supervisor process, including its CPU endpoint audit, has exited. OnA, all
Goop-owned GPU work must first be stopped/reaped and its old
monitor must have exited. No live supervisor detach, replacement or handoff
is authorized. Concrete PID/start/argv closure receipts and fresh independent
process checks are required before the first Sand child is launched.

The new scoped supervisor requires exclusivity only on its assigned physical
GPUs. It preserves inventories across all four GPUs, verifies exact owned
PID/start/parent/argv/UUID identities, rejects foreign work on its GPUs and
observes other GPUs without controlling their processes. A separately released
Goop evaluator may coexist onB GPU0/1. Never signal any unowned process. Physical
GPU UUIDs and actual co-resident load must be retained. This amended assignment
and concurrent load were not measured in the original six-job capacity probe.

The original measured q6=.19001664200914092seconds/update and
r6=46.94376225024462seconds,35% engineering margin and991.9359722436873seconds
terminal reserve give27404.671591931674seconds per concurrent training group.
This is an engineering allowance, not a measured co-residency bound. Root's
targetA start is Oct6 14:00UTC; its latest actual child launch is15:05UTC.
Every child must satisfy the same complete-allocation fit at actual launch.
Both groups have common training/end-of-endpoint-audit stop Oct6 22:44UTC.
Owned-child cleanup begins no later than stop minus reviewed clock error minus
15seconds; its escalation remains SIGINT/5s SIGTERM/10s SIGKILL. No deadline
guarantee beyond measured process observation is claimed.

The full posttraining reserve is18960seconds:11760seconds concurrent evaluation
streams,900seconds test/source acquisition/conversion/contract preparation,
2700seconds cohort/source verification and3600seconds paired analysis/review.
Each final model retains the declared7200second full-test-rollout,
1800second same-state-validation,1800second same-state-test and900second
clean-validation allocations, plus15seconds cleanup per stage. These are fixed
whole-invocation allocations after unsuccessful complete timing, not empirical
completion forecasts. Root must separately release evaluation; this training
supervisor performs no test access or evaluation. All six streams must fit;
two-GPU/three-wave fallback is not admitted.

Compute plus analysis ends Oct7 04:00UTC, retaining04:00..08:00 for writing and
review before the unchanged08:00 no-new-work cutoff and11:59 deadline. All
required outcomes are still required for complete means; quotas may leave the
study incomplete. Preserve every numerical/implementation failure, stopped
attempt, missing case and accepted prefix under its actual scope. The original
Sand timing guards and all prior failed attempts remain part of the evidence.

The new supervisor imports unchanged scientific endpoint/history/Adam/pairing
checks from `supervise_sand_graph_support_science.py` (SHA256
`a22a59c46231c328e4939a0f021b3bb2fab9e2f93c0848bb1abd39aad2d5d305`).
It does not call that source's original host-wide runner or01:00 deadline gate.
Root releases bind this amendment, exact source/data/configuration bytes, fixed
plan, closed-Goop process evidence, reviewed clock, original paths and fresh
output. Endpoint audit applies only after every assigned child is reaped; its
result is host-level evidence, not complete-cohort/test admission. Operational
metadata belongs with reproducibility records; manuscript claims require the
actual complete scientific results.
