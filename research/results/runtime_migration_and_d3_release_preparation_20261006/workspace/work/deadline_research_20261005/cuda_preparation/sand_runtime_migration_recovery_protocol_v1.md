# Sand interruption recovery, version 1

This separate operational amendment implements the user's October 6 request to
replace the interrupted Sand execution. It does not issue a live release. The
original trainer, scientific protocol, scoped allocation, old attempts, locks,
logs, pointers and checkpoints remain byte-identical and retained.

The scientific study remains the same six faithful models: base and mix at
seeds 0, 1 and 2, each at 100,000 updates. The completed seed-0 pair remains part
of the cohort. The four interrupted A models resume only their own committed
50,000-update states in new output directories, with the original GPU-index
mapping: base1/0, mix1/1, base2/2, mix2/3. Uncommitted updates reported after
50,000 remain interruption evidence and are not used as recovery states.

The original `run_config` is immutable historical lineage. In particular, its
runtime identifies the original training device and its initialization field
describes the original step-zero initialization. A required separate
`runtime_migration.json` sidecar records the original and actual runtime,
original and replacement hostname, exact source/parent/release identities and
the 50,000-to-100,000 continuation. All runtime fields must match exactly
except the explicitly released old-to-new GPU UUID. Platform, package versions,
device index, device name and capability, CPU thread count, precision,
deterministic settings and environment receive no exception. The native driver
library path is the original process-local path
`/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`, with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and no `CUDA_VISIBLE_DEVICES` remapping.

The adapter imports the exact frozen numerical trainer, SHA256
`fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124`.
It invokes that trainer's unmodified data loader, CUDA configuration, model
construction, restore checks and training loop. It supplies the immutable
historical runtime to configuration construction only after independently
checking the actual runtime against the released migration mapping. No
numerical function is patched. Model tensors, simulator normalization, full
Adam state and exact counters, CPU and selected CUDA RNG, all 50,000 saved
graph/history rows and source/configuration identity must satisfy the frozen
restore predicates. Absolute steps 50,000 through 99,999 preserve the original
frame, noise, exposure coin, pair selection, learning-rate and optimizer
schedules. The original scientific protocol hash stays
`e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d`.

Before any scientific continuation, a separate root-issued replay release
requires all four original parents on the actual replacement hardware. Each
replay restores its exact 50,000 state, performs 100 updates at absolute schedule
steps 50,000 through 50,099, restores the original parent again and repeats all
100 updates. Full model,
Adam, CPU/CUDA RNG, loss, learning rate, source/frame/noise and graph-ledger
states must match exactly between repeats. Each repeat must also match the
retained original 50,100 log's exact loss, learning rate, frame identities and
particle count. No tolerance or shorter fallback is permitted. Replay produces a receipt, never a
scientific checkpoint; the subsequent training process loads the immutable
original parent again. This tests restoration and deterministic replay on the
current hardware and the available original scalar continuation. The original
50,100 model, Adam and RNG states were not saved, so the scalar comparison does
not establish full cross-device state equivalence or uninterrupted-run identity.

Only new directories and locks are created. The three origin files
`protocol.json`, `latest.json` and `checkpoint-000050000.pt` are copied byte for
byte into each new training directory; old locks and old attempts are never
moved or cleared. Every parent is rehashed before and after the attempt. The
four original directories remain read-only, including old failure evidence.
There is no automatic retry and no checkpoint overwritten in either tree.

The new owner launches only its four released children and records native
PID, parent, executable, argument and start-time identities. It may signal only
those locally spawned and rechecked children. Old-host PIDs are retained as
historical evidence and are never treated as current kill targets. Root must
provide fresh closure and idle-host evidence before each release. Every child
is reaped and failures, timeouts and unstarted jobs remain visible.

Keep the entire original training allocation of 27,404.671591931674 seconds;
do not shorten it using the 50k checkpoint or an optimistic measured rate.
Latest scientific launch is no later than October 6 at 15:05 UTC. The common
training and endpoint-audit stop remains 22:44 UTC, with the original 18,960
seconds for the complete evaluation/analysis allocation through October 7 at
04:00 UTC. Fresh clock error is at most five seconds, and the original cleanup
reserve is retained. Replay receives a fixed 600-second whole-owner quota,
including cleanup, and 180 seconds afterward for stopped-result review and the
scientific handoff. Replay stops no later than 15:02 UTC and starts no later than
14:51:55 UTC at a five-second clock bound. Failure to fit leaves the full study
incomplete; no reduced model, seed, policy, horizon or split is substituted.

The separate endpoint audit joins the original 0, 10k, 20k, 30k, 40k and 50k
checkpoint inventory with the new 60k, 70k, 80k, 90k and 100k inventory. It
checks the new 50k copy against its original, the complete final history against
the original 50k prefix, continuation-only stdout, all frozen model/Adam/history
predicates, initial/parent/final simulator identity, and the mandatory migration
sidecars and replay receipt. Pairing uses the original step-zero tensors and
the complete new 100k histories. Recovery endpoints cannot pass an unmodified
fresh-attempt interpretation: the separate provenance audit is mandatory.

The complete six-model cohort, all six policies, full H314 rollouts, common-state
and clean-validation diagnostics, failure retention, source-grid and analysis
rules remain those of the original scientific study. Root alone may admit the
complete cohort after the original seed-0 audit and new recovered-pair audits.
Report the pod replacement, lost uncommitted work, replay scope and operational
recovery in the released evidence; do not describe the run as uninterrupted.
