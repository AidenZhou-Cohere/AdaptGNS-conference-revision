# Goop prospective saved-batch and same-CUDA implementation checks

Preparation only. Root separately releases batch creation and CUDA validation. No experiment, model endpoint, CPU equivalence or deadline-fit claim follows from this document. Preserve all unsuccessful checks and old Sand CPU/CUDA failures. Never modify frozen Sand sources or promote a validation checkpoint.

Use the exact Goop trainer `train_goop_graph_support_cuda.py`, SHA256 `dd9ef01a16116f04a578bbfe9ed01774e66fdc08d60b913072a33ad211b319c1`, with root data/context admission `goop_training_admission_v1.json`, SHA256 `c2a12ef0c55b47f4c9493027450f8b51f52dbbd6043915bb09c1648b6cb9edeb`. This admission covers source compatibility only. It binds all 1,000 official training trajectories, T401/type7, exact metadata/TFRecord generation/CRC/checksums/conversion and the separately reviewed unused-context declaration. Auxiliary bytes remain preserved and checked; their positive-zero/NaN values are not the reason for omitting them.

## Prospective batches

Before any numerical outcomes, sort all training trajectories by `(particle_count, source_index)`. Use consecutive pairs at zero-based sorted ranks `(0,1)`, `(500,501)` and `(998,999)`, named small, median and large. For the first trajectory of each pair use target6; for the second use target200. Each example observes the six frames immediately preceding its target. This is a structure-selected stress check, not a representative distributional estimate.

Save precisely three numeric NPZ files containing position_sequence, particle_types, nparticles_per_example, next_positions and position_sequence_noise. Preserve actual full type7 vectors, position/noise float32 and type/count int64. Use pinned `host_noise(seed=0, step=case_index)` with case indices0,1,2. Save source trajectory ordinals/IDs, target and flattened sample indices, counts, batch checksum, selection-release checksum and complete training-data provenance before any forward call. The save stage has zero optimizer updates, no GPU model construction and no validation/test trajectory access.

The root batch-selection release must fix the exact three-case schedule derived from source structure, trainer/source/protocol hashes and data-admission hash. The next CUDA release binds the completed saved-batch report and selection release, not a batch chosen using losses. Each saved file and exact host-noise bytes are checked again before use. Root reviews any failed preparation; no automatic retry or replacement occurs.

## One-device bounded numerical check

Initialize one fresh full-architecture faithful Goop model with seed0 on the released GB200 CUDA device and preserve its initial model state. Use the pinned deterministic Torch2.13.0+cu129 stack, two CPU threads, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, no TF32/AMP/compile/DDP and unchanged Adam hyperparameters. Root supplies the physical GPU UUID and fresh process-identity check.

For each saved case, restore identical initialization/RNG and perform first-Adam-step branches `old_native`, `base` and `mix`. `old_native` is the original pinned native forward path on the same fresh Goop model and device; it is not an earlier trained checkpoint. Compare reference-native against base bytewise for graph/features, predictions/heads/targets/loss, all gradients, model/optimizer tensors and metadata, parameter updates and RNG. Mix must retain exact native prefix, fixed annulus budget/order, partition, independent RNG and ledger semantics, unchanged target, finite outputs/gradients/state/moments and ordinary faithful Adam behavior. Every branch is optimizer step0→1; the prospective case index selects only its graph-exposure RNG schedule. At least one observed mixed example must actually select optional edges, otherwise coverage is incomplete.

Use the unchanged, hash-pinned graph validator's `advance`, graph audit and comparison functions. The wrapper checks numerical trainer/run-loop AST equivalence with the frozen Sand origin, allowing only Goop names and final excluded-auxiliary integrity verification. No model, loss, gradient, numerical tolerance or ReLU intervention is introduced.

For each arm base/mix, use the small saved case for a first update, serialization, uninterrupted second update and separately restored second update. Require exact checkpoint bytes after load, model/Adam/history/RNG restoration, independent CPU/CUDA random probes, resumed tensors and final payload. The inherited seed0 step1 mix coin is unexpanded: replay restores a state affected by prior mixed exposure but does not test an actively expanded post-resume update. Report that coverage limitation explicitly.

The maximum is **15 optimizer updates**: nine first-step branches plus three updates for each of two replay arms. This is an operation bound, not a wall-clock limit. The worker has no internal timeout/alarm or scheduler; root must supervise wall time, process identity and deadline, preserve interrupted evidence, and prevent contention with the reserved Sand scientific hosts. No execution starts automatically.

## Admission and limits

Only all three complete branch comparisons, at least one positive mixed optional budget, both exact replays and final source/input rechecks yield `implementation_passed`. Undefined coverage, malformed input, nonfinite tensors or a failed comparison cannot pass. Preserve complete branch/replay evidence and exception context. This single-device seed0 check establishes bounded implementation behavior only, not convergence, all-state correctness, cross-device or CPU/CUDA equivalence.

Root may issue a distinct timing-only numerical admission after reviewing the actual complete report. It must bind the trainer, validation-source/report, saved-batch report, train-manifest and data-admission hashes. That permits only separately released six×512 capacity infrastructure. All six scientific models require another root release and complete evaluation/runtime/ledger forecast.
