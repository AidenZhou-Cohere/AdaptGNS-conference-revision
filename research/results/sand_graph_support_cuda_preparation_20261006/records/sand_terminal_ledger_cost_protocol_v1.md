# Synthetic Sand terminal-ledger publication cost v1

Infrastructure planning only. This worker does not construct a model, execute
a forward/backward pass or optimizer update, restore RNG, read trajectories, or
create a resumable training checkpoint. Root owns all execution and must issue
an exact fresh release for each assigned GPU after process-identity checks.

Use the six completed 512-update graph-support capacity probes, in their same
concurrent host A four-worker / host B two-worker allocation, and the intended
scientific checkpoint filesystem. Run three repetitions per worker. Retain
launch/stop/process/GPU/clock evidence sufficient to establish actual overlapping
I/O on each host; do not infer concurrency from the release alone. Keep both
hosts' measurements even if shared storage differs. No automatic retry.

The worker validates the fixed probe source/config/pointer/status/history and
checkpoint hash, then loads its tensor bytes with weights_only=True. It builds
100000 explicitly synthetic graph records from the probe's shape/count/coin
patterns; unique hashes and fresh strings prevent pickle identity memoization
from understating ledger size. It records 1001 synthetic scalar rows at the
scientific log cadence. These records are not observations of later training.

The fixed probe model/Adam/RNG/config tensor tree is copied to the assigned GPU.
Each measured repetition validates the full synthetic ledger, copies the tree
to CPU, serializes/fsyncs/renames a fresh .bin envelope, hashes that binary,
atomically publishes a synthetic pointer and two small synthetic status files,
and atomically writes/fsyncs the full synthetic history JSON. Per-phase timings,
file sizes, hashes, UTC intervals and peak host/GPU memory are retained. Exact
tensor-byte comparison and history hashing occur outside the measured interval.

All envelopes and synthetic pointer/status files use a separate schema and
NOT_A_TRAINING_CHECKPOINT=true. Envelopes retain actual_probe_completed_optimizer_updates=512,
nest tensors under fixed_probe_tensor_bytes, and omit normal checkpoint schema,
state_dict and completed_steps keys. No file is named latest.json or checkpoint*.pt.
The worker's operational status.json is infrastructure status only.

For a planning allowance take the maximum synthetic_publication_seconds across
all 18 repetitions, then multiply by 1.35*11. Terminal-size history JSON is
included at every repetition, conservatively applying it to all 11 publications
(step0 and 10k..100k). This is not a statistical upper bound or a complete
scientific-checkpoint bound. Root must separately reserve time for model finite
predicates, live Adam parameter/order/hyperparameter/moment predicates, live
CPU/CUDA RNG capture, and remaining training configuration/payload construction.
Factor1.35 does not establish that omitted work is covered. Root also reviews
concurrency/storage representation, ledger-size growth assumptions, preparation
cost and memory. If coverage is incomplete, leave full deadline fit undefined.
The initial checkpoint is smaller than this synthetic path; no credit is taken.

The worker stages originally CPU counters/RNG on CUDA too, so D2H bytes are
slightly conservative but do not reconstruct a live optimizer. Synthetic hash
strings do not reproduce graph selections or late-run numerical behavior.
Any preflight or execution failure remains an unsuccessful infrastructure
attempt; retain stdout/stderr and the fresh output directory, including temporary
files. The 900-second budget is checked between repetitions and is not a hard
per-I/O timeout; root supervises all process lifetimes. No script changes to any
frozen trainer, capacity worker, lifecycle module or scientific protocol.

The root release uses schema adaptgns_sand_terminal_cost_release_v1,
status admitted_for_infrastructure_io, issued_by root, scientific_training_admitted=false,
exact file hashes, explicit GPU UUID/index, shared cohort ID, expected host
workers 4 or 2, and a timezone-aware process check no more than 300 seconds old.
The template is deliberately not admitted. Runtime is pinned Torch2.13.0+cu129,
CUDA12.9, GB200, Linux, two CPU threads, strict deterministic algorithms and no
TF32. Use the exact environment in the release template; CUDA_VISIBLE_DEVICES
must be unset. No launches at or after 2026-10-07 01:00 UTC.

Concrete invocation with root-populated paths (not a launch authorization):

```text
python measure_sand_terminal_ledger_cost.py --execute
  --probe-dir COMPLETED_GRAPH_SUPPORT_512_PROBE
  --trainer train_sand_graph_support_cuda.py
  --protocol sand_terminal_ledger_cost_protocol_v1.md
  --release ROOT_WORKER_RELEASE.json
  --output-dir FRESH_SYNTHETIC_COST_DIRECTORY --cuda-index 0
```

Local review uses only tiny synthetic CPU tensors and up to 600 graph rows:

```text
work/venv/bin/python -B -m unittest discover
  -s work/deadline_research_20261005/cuda_preparation
  -p test_measure_sand_terminal_ledger_cost.py -v
```
