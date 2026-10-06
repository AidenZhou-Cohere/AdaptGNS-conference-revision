# Bounded graph-support CUDA implementation check

This entry prepares one fixed-seed (`0`), faithful-only check on the same
deterministic GB200 CUDA device. It does not reconsider the failed CPU/CUDA
validation, apply ReLU masks, admit scientific training or promote a checkpoint.
The default command imports no scientific packages and prints the declaration.

Inputs are the three exact saved training-batch NPZ files from the pinned failed
`sand_actual_data_validation_v2_report.json`, and the original metadata. No
trajectory manifest, validation data or test data is loaded. All helper files
must be adjacent to the entry, with their embedded hashes intact. The frozen
repository source pins are checked before and after execution.

The fixed plan uses the full original architecture, initialized on CUDA at
seed 0, and resets identical model bytes and CPU/CUDA RNG states for each
semantics branch. Each saved case runs old native, new base and new mix for
one first Adam update. Adam schedule step is 0; the graph RNG schedule step is
the fixed case index (small 0, median 1, large 2). The original saved noise is
reused without regeneration. Actual network inputs are captured read-only.
Every base/native node, edge, feature, output, target, loss, parameter gradient,
model state, parameter update, Adam moment and RNG tensor must pass the pinned
bytewise comparator, including signed zero. Adam metadata must also be exact.

Mix audits independently reconstruct the seed materials, Bernoulli coin,
annulus permutation, exact floor-25% budget, ordered appended suffix and full
ledger using the pinned graph primitives. The native directed cap-128/self
prefix, its features, batch partition and RNG state are checked. All gradients,
parameters and optimizer moments must be finite. At least one saved-case mix
branch must actually append edges; there is no adaptive search if it does not.

Each arm separately replays two small-batch updates at graph steps 0 and 1.
Step 0 uses the saved noise and step 1 uses the original host-noise function
at seed 0/step 1. A new-schema checkpoint is serialized after the first update,
then reloaded into a recreated model/Adam. CPU/CUDA RNG state restoration and
independent probe draws are bytewise checked. The resumed second update and
complete second checkpoint payload must equal the uninterrupted path. The
first payload is deep-copied before continuation to avoid history aliasing.
These are validation-only checkpoints with a separate purpose marker.

The fixed maximum is 15 optimizer updates. All completed branch tensors and
replay evidence are saved; guarded-update failures retain the current outputs,
graph, gradients and model/Adam state. A fresh output directory is mandatory.
Incomplete or failed evidence cannot pass. Success is `implementation_passed`,
with `scientific_training_admitted: false`; root controls any later admission.

Root execution command (paths supplied by root; not executed during preparation):

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python validate_sand_graph_support_cuda.py --execute \
  --repo /pinned/repository \
  --validation-report /pinned/sand_actual_data_validation_v2_report.json \
  --batch-dir /existing/saved/batch/directory \
  --metadata /original/metadata.json \
  --output-dir /fresh/graph_support_cuda_check --cuda-index 0 --threads 2
```

Preparation tests use width-8, one-block synthetic CPU networks and a mocked
selected-CUDA RNG. They do not execute the full model, any real dataset or CUDA.
