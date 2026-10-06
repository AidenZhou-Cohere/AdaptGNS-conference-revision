# CUDA execution validation (October 6, 2026)

This family establishes bounded **synthetic execution checks**, not a new particle-simulation experiment or evidence of accuracy, generalization, calibrated risk, or conference readiness. The completed MPS100k study remains unchanged. No real trajectory, historical checkpoint or test outcome was used here.

Four NVIDIA GB200 devices (Blackwell capability10.0, about184.3GiB as reported by PyTorch) each passed the tiny tensor, finite-gradient and CPU/CUDA RNG restoration checks. One device then passed full128-width/10-message-block CPU-versus-CUDA checks for ordered SciPy-host graphs, faithful/NLL predictions and loss, every parameter gradient, faithful/MSE shared-gradient semantics, serialized model/Adam/RNG restoration and two-update replay on both CPU and CUDA. The validator checks complete Adam-state coverage and exact step counters. All predeclared numeric tolerances and per-tensor measurements are retained. CPU replay was byte-exact; CUDA replay passed numerical tolerances but was not byte-exact. The largest CPU/CUDA gradient tolerance ratio was0.96273 for NLL (1.0 is the fixed limit), so this is not evidence of a large numerical margin. All thresholds remain unchanged.

Runtime: Linux ARM64, Python3.12.3, PyTorch2.13.0+cu129, NumPy2.5.3, SciPy1.18.1, PyG2.8.0.post1. The complete environment lock is included. This differs from the original MPS/PyTorch2.14.1 runtime and is a separate backend boundary. TF32 and AMP were disabled; graph construction used explicit `scipy_host`; Adam foreach/fused were disabled. Each published core source hash was checked before and after execution.

## Preserved initial failure and environment correction

The initial GPU0 smoke returned `not_run`/exit2 because CUDA was unavailable; GPUs1–3 were consequently unattempted. A direct initialization diagnostic reported CUDA error803. The image placed compatibility `libcuda575.57.08` ahead of its installed native driver/kernel580.159.03. A process-local library path selecting the matching native driver corrected initialization. No image, system driver, frozen simulator source, or tolerance was changed. The original unsuccessful report is retained under `environment_smoke_before_driver_fix`; all four subsequent reports are in `environment_smoke_after_driver_fix`.

For this tested Linux ARM64 image, the execution environment was:

```sh
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export CUDA_VISIBLE_DEVICES=0
python validate_cuda_execution.py --execute --repo /path/to/AdaptGNS --cuda-index 0 --threads 2
```

Other machines must use their actual matching driver paths. The source hash, fixed core pins, CLI scope and predeclared tolerances are in the launch freeze and execution note. Default invocation only describes the validator; tensors require `--execute`. The environment probe's default inventories dependencies only; `--synthetic-smoke` explicitly runs its tiny checks.

## Timing scope

The full check also records one warmup and three measured synthetic1152-node updates, including host graph construction, transfers, per-parameter finite checks, synchronization and Adam. Its median guarded update is approximately0.115s. This is **not real-data throughput or a cohort completion forecast**. Complete dataset admission, actual-data validation, fixed scientific protocols and measured training/evaluation forecasts remain necessary before scientific runs.

Ten root CPU/mock validation tests passed; the full synthetic GPU validation and four tiny GPU checks then passed. Independent source review fixed empty/partial Adam-state acceptance and exact-counter/byte-equality issues before execution. Previous preparations and this execution are explicitly distinct; `execution_validation_local_checks.json` describes the earlier pre-execution check.
