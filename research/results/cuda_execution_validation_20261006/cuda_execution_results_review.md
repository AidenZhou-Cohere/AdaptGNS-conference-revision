# Saved CUDA execution results: independent report review

All declared synthetic checks pass in the saved full-model report and four indexed environment-smoke reports. This review checked **57,514 structural/scalar conditions and 9,874 tensor-difference records**, without rerunning a model or recomputing raw tensor differences. All five input files remained byte-identical during review. Earlier unsuccessful environment smoke is a separate retained artifact and is not superseded by these passes.

The full-model run used Linux ARM64, NVIDIA GB200 capability10.0, PyTorch2.13.0+cu129, NumPy2.5.3, SciPy1.18.1 and PyG2.8.0.post1, with float32, TF32 disabled and two CPU threads. The checked architecture has **1,624,979 parameters in189 tensors**, width128 and ten message-passing blocks. Independently enumerated names/shapes match all189 gradients for both objectives and the189 model plus567 Adam-state tensors in each of four replay cases. Three graph fixtures and both actual parity-batch edge lists are bitwise exact; targets and normalization also agree bitwise.

| CPU→CUDA check | Faithful maximum tolerance ratio | NLL maximum tolerance ratio |
|---|---:|---:|
| Acceleration | 0.015246 | 0.018824 |
| Variance | 0.001607 | 0.001574 |
| Loss | 0.000660 | 0.000971 |
| All parameter gradients | 0.533319 | **0.962727** |

Ratios must be≤1. The worst gradient in both objectives is processor block7's first edge-MLP weight matrix. Maximum absolute differences in that matrix are2.7066e−5/4.8333e−5. **NLL is close to its predeclared acceptance boundary**, despite passing; this is not a generous margin. Near-zero relative errors are retained in the raw report and do not replace the absolute-plus-relative gate. Tolerances were not changed.

The faithful-versus-MSE mean-gradient control covers183 tensors per backend: CPU is bitwise exact; CUDA is numerically within tolerance (maximum ratio0.533195). The MSE variance head receives no gradient and the faithful head has nonzero gradient. All checkpoint load tensors, CPU/CUDA RNG streams, optimizer coverage, scalar counters and moment placement pass exact checks. CPU continuation is bitwise exact. CUDA continuation passes numerical replay with maximum state ratios0.029650/0.049336 and output ratios0.022687/0.041719 for faithful/NLL, but **511/513 of756 state tensors differ bitwise**. Deterministic CUDA algorithms were disabled; these reports do not establish exact CUDA replay or identify the cause of the residual differences.

The synthetic timing used two576-node grids and22,120 directed edges. After one warmup, guarded updates took0.119586,0.114714 and0.101734seconds; median **0.114714seconds**. Median forward/backward including its internal host graph was0.049977seconds; gradient guards plus Adam0.064547seconds. A separately timed additional graph call had median0.016044seconds and is excluded from update time. These are synchronized measurements for one synthetic geometry, not real-data throughput, an accuracy/runtime comparison or a complete-cohort forecast.

Each of the four environment-smoke reports passes finite outputs/gradients and exact RNG replay/restoration with no optimizer update or missing core import. Each file internally identifies one visible device0; mapping the four files to distinct physical GPUs relies on the operator's launch records. No private connection identifiers are included here.

This establishes bounded synthetic execution evidence only. Real-data optimization, material features, autonomous rollouts, crash recovery, convergence and conference claims remain unvalidated by these reports. Input hashes, unchanged core identities, exact coverage, thresholds and compact measurements are recorded in `cuda_execution_results_review.json`.
