# ReLU mechanism review and timing-only release recommendation

**Recommend a separate deterministic native-CUDA capacity measurement only.** Both strict CPU–CUDA gradient validations remain `validation_failed`; scientific training remains blocked on a separately frozen native-study decision and the complete deadline forecast. No tolerance or scientific endpoint is changed.

Reviewed the saved scalar report `sand_relu_kink_v1_report.json` (SHA256 `d3b9c0a1c6953dc26ba3a00f5a2425231cf5c131984a1b404364d73b7bb9bbe3`), all six case/objective combinations and all four branches. No tensor artifacts, trajectories, checkpoints, models or CUDA execution were used in this review. The frozen methodology memo was read unchanged.

| Case | Objective | ReLU sign disagreements | Natural Adam opposite signs | Natural update gate | Post-Adam prediction gate |
|---|---|---:|---:|---|---|
| small | faithful | 0 | 0 | pass | pass |
| small | nll | 0 | 0 | fail | pass |
| median | faithful | 19 | 13 | fail | pass |
| median | nll | 19 | 15 | fail | fail |
| large | faithful | 17 | 6 | fail | fail |
| large | nll | 17 | 8 | fail | fail |

Crossing preactivations have absolute magnitude at most **1.6726553e-6**. The own-mask sham preserves natural CUDA gradients and pre-update outputs bit for bit in all six comparisons. The diagnostic CPU-mask intervention removes all gradient tolerance failures and gradient sign flips, and all post-Adam output failures. These controls support near-zero ReLU-mask disagreement as the dominant measured mechanism for the larger differences.

**The adverse update evidence remains material.** Natural CPU/CUDA first-Adam updates disagree in sign at 0/0/13/15/6/8 coordinates; their largest absolute difference is **1.9890070e-4**. Five of six update comparisons and three post-update prediction comparisons fail the unchanged tolerances. Even the CPU-mask control retains a small/NLL update failure at two coordinates, maximum **6.1132014e-6**, despite zero ReLU flips and passing gradients/post-update outputs. The intervention therefore does not explain every numerical update difference or establish backend equivalence.

This evidence is sufficient to investigate native CUDA throughput under its own explicitly declared implementation. It does not show that the discrepancies are harmless, prove equality of long training trajectories, or pass either prior reproduction gate. Common-mask backpropagation remains a diagnostic counterfactual and must never enter infrastructure or scientific training.

Root should freeze the timing-only 4+2-wave, 512-update protocol with natural deterministic CUDA operations, unchanged objectives/Adam and paired host schedules. Use fresh infrastructure directories and retain every failure and checkpoint. Admit no scientific run from those checkpoints. A complete six-run capacity result and a conservative forecast covering full evaluation, diagnostics and verification must precede any separate scientific cohort admission; raw test remains reserved.

Exact per-case scalar summaries, report hashes, unchanged tolerances and release conditions are in `sand_relu_timing_release_review.json`.
