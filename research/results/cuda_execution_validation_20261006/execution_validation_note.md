# Synthetic CUDA execution validation — prepared for review only

`validate_cuda_execution.py` is a separate validation command. It does not edit frozen research code or load any dataset, model checkpoint or scientific result. Its only optimizer work consists of explicitly requested synthetic validation steps; checkpoint serialization stays in memory. Local checks so far cover parsing, source-identity refusal, pass/fail logic, RNG backend refusal and syntax with standard-library mocks. **The actual tensor/core path has not been executed by its author.**

The default prints the fixed contract without importing Torch or repository modules:

```sh
python validate_cuda_execution.py --describe
```

After root review, the explicit command for a verified repository copy is:

```sh
python validate_cuda_execution.py --execute --repo /path/to/AdaptGNS --cuda-index 0 --threads 2
```

Execution refuses changed bytes in five imported core modules and checks that Python imported those exact files. It requires the selected CUDA device; no CPU/MPS fallback occurs. Both objectives use the original128-wide, ten-block architecture, two-layer MLPs, float32, explicit SciPy-host graphs, identical CPU-created weights, and deterministic six-frame2D synthetic inputs. TF32/AMP/compile are disabled; Adam uses explicit `foreach=False, fused=False`. Runtime/version/actual-device evidence is recorded. This is a new runtime evaluation, not an assertion of compatibility with the original MPS/Torch runtime.

The fixed checks are:

- Exact CPU/CUDA ordered graphs and an independent brute-force order check for strict-radius boundaries, coincident cap/tie cases and separate batch graphs; exact actual training-batch graph comparison.
- CPU/CUDA acceleration, variance, target, normalization, loss and every named parameter gradient for faithful and NLL objectives. Missing or nonfinite gradients fail. Faithful mean/trunk gradients are also compared with MSE on each backend, while the faithful variance head must learn and the MSE head must receive no gradient.
- One synthetic Adam update, in-memory save/load, then a second update versus the uninterrupted branch, separately for CPU/CUDA and both objectives. Model/optimizer/RNG restoration is exact; subsequent state/output differences retain both numerical and bitwise evidence (element-byte comparison, including signed zeros). The first update must change parameters, every model parameter must have Adam state, and counters/moment shape/dtype/device must match. Global CPU/CUDA RNG state is isolated with `fork_rng`; restored streams are checked exactly. Added replay noise remains masked on type3 particles.
- Only after those pass, one warmup and three measured updates of a two-by576-node synthetic grid. Synchronized transfer, forward/backward with the internal host graph, guarded Adam, and an additional isolated graph-call timing are reported separately. This is a capacity measurement for one fixed synthetic geometry, **not representative real-data throughput or a deadline forecast**.

All numeric differences are reported independently of the decision. Forward comparison uses `atol=2e-5, rtol=2e-4`; loss `1e-5, 2e-4`; gradients `5e-5, 5e-4`; within-backend numeric replay `5e-6, 5e-5`. These are predeclared engineering tolerances for float32 reduction-order accumulation through ten graph blocks, not empirically fitted thresholds or scientific equivalence bounds. A cell passes only when finite and `abs(candidate-reference) <= atol + rtol*abs(reference)` for every element. Maximum absolute, relative, relative-L2 and tolerance-scaled errors remain visible, including near-zero relative effects. Graph/state/RNG exact checks do not use those allowances. Do not relax thresholds after a failed run; preserve it and review the cause first.

JSON goes to stdout. Exit0 means description-only or all four required sections passed; exit2 means CUDA unavailable; exit1 means validation failure. Check the JSON status, not exit0 alone. Completed objective/replay cases and timing rows remain in an exception report. Python stream diagnostics are suppressed, but native extensions may write directly to OS streams. No retry or subsequent training launch is built in. A pass does not validate material features, deployed rollouts, full-data optimization, source transfer, remote job recovery, cross-device bitwise continuation, DDP or conference claims.

Root must review the actual VM report before any study admission. In particular, inspect whether exact graph order holds, which parameters approach tolerance, whether replay is bitwise or merely numerically close, and whether the dependency/runtime differences are acceptable for the separately declared CUDA study. This preparation is not frozen or pushed.
