# Bounded ReLU/Adam mechanism diagnostic

This diagnostic does not revise the failed v1/v2 strict CPU/CUDA validation.
It uses every original saved small/median/large batch, both faithful and NLL,
the same initialization seed 1729, exact metadata, pinned core and deterministic
v2 runtime. No source trajectory is loaded or batch selected from outcomes.

Each case has four branches from identical initial weights and empty Adam state:

1. Original CPU forward/backward.
2. Original CUDA forward/backward.
3. CUDA forward with its own recorded ReLU signs used for backward: the sham.
4. CUDA forward with the CPU-recorded ReLU signs used for backward: the control.

All ReLU calls are identified by module path, invocation ordinal and exact shape.
The sign mask is `x > 0`, including false at zero. The custom autograd function
returns `torch.relu(x)` and changes only backward to `where(mask, gradient, 0)`.
It is a diagnostic counterfactual: where signs differ, the forced derivative is
not the derivative of the ordinary CUDA forward function. It is never proposed
as a production or scientific-training modification.

The sham must preserve CUDA pre-update outputs bitwise and reproduce gradients,
the first Adam update and post-update outputs under the original tolerances.
Without that check, a control improvement could come from hook/kernel changes.
Post-update evaluations occur after ordinary module forwards have been restored.

The report records every ReLU layer's shape, positive/zero counts, minimum
preactivation magnitudes and CPU/CUDA differences. Every sign-disagreement flat
index and its two preactivation values are saved. The fixed guard is 300 million
preactivation elements per branch and one million disagreement entries; exceeding
it fails the diagnostic rather than selecting a subset.

For each branch, numeric artifacts retain pre-update outputs, every parameter
gradient, first-Adam state/update, optimizer state and post-update outputs. JSON
comparisons include failing gradient/update indices, all sign-change counts,
strict opposite signs, absolute/relative differences and unchanged forward,
gradient and replay tolerance checks. Sign detection uses comparisons rather
than multiplication so tiny opposite gradients cannot disappear by underflow.

A reduction in CPU/CUDA gradient discrepancies under the CPU-mask control,
combined with a successful sham and bitwise unchanged CUDA forward, supports a
ReLU-mask contribution conditional on the CUDA activations/loss. It does not
exclude other rounding/reduction differences, establish semantic equivalence,
or automatically admit training. Adam's normalization makes actual update and
post-update output comparisons necessary; a small fraction of failing gradients
alone is insufficient. Every comparison failure remains in the result.

The final status is only `diagnostic_complete` or `diagnostic_error`. Completion
requires all six case/objective combinations and all four branches, but never
requires the numerical comparisons to pass. The original failed report and all
source/input hashes, including this diagnostic's source and frozen cores, are
reverified. Root owns interpretation and any distinct native-CUDA study admission.

Local verification: ten tiny CPU/network or pure helper tests passed in 1.27
seconds. No repository model, actual batch array, CUDA device, source data,
network or remote execution was used during preparation.
