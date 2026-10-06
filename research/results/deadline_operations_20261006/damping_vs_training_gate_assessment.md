# Next-experiment value assessment

The matched graph-support continuation remains the highest-priority GPU study. It directly addresses whether extra edges fail because the model was trained only on the base graph, and its paired initialization and equal-update control make its answer more interpretable than a new learned allocation rule near the deadline.

## Cheap second choice: a fixed prediction blend

A saved-output calculation can ask whether the magnitude of the prediction change, rather than its residual alignment, accounts for the observed one-step ordering. Define `p_alpha = p_base + alpha * (p_action - p_base)`. In the decomposition's normalized coordinate units, the benefit over base is exactly `alpha * A - alpha^2 * C`, where `A` is residual alignment and `C` is squared prediction change per coordinate. When the same alpha is applied to both risk and random, their paired error difference is exactly `alpha^2 * DeltaC - alpha * DeltaA`.

The inspected original no-loop results have `DeltaC > DeltaA > 0` in every model seed. The paired curve must therefore cross zero at `alpha = DeltaA / DeltaC`. A test-derived crossing or optimum is an algebraic description of inspected outcomes, not evidence that a deployable damping rule generalizes. A bounded curve from the saved summaries is almost free CPU arithmetic, but it would mostly visualize an already established identity. It should be added only if it materially clarifies the paper, with the complete predeclared curve retained rather than selecting a favorable alpha.

A more informative follow-up would freeze a single scalar or a small fixed set before new evaluation, or fit a scalar solely from training-source histories using a fixed, disjoint fitting population and criterion. Freeze the population, weighting, policy treatment, bounds and scalar before scoring evaluation histories. The repeatedly inspected validation/test sources remain exploratory reuse. Do not choose alpha from their displayed crossings, claim an independent holdout, or insert the blend into any existing frozen five-policy study.

This intervention blends final predictions; it does not scale optional messages inside the GNN. Running it autonomously requires both a base and an action forward pass at every step. Previous-observed-base-risk additionally requires its scoring history; a cached autonomous-risk implementation has a different scoring convention and warmup. Any claim about rollout stability or latency therefore needs matched full autonomous evaluation and measured runtime. Saved original no-loop predictions cannot establish the effect under the separate native self-loop convention or the 110k graph-support continuations.

## Lower priority: a learned training-only gate

A learned gate could aim at signed action benefit rather than baseline difficulty, but it adds supervised target construction, fitting choices, potential selection leakage, calibration and distribution-shift concerns. A particle-wise output gate can also alter relative motion in ways that a global scalar does not. Reliable comparison would require independent fitting data, a frozen gate, appropriate constant/random controls, failure accounting and autonomous runtime. Those costs make it lower priority than the paired graph-support continuation and a clearly labeled fixed-blend diagnostic.

Recommendation: finish and analyze the matched continuation first; use a saved-output damping curve only as a concise exploratory explanation if manuscript space and the deadline allow. Do not launch another learned-policy or GPU study now. This note proposes no parameter choice, changes no protocol, and executes no experiment.
