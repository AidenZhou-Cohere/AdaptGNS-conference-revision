# Self-message and cap controls

At fixed WaterDrop checkpoints and histories, vary self-messages and receiver capping while retaining current-base and preceding-observed-base risk controls. The six factorial cases are base with cap128/loops0, cap128/loops1, uncapped/loops0 and uncapped/loops1, plus dense with cap128/loops0 and cap128/loops1. Additional uncapped policy comparisons evaluate each loop convention. Use the fixed 128 validation and 297 test histories separately, paired across graph cases, objectives and seeds.

This is an observed-history graph-input intervention. It does not establish autonomous stability, conservation or equal optimization across training backends. Native parity tolerances are prediction absolute 2e-7, risk absolute 1e-6 and risk relative 1e-5.

The simulator uses six observed position frames, width 128, ten message-passing blocks, two-layer MLPs and a 16-dimensional particle-type embedding. In graph-exposure comparisons, base-only and mixed training share initialization and the frame/noise schedule within each seed. Mixed training independently applies a uniformly sampled, floor-rounded quarter of the optional annulus to each example with probability one half.

For native-graph comparisons, the directed graph preserves self-message candidates and the 128-edge receiver cap. Optional pairs are the strict float32-distance annulus between r and 1.267r, appended in both directions after the native prefix without recapping. Sparse selectors use exactly floor(0.25 times the available annulus pairs); score ties use lexicographic particle IDs.

Reported means and sample SDs retain all three seeds. Full-horizon metrics become undefined if any required outcome fails or is missing. Accepted prefixes and shorter-horizon metrics remain separate. Boundary diagnostics are not computational guards or conservation tests. Descriptive timing retains its measured scope and does not establish a causal speedup.
