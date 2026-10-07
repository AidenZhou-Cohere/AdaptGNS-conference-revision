# Residual prediction and complete-action benefit

For each fixed history, model and target, compare the full base graph action with dense and equal-budget sparse graph actions. Let r=y-p_base and delta=p_action-p_base. Signed vector benefit is ||r||^2-||r-delta||^2 = 2*r dot delta-||delta||^2. Positive benefit means the complete action lowers error; it is not automatically an additive sum of edge values.

The WaterDrop control contains 1,782 observed-history cases: faithful and NLL objectives, three seeds, and 297 histories per model. Average particles within frame, eleven frames within trajectory, 27 trajectories within seed, then report all three seed means and sample SD. Preserve undefined correlations, negative benefits and full required denominators. Observed associations share the base residual algebraically and do not identify a causal effect of the risk score.

The simulator uses six observed position frames, width 128, ten message-passing blocks, two-layer MLPs and a 16-dimensional particle-type embedding. In graph-exposure comparisons, base-only and mixed training share initialization and the frame/noise schedule within each seed. Mixed training independently applies a uniformly sampled, floor-rounded quarter of the optional annulus to each example with probability one half.

For native-graph comparisons, the directed graph preserves self-message candidates and the 128-edge receiver cap. Optional pairs are the strict float32-distance annulus between r and 1.267r, appended in both directions after the native prefix without recapping. Sparse selectors use exactly floor(0.25 times the available annulus pairs); score ties use lexicographic particle IDs.

Reported means and sample SDs retain all three seeds. Full-horizon metrics become undefined if any required outcome fails or is missing. Accepted prefixes and shorter-horizon metrics remain separate. Boundary diagnostics are not computational guards or conservation tests. Descriptive timing retains its measured scope and does not establish a causal speedup.
