# Native graph rollout

Evaluate the fixed model on its native strict-radius capped graph, preserving its self-message candidates. Compare base, dense, Random25, Speed25 and cached Risk25. Radius r=.015; candidate radius 1.267r. Cached risk pays for one initial base scoring pass and then retains the exact optional budget from forecast 1. Each subsequent step predicts, integrates, rebuilds the graph and updates the risk cache. Only the initial six frames are observed.

WaterDrop uses sources 3–29 and horizon 995. The random stream is 93000+1000*training_seed+source_index. Guards reject nonfinite states/predictions/risks, absolute coordinates above 10, or more than 100,000 candidate pairs. Record boundary excursions separately and preserve every unsuccessful prefix. Fixed predicted-state and risk parity checks remain in the numerical kernel.

The simulator uses six observed position frames, width 128, ten message-passing blocks, two-layer MLPs and a 16-dimensional particle-type embedding. In graph-exposure comparisons, base-only and mixed training share initialization and the frame/noise schedule within each seed. Mixed training independently applies a uniformly sampled, floor-rounded quarter of the optional annulus to each example with probability one half.

For native-graph comparisons, the directed graph preserves self-message candidates and the 128-edge receiver cap. Optional pairs are the strict float32-distance annulus between r and 1.267r, appended in both directions after the native prefix without recapping. Sparse selectors use exactly floor(0.25 times the available annulus pairs); score ties use lexicographic particle IDs.

Reported means and sample SDs retain all three seeds. Full-horizon metrics become undefined if any required outcome fails or is missing. Accepted prefixes and shorter-horizon metrics remain separate. Boundary diagnostics are not computational guards or conservation tests. Descriptive timing retains its measured scope and does not establish a causal speedup.
