# Common observed-history diagnostics

Use fixed ground-truth histories to separate equal-budget placement from changed autonomous geometry. For the WaterDrop objective control, evaluate sources 3–29 at target frames 7,106,205,304,403,502,601,700,799,898,1000, yielding 297 histories per model. Validation and test results are separate.

Measure normalized acceleration and position-coordinate error, predicted residual variance, graph counts, whole-action signed benefit and synchronized descriptive runtime. Risk from the preceding observed base graph is distinct from cached-own-graph autonomous risk. Dense benefit and residual/error correlation do not establish marginal edge values or useful sparse action benefit. Randomized and score-ranked actions must retain the same exact annulus budget on each common history.

The simulator uses six observed position frames, width 128, ten message-passing blocks, two-layer MLPs and a 16-dimensional particle-type embedding. In graph-exposure comparisons, base-only and mixed training share initialization and the frame/noise schedule within each seed. Mixed training independently applies a uniformly sampled, floor-rounded quarter of the optional annulus to each example with probability one half.

For native-graph comparisons, the directed graph preserves self-message candidates and the 128-edge receiver cap. Optional pairs are the strict float32-distance annulus between r and 1.267r, appended in both directions after the native prefix without recapping. Sparse selectors use exactly floor(0.25 times the available annulus pairs); score ties use lexicographic particle IDs.

Reported means and sample SDs retain all three seeds. Full-horizon metrics become undefined if any required outcome fails or is missing. Accepted prefixes and shorter-horizon metrics remain separate. Boundary diagnostics are not computational guards or conservation tests. Descriptive timing retains its measured scope and does not establish a causal speedup.
