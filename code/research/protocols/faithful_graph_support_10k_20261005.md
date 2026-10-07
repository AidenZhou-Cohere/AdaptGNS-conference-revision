# WaterDrop paired graph-exposure continuation

For each faithful seed 0, 1 and 2, continue the same 100,000-update parent in base-only and mixed arms for exactly 10,000 additional updates. Restore the full mean/variance model, Adam moments/counters, normalization and RNG state. Use batch size 2, fixed learning rate 1e-5, the original noise scale 6.7e-4 and absolute schedule steps 100000–109999. Model selection uses the fixed 110k endpoint. Resetting Adam or training fresh 110k models changes this comparison.

Observed diagnostics use 128 validation and 297 test histories. Native autonomous evaluation uses all 995 forecasts for official test sources 3–29, all five declared policies, and separate full-denominator seed summaries. Observed Risk25 scores the preceding observed base history; autonomous Risk25 caches its own selected graph. The runtime is Apple MPS with two CPU threads and CPU fallback disabled.

The simulator uses six observed position frames, width 128, ten message-passing blocks, two-layer MLPs and a 16-dimensional particle-type embedding. In graph-exposure comparisons, base-only and mixed training share initialization and the frame/noise schedule within each seed. Mixed training independently applies a uniformly sampled, floor-rounded quarter of the optional annulus to each example with probability one half.

For native-graph comparisons, the directed graph preserves self-message candidates and the 128-edge receiver cap. Optional pairs are the strict float32-distance annulus between r and 1.267r, appended in both directions after the native prefix without recapping. Sparse selectors use exactly floor(0.25 times the available annulus pairs); score ties use lexicographic particle IDs.

Reported means and sample SDs retain all three seeds. Full-horizon metrics become undefined if any required outcome fails or is missing. Accepted prefixes and shorter-horizon metrics remain separate. Boundary diagnostics are not computational guards or conservation tests. Descriptive timing retains its measured scope and does not establish a causal speedup.
