# Sand graph-exposure comparison

Use all 1,000 ordered Sand training trajectories with 320 frames, float32 positions, type-6 particles, radius .015 and timestep .0025. Preserve the original scalar/vector int32/int64 particle-type representation. Train faithful base-only and mixed arms at seeds 0, 1 and 2 for exactly 100,000 updates with batch 2, Adam (betas .9/.999, epsilon 1e-8, no weight decay), exponential LR 1e-4 to 1e-5 and noise scale 6.7e-4.

Observed validation and test each use 150 histories; clean validation uses 128 frames. Autonomous test evaluates all 30 sources and all 314 forecasts under base, dense, Random25, Speed25, cached Risk25 and RMS25. RMS uses float64 relative velocity over incoming nonself native neighbors, timestep .0025, and zero score for an empty neighborhood. Preserve errors, boundary diagnostics, failures and their full declared denominators for every policy and seed.

The training profile is NVIDIA GB200, PyTorch 2.13.0+cu129, CUDA 12.9, two CPU threads, deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8, with TF32, AMP, compilation and distributed training disabled. The reported profile is distinct from Apple MPS optimization.

The simulator uses six observed position frames, width 128, ten message-passing blocks, two-layer MLPs and a 16-dimensional particle-type embedding. In graph-exposure comparisons, base-only and mixed training share initialization and the frame/noise schedule within each seed. Mixed training independently applies a uniformly sampled, floor-rounded quarter of the optional annulus to each example with probability one half.

For native-graph comparisons, the directed graph preserves self-message candidates and the 128-edge receiver cap. Optional pairs are the strict float32-distance annulus between r and 1.267r, appended in both directions after the native prefix without recapping. Sparse selectors use exactly floor(0.25 times the available annulus pairs); score ties use lexicographic particle IDs.

Reported means and sample SDs retain all three seeds. Full-horizon metrics become undefined if any required outcome fails or is missing. Accepted prefixes and shorter-horizon metrics remain separate. Boundary diagnostics are not computational guards or conservation tests. Descriptive timing retains its measured scope and does not establish a causal speedup.
