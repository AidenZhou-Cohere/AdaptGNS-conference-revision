# WaterDrop faithful/NLL objective comparison

Train each objective at seeds 0, 1 and 2 on all complete training trajectories for exactly 100,000 updates. Use batch size 2, Adam without weight decay, learning rate 1e-4 decaying exponentially to 1e-5, and random-walk position noise with final velocity-noise scale 6.7e-4. Save training normalization and optimizer/RNG state. Fixed one-step validation uses 128 histories every 5,000 updates; the final checkpoint is selected by update count.

Faithful regression uses a squared-error mean path and detached variance-head features/residual target. Corrected isotropic Gaussian NLL retains the dimension-dependent log-variance coefficient. Six paired models compare these objectives; faithful regression is an established method, not a new contribution here.

The base training radius is .015 with native capped self-message graphs. The separate objective-control rollout uses uncapped symmetric pairs without self-messages, five policies (base, dense, Random25, Speed25 and cached Risk25), and official test sources 3–29 for all 995 forecasts. This graph convention differs from the native exposure comparisons. Historical aggregates and sources 0–2 were inspected; the extension is not pristine independent confirmation. The coordinate guard is absolute value 10; the candidate-pair guard is 100,000. Eight NLL seed-2 coordinate failures remain in the released objective-control results.

Training and evaluation use Apple MPS, two CPU threads, SciPy host graphs and CPU fallback disabled.

The simulator uses six observed position frames, width 128, ten message-passing blocks, two-layer MLPs and a 16-dimensional particle-type embedding. In graph-exposure comparisons, base-only and mixed training share initialization and the frame/noise schedule within each seed. Mixed training independently applies a uniformly sampled, floor-rounded quarter of the optional annulus to each example with probability one half.

For native-graph comparisons, the directed graph preserves self-message candidates and the 128-edge receiver cap. Optional pairs are the strict float32-distance annulus between r and 1.267r, appended in both directions after the native prefix without recapping. Sparse selectors use exactly floor(0.25 times the available annulus pairs); score ties use lexicographic particle IDs.

Reported means and sample SDs retain all three seeds. Full-horizon metrics become undefined if any required outcome fails or is missing. Accepted prefixes and shorter-horizon metrics remain separate. Boundary diagnostics are not computational guards or conservation tests. Descriptive timing retains its measured scope and does not establish a causal speedup.
