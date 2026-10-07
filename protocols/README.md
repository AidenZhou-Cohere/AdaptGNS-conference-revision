# Experimental definitions

`studies.json` lists the training, graph and evaluation settings. Base-only and mixed arms share initialization, sampled frames, noise and optimizer schedules within each seed. Mixed training exposes each example to Random 25 with probability one half. Goop and Sand start from scratch; WaterDrop continues fixed 100k parents for 10k updates. Goop3D uses six shorter 25k endpoints and remains a separate comparison.

The native base graph is a directed, receiver-capped prefix with self-messages. Expanded policies append uncapped symmetric pairs from the geometric annulus. A smaller graph along a different autonomous trajectory does not demonstrate cheaper placement on the same state. Speed uses last-step displacement. RMS uses float64 relative velocity over incoming nonself native neighbors with timestep .0025. Risk uses the preceding observed base graph for common-history diagnostics and its own preceding selected graph during autonomous rollout.

All three seed means enter reported means and sample SDs. These SDs are descriptive, not confidence intervals. Undefined means and guard failures remain in `results/failure_accounting.json`. Boundary crossings are physical diagnostics, distinct from computational guards. Runtime values retain their recorded measurement scopes; timings across devices or studies are not pooled.

The complete graph-exposure comparisons and the shorter Goop3D extension follow inspection of earlier evidence. They do not constitute pristine independent confirmation. The full WaterDrop objective control used official test sources 3–29; sources 0–2 and historical aggregates had already been inspected.
