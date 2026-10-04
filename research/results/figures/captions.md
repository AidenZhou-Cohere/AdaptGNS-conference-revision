# Figure captions

## legacy_full_rollout_curves.pdf

Means across 30 saved trajectories per method, conditional on one saved checkpoint per method. No training-seed variability is shown. Vertical guides mark step 200. WaterDrop contains 995 and Sand 314 predicted steps after the input history. These are legacy results, not corrected-objective training.

## waterdrop_conditional_delta_intervals.pdf

100,000 paired bootstrap resamples of 30 whole trajectories, conditional on the saved models. Trajectory pairing is assumed from row order because IDs are absent. Intervals are pointwise and unadjusted for multiple comparisons; they are not uncertainty over training seeds. Negative values favor Adaptive. The rollout mean averages all 995 predicted steps and is not an unnormalized physical-time integral. All numerical statistics are reused from reanalysis.json.

## sand_posthoc_tradeoff.pdf

Observed point estimates from the existing Sand test artifacts. The 11 adaptive configurations vary the node percentile and radius factor; the star marks the configuration reported in the original paper. This is a post-hoc sensitivity display, not validation-only tuning or a new selection of hyperparameters. Realized edge counts come from different predicted geometries, not fixed-state compute matching. Both axes include zero. No training-seed uncertainty is available.
