# Prospective qualitative figure selection v2

This source-only refinement supersedes v1 before the new test outcomes. The
frozen evaluators retain forecasts1,10,50,200,H rather than floor(H/2); use
the shared saved checkpoint200 as the intermediate display time. This avoids
requiring extra inference or substituting unavailable frames after seeing
results. The v1 plan is retained. No scientific protocol or source changes.

Select examples by geometry metadata, before considering any new policy error.
For each complete new dataset study, sort test sources by initial particle count,
then original source index. Use the lower median source and training seed0.
Keep the same source, seed and time indices for every displayed policy. This is
an illustration of one trajectory, not an estimate of population performance.

Show ground truth, ordinary-graph-trained base, mixture-trained random25 and
mixture-trained cached risk25. If the separate Goop global gate completes its
entire declared comparison, append its seed0 trajectory for the same source;
do not replace an unfavorable existing panel. Use forecast1, forecast200 and H,
with each dataset's actual full H explicitly labelled. WaterDrop110k
continuations, fresh Goop/Sand100k and Goop3D25k remain distinct training budgets.
If a requested frame is absent because of failure or timeout, label the panel
with the failure/missing state and accepted prefix length; do not silently
substitute an earlier or better trajectory.

Use common physical axes across methods at each time. For3D use a fixed
orthographic view or a clearly labelled projection, with the same view and
color scale for truth and every predictor. If a plot clips at the metadata
box, explicitly mark/count out-of-view particles. Do not conceal excursions
through arbitrary autoscaling, selective particles or axis cropping. Any
fixed particle subsampling for legibility applies to identical IDs in all
panels and is disclosed; metrics always retain every particle.

Prefer a legible main figure illustrating Goop2D and Goop3D as contrasting
physical regimes, chosen for the declared breadth question rather than
favorable results. Retain the corresponding Sand/WaterDrop panels in the
appendix when space requires. Full paired quantitative comparisons, failure
counts and realized compute remain the evidence for claims; these visual
examples neither replace them nor identify dimensionality as a causal factor.

This plan does not change any frozen experiment or admission rule. It has
not accessed test source arrays or model outcomes. Generate figures only
from admitted completed evaluation artifacts, retaining every selected
failed/missing cell.
