# Goop whole-graph action-gate feasibility

The exploratory gate was fitted and frozen using all 24,000 training and 450 validation paired-action labels. The subsequent train-only capacity study closed with **32 complete H395 outcomes, one graph resource guard and three unexecuted outcomes**. Its requirement that all 36 outcomes complete was not met, so the planned 360-outcome test evaluation was not admitted. This record supports a method and feasibility account; it provides no held-out accuracy or runtime comparison.

## Method and target

For an observed six-frame history (x), next-position target (y), and a randomized complete graph action (u), the supervised label was

\[
b(x,u)=\frac{1}{2N}\lVert f_{\mathrm{base}}(x)-y\rVert_F^2
       -\frac{1}{2N}\lVert f_{\mathrm{random25}}(x,u)-y\rVert_F^2.
\]

Both predictions used the same frozen simulator and history. Positive benefit means the randomized action reduced one-step coordinate MSE. That action adds a uniformly sampled quarter of optional annulus pairs to the capped base graph, rounding down. One randomized action per labeled state targets expected whole-action benefit, not marginal edge utility.

Three original Goop2D `mix` models, seeds 0/1/2 at 100,000 updates under the faithful objective, supplied predictions with mean and variance parameters frozen. The gate requests expansion iff predicted benefit is positive, using one deployed simulator forward per forecast. Its inputs contain no future targets, predictions, variance outputs or processor caches.

The 11 features use only the current history and fixed domain bounds:

| Feature group | Scalars |
|---|---:|
| Logarithm of one plus particle count | 1 |
| Mean, 90th percentile and maximum displacement magnitude | 3 |
| Mean, 90th percentile and maximum second-difference magnitude | 3 |
| Mean and 10th percentile signed wall distance | 2 |
| Outside-particle fraction and maximum coordinate excursion | 2 |

Differences are per stored frame, without division by a time step. Boundary features are geometric descriptors.

Each model used all 1,000 training sources at eight fixed frames (8,000 states) and all 30 validation sources at five fixed frames (150 states). Training-only normalization preceded float64 ridge fits with penalties **0.001, 0.1 and 10.0**. One common penalty minimized validation selected-action MSE with equal source and seed weighting. The completed selection chose **λ = 0.1**; the threshold remained zero. Previously used validation data make this exploratory selection.

## Planned evaluation and observed feasibility stop

The policies were base, always-expanded random25, the learned gate, and a random gate with its expansion-request probability frozen from validation. Matching that request rate does not establish equal deployed cost. The intended test grid was **4 policies × 3 frozen heads/models × 30 sources = 360 H395 outcomes**.

Before test admission, the same four policies had to complete H395 rollouts on three training sources selected by particle-count rank: minimum, lower median and maximum. Their source indices were 238, 318 and 988. The complete capacity accounting is:

| Model seed | Complete H395 | Committed guard | Unexecuted | Required |
|---|---:|---:|---:|---:|
| 0 | 12 | 0 | 0 | 12 |
| 1 | 12 | 0 | 0 | 12 |
| 2 | 8 | 1 | 3 | 12 |
| **Total** | **32** | **1** | **3** | **36** |

For **seed 2, source 988, base policy**, construction of the current graph at forecast **261** triggered `candidate_pair_resource_guard`: the candidate universe contained **100,498 pairs**, exceeding the fixed **100,000-pair** limit. This is the unordered geometric universe inside the expanded radius, counted before native receiver capping or policy selection, including for the base policy. The row retained **260 completed forecasts and 260 deployed forward attempts**; forecast 261 did not reach the simulator forward. Its full-H395 mean and final-horizon MSE are undefined. The other three policies for that seed/source were never started.

This native graph resource guard does not establish learned-gate failure, nonfinite-state instability, an out-of-memory event or physical divergence. All three children were reaped with no recorded signals or timeouts; their collections recorded unchanged models and reverified inputs.

## Scientific scope and evidence

The failed capacity requirement leaves the test-cost forecast undefined and test admission closed. No survivor accuracy, reduced-grid comparison or test performance is reported. Missing H395 outcomes at seed 2/source 988 invalidate the complete source/seed denominators for the planned paired comparisons. Teacher-forced training also leaves autonomous distribution shift unresolved. This stop does not establish whether action-value learning generally succeeds or fails.

The [completed-fit manifest](/Users/aiden.zhou/Documents/Codex/2026-10-04/why-cangmai/work/deadline_research_20261005/cuda_preparation/goop_gate_complete_fit_json_1258_code_audit_v1/transport.json) binds **17 original JSONs**: nine candidates, three heads, selection, fit result and three CPU-owner records. Their bytes match that manifest and the [original fit review](/Users/aiden.zhou/Documents/Codex/2026-10-04/why-cangmai/work/deadline_research_20261005/cuda_preparation/goop_complete_fit_original_json_review_1301_code_audit_v1.json). The fit result binds all six training/validation collections; selection confirms λ = 0.1 and three 8,000-state heads.

The [stopped-capacity review](/Users/aiden.zhou/Documents/Codex/2026-10-04/why-cangmai/work/deadline_research_20261005/cuda_preparation/goop_capacity_failure_independent_review_saved_array_v1.json) reconciles 36 cells, 77 JSONs and 33 opaque-artifact hash chains. Arrays remain on the original host; prior numeric audits remain distinct from this local scalar review. This reporting pass accessed no array/model payloads or test data and reran no experiments or tests.

The short [method-and-scope note](/Users/aiden.zhou/Documents/Codex/2026-10-04/why-cangmai/work/conference_presentation_20261006/goop_gate_feasibility_20261006/goop_gate_feasibility_appendix_v2.tex) is integrated in the existing manuscript appendix. Native compilation succeeds; the main text, title, abstract and earlier results remain unchanged. Final exported-PDF visual inspection remains pending.
