# Reader-facing Speed25 / RMS25 definition

Source-only trace for the completed Goop2D and Sand graph-exposure studies. No model, array, metric product, scientific program, test, compiler, network or UI was invoked. Only frozen program text, package README/dependency text and source-file SHA256 values were inspected. The manuscript is unchanged.

## Minimal proposed main-text sentence

For insertion alongside the policy list at manuscript line 814:

> Speed25 uses latest-displacement magnitude, while RMS25 uses relative-velocity RMS over incoming nonself native neighbors; both rank pairs by the larger endpoint score (Appendix~\ref{sec:implementation-details}).

## Proposed precise appendix paragraph

> For the full-model Goop and Sand kinematic controls, let $z_i^0,\ldots,z_i^5$ be the current six-frame history. Speed25 uses $s_i=\|z_i^5-z_i^4\|_2$, computed directly in float32 without normalization or division by the timestep. RMS25 instead computes $v_i=(z_i^5-z_i^4)/\Delta t$ after converting both positions to float64, and uses $s_i=(|\mathcal N_i|^{-1}\sum_{j\in\mathcal N_i}\|v_j-v_i\|_2^2)^{1/2}$, with score zero if $\mathcal N_i$ is empty. Here $\mathcal N_i$ contains the incoming nonself neighbors remaining after the native base's 128-edge receiver cap; self-edge candidates participate in that cap but are excluded from the RMS sum and count. Both studies use the adapter's fixed $\Delta t=.0025$; autonomous entry additionally checks equality with metadata. Both controls rank optional pairs by $\max(s_i,s_j)$, with lexicographic particle-ID tie breaking. Histories are observed in the common-state comparisons and updated with the policy's predictions in autonomous rollouts. Native pair construction uses a cKDTree query at each radius followed by a strict float32-distance comparison ($d<r$ or $d<1.267r$, with $r=.015$); the annulus is the expanded pair set minus the geometric base pair set. Native incoming edges are sorted by receiver, distance and source ID before capping. Dense appends both directions of every optional annulus pair; sparse policies append their selected pairs. Neither restores short-range edges omitted by the cap or recaps the final graph.

This paragraph intentionally does not reuse the compact-pilot RMS neighborhood. The implementation sums RMS terms in the native incoming-edge order using float64 `np.add.at`; this arithmetic detail can remain in code rather than the paper.

## Exact definitions and source references

All paths below are relative to the workspace `/Users/aiden.zhou/Documents/Codex/2026-10-04/why-cangmai`.

Define these path prefixes:

- `S = outputs/AdaptGNS/research/results/sand_final24_launch_20261006/operator/sources`
- `G = outputs/AdaptGNS/research/results/cuda_graph_support_sources_20261006/workspace/work/deadline_research_20261005/cuda_preparation`
- `GR = outputs/AdaptGNS/research/results/cuda_graph_support_sources_20261006/workspace/outputs/AdaptGNS/research`

The Goop2D completed-package README explicitly points to `scoped_execution_completion_preparation_20261006`. That package's `dependency_paths.json` lines 45–86, 159–164, 201–206 and 513–518 bind the same numerical helper/evaluator hashes below through its frozen predecessor layout. The inspected `GR` copies have those exact hashes. Sand's completed package refers to its earlier frozen execution contracts; `sand_final24_launch_20261006` contains the exact pinned source bundle. This trace adds no execution authority.

| Reader-facing fact | Frozen source location |
| --- | --- |
| Speed is the norm of the latest displacement, `history[-1] - history[-2]`; not a six-frame average and no normalized-velocity conversion or dt division | `S/research/full_rollout.py`, lines 132–144, especially 140–141 |
| RMS casts positions before subtraction, divides by positive dt, uses incoming nonself edges, sums squared relative velocities, divides by their count, square-roots, and returns zero for count zero | `S/cuda_preparation/sand_graph_support_policy.py`, lines 16–36 |
| RMS support is the actual native capped base, not all uncapped radius neighbors or the newly expanded graph | Same policy source, lines 39–62, especially 48–54 |
| Fixed adapter dt `.0025`; common-state `physical_graph` uses `self.dt`; autonomous entry checks metadata equality and `_physical_rollout` supplies `float(metadata["dt"])` | Same policy source, lines 85–91 and 251–277 |
| Both materials instantiate that same adapter without overriding dt | `G/benchmark_goop_graph_support_rollout.py`, lines 169–188; `S/cuda_preparation/benchmark_sand_graph_support_rollout.py`, lines 236–255 |
| Float32 current observed histories contain the six frames immediately before the target; kinematics use this current history, while only risk uses `previous` | Goop evaluator lines 224–239 and 264–268; Sand evaluator lines 284–299 and 324–328 |
| Autonomous six-frame history starts as float32 and shifts in each prediction; both policies recompute from that current history | `S/research/native_graph_rollout.py`, lines 81, 134–149 and 182–183; physical-policy source lines 107, 160–175 and 208–209 |
| Pair score is maximum endpoint score; descending stable sort retains the first floor-quarter of lexicographic candidates | `S/research/budget_graph.py`, lines 33–47; `S/research/full_rollout.py`, lines 132–144; `S/research/full_same_state.py`, lines 35–37; bridge lines 53–59 |
| Base/expanded candidate pair membership uses cKDTree query followed by float32 norm strictly below the given radius; geometric annulus is expanded minus base | `S/research/graph_convention_bridge.py`, lines 43–62 |
| Receiver-major native edge order, then distance, then source ID; add diagonal candidates before applying cap 128; the cap can make directed support asymmetric | Same bridge, lines 65–91; `S/research/native_graph_rollout.py`, lines 32–34 |
| Dense selects all optional pairs, and the native graph appends their two directions without a post-selection cap | `S/research/full_rollout.py`, lines 132–137; `S/research/native_graph_rollout.py`, lines 25–45 |

There is **no Goop-versus-Sand difference in these score, neighborhood, candidate, pair-reduction or tie semantics**. Both benchmark helpers pin identical six shared numerical sources and the identical RMS adapter. Their source metadata, particle types, scientific cohorts and training backends differ, but those differences do not alter these control definitions. The fixed timestep comes from the shared adapter default, not an unconditionally read metadata value in the observed-history scorer; autonomous RMS additionally requires and uses the matching metadata value.

## Source SHA256 values

| Source | SHA256 |
| --- | --- |
| `S/research/full_rollout.py` = `GR/full_rollout.py` | `b0a37ee47619e699298b86865649e63c402cc1cb5dc2fa3dfd4055f7fa966eb8` |
| `S/research/native_graph_rollout.py` = `GR/native_graph_rollout.py` | `b4bbca6660dc449c81958010e30fda8be2bc0aaf58f677e0946d26e600e6daf5` |
| `S/research/graph_convention_bridge.py` = `GR/graph_convention_bridge.py` | `2c589c3c762631de5d3b3d60b986cc71178b97b3a76d0ce0d02132247b0be42d` |
| `S/research/full_same_state.py` = `GR/full_same_state.py` | `ff0f9b428791453a1592c23e0c1a4f31653418f74654c859701f52a70f32f592` |
| `S/research/budget_graph.py` = `GR/budget_graph.py` | `951f0d13863672f1dd248febf8cc960ba500103ca95361e06c9b0577913a8187` |
| `S/cuda_preparation/sand_graph_support_policy.py` = `G/sand_graph_support_policy.py` | `4a1db45f742043ba261c06273c1538fa0195dfb7f203423510b9a353c5ddb03a` |
| `S/cuda_preparation/benchmark_sand_graph_support_rollout.py` | `8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13` |
| `S/cuda_preparation/evaluate_sand_graph_support_final.py` | `952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58` |
| `G/benchmark_goop_graph_support_rollout.py` | `2e0ef0b84635c8430102cd797648d24cec14b457e1df900eb784d5e167966f8f` |
| `G/evaluate_goop_graph_support_final.py` | `cd970e04012930d896b31944271ccaf7da2b0881f22fe2f903533a8d5911dc6d` |

Goop benchmark source pins are at lines 37–49 and 169–188; Sand equivalents are at lines 37–48 and 236–255. The hashes above were read from the source files, without importing them.
