# Figure/story copy for root

## One-sentence thesis

Connectivity is part of a particle model's training distribution; learning to use additional messages and choosing where they help are separate problems.

## Recommended compact flowchart

Use a simple left-to-right argument, with one small particle-graph intervention above it. The figure is a guide to the experimental comparisons, **not a measured causal model or a guarantee**.

| Stage | Question shown to reader | Controlled change | Result carried by the next evidence panel |
|---|---|---|---|
| Construct | Which messages can change? | Keep native edges; select an exact quarter of annulus pairs; append both directions. | A common-state budget isolates placement from edge count. |
| Learn | Can the model use extra messages? | Base-only versus mixed-graph training; evaluate the same Random25 policy over full rollouts. | Goop and WaterDrop improve in all three paired seeds; Sand improves in two. |
| Place | Does risk identify useful messages? | Freeze model and observed history; compare risk with random at the same budget. | Goop/Sand risk still loses; WaterDrop reverses on test, not validation. |
| Roll out | Does the gain survive feedback? | Evolve each policy's own positions; cached risk uses its own preceding selected graph. | Sand cached-risk exposure worsens every seed; WaterDrop interaction is mixed; Goop risk interaction is undefined. |

An arrow joins the questions, not the scientific outcomes: the apparent success at one stage motivates the next test, but does not imply success there.

If four columns are too crowded, place the “Construct” graph above three equal columns labeled **Learn → Place → Roll out**. Each column gets one short question, one control switch and one result line. Do not put all dataset names, hashes, quotas or statistical definitions in the flowchart.

## Minimal figure labels

- Native messages (thin gray)
- Optional annulus pairs (light dashed)
- Selected extra messages (one saturated color)
- Same number, different placement
- Training: native only / random expansions
- Observed history: shared state, base-history scores
- Autonomous: predicted state, cached selected-graph scores
- Feedback arrow: prediction → positions → next graph

## Evidence figure priorities

1. Keep exact paired Goop exposure/placement values; place them after the intervention is understood. This is evidence, not the conceptual opening.
2. Give the Sand reversal equal visual weight. Display the observed risk-gap training interaction and autonomous cached-risk exposure as separate quantities/panels, with separate scales and clear labels. Do not draw a line between unlike units or imply the same statistic reverses.
3. Retain complete policy table(s), including guard/undefined/NP cells, beside compact result plots or in the main-plus-appendix arrangement root chooses. Do not select only favorable policies.
4. A particle-state visual must use the existing fixed metadata-selected example, common bounds and visible out-of-box markers. Do not choose a new example by error or aesthetic preference.

## Scientific guardrails for captions

“Improvement” in the first column is mixed-minus-base training at a fixed evaluation policy, not expansion versus native graph. “Risk loses” in the second is the observed risk-minus-random policy gap. “Feedback” in the third changes both predicted positions and the source of cached scores. The diagram links questions; it cannot identify a material-only cause or establish a universally stable controller.

The first exposure comparison already uses full-rollout error at fixed Random25. The third question introduces autonomous **risk placement**, not the first use of rollouts. Do not draw the flowchart as if all early evidence were one-step and only the last experiment used rollouts.
