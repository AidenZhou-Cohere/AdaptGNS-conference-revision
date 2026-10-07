# Main-paper narrative red team

Scope: current main text in `work/manuscript_body.tex` before `\clearpage` and `work/conference_experiments_main.tex`. Editorial preparation only; no canonical changes, new science, real-data audit, or new autonomous Goop3D claim. Keep the title. The replacement files are proposed prose around the existing unchanged scientific figures/tables.

## The paper's strongest claim

**A simulator must learn to use additional interactions before an allocator can usefully choose them; training support can improve the first problem without solving the second.** Residual prediction measures where the current simulator is wrong. Allocation needs to know where a particular graph change will help. The paper makes this distinction concrete with a simple cached controller, a common-state pair budget, and a progression from observed interventions to autonomous feedback.

The results already support an engaging argument. Goop provides the initial surprise: exposure improves random-policy rollouts in every seed, yet residual placement still loses to random on matched histories. WaterDrop demonstrates that the placement ordering can recover after exposure. Sand supplies the decisive stress test: its observed deficit shrinks, but its cached-risk rollout error and risk-minus-random gap worsen in every seed. A useful observed interaction is therefore an intermediate result, not the end of the allocation argument.

## Changes with the largest reader payoff

1. **Replace the abstract's story, not just its wording.** It currently describes a reproducibility audit and a small CPU WaterDrop study as the central evidence. That is materially out of step with the main paper's paired Goop/WaterDrop/Sand exposure studies. Lead with the residual-risk/action-benefit question, introduce the cached fixed-budget controller, then state the exposure/placement/feedback findings. The old WaterDrop step-200 reversal and released-rollout audit belong in the historical appendix. Faithful regression and the conditional bound support the design; neither should crowd out the main experiment in the abstract.
2. **Give each experiment the next unanswered question.** Use the sequence: can the model use extra messages → does residual ranking place them well → do those placement findings survive feedback → what computation is actually saved? Current prose often starts a new material rather than explaining why the reader needs it. Replacement transitions make WaterDrop a test of whether Goop's failure is universal and Sand a test of whether the observed improvement transfers to rollout.
3. **Make the Goop figure the opening result.** Its two panels already display the paper's thesis. Introduce it before a long protocol paragraph. Panel A holds policy fixed while changing training; panel B holds model/history/budget fixed while comparing policies. Say those interventions explicitly. Keep the two scales visually distinct and label B's zero reference as “risk = random.” “Exposure helps the simulator; risk still misplaces the budget” is the takeaway. The caption's detailed 1,077/1,080 history does not explain either displayed complete subset; move that accounting to the rollout table/narrative while retaining the three failed outcomes there.
4. **Stop relaunching the paper's argument.** The last control subsection currently ends “These findings motivate testing exposure separately from placement” after the exposure results have already been given. Replace this backward-looking sentence with an explanation of the observed gap, or put a short control result before the exposure experiment. The proposed replacement keeps it next to the placement result as corroborating explanation.
5. **Use the main text for decisions the evidence supports.** Native base beating random in every Sand seed is essential: a gain from exposure at a fixed expanded policy does not make expansion worthwhile. Keep that result prominent. Keep Goop's missing full-horizon risk comparison and the observed/autonomous scoring distinction. Replace repeated inventory-style totals and appendix catalogues with one clear test definition plus a concise statement of the failure's inferential consequence.
6. **End with what a reader now understands.** The current conclusion reopens original Sand/WaterDrop history and then recites material results. It should resolve the motivating question: cached residuals offer a cheap signal, but the signal must predict the effect of an action, and the model must remain reliable after that action changes future states. The replacement conclusion gives this synthesis while preserving WaterDrop's positive and Sand's adverse evidence.

## Main-versus-appendix placement

Keep in the main: the cached controller and budget equation; the residual/action-benefit distinction; the paired base-only versus random-exposure design; a clear common-state versus autonomous comparison; the Goop paired figure; observed contrast and autonomous-effect tables; three-seed scope; Goop's undefined risk family; WaterDrop's inspected-parent conditioning; Sand's adverse autonomous signs and native-base comparison; concise physical-validity and runtime limitations.

Move or compress into the appendix: historical audit chronology; original-checker or execution lineage; exact guard state inventories and every denominator repeated in captions; width/block counts and RNG/noise pairing implementation; detailed native-cap/self-loop conventions after one accurate main-text sentence; strict CPU/CUDA gradient-check history; a six-appendix citation chain; the catalogue of small control studies; exact per-policy excursion numbers. The Goop3D 25k observed-only addition is better as a clearly labeled supplementary endpoint, retaining its adverse one-of-three random-policy result. Do not present it as a completed autonomous replication.

The physical warning should survive compression: substantial predicted mass lies outside the metadata box while truth remains almost entirely inside. These are box diagnostics, not conservation tests. The runtime warning should also survive: a fixed optional-pair budget compares placement on identical states, while full rollouts change geometry; edge counts alone do not establish wall-clock savings.

## Particular sentence-level problems

- “A reproducibility audit shows why…” makes the abstract about research administration. Open with the scientific obstacle instead.
- “The resulting evidence supports a reproducible study…” does not tell the reader a result. State what exposure improved and what residual allocation failed to do.
- “The original Sand and WaterDrop experiments show why the idea deserves study…” weakens the conclusion and reintroduces a second chronological storyline. Remove it from the main conclusion.
- “Their three-seed full-horizon means remain undefined … we do not average those survivors” can become “The guard failures leave the full-horizon cached-risk interaction undefined,” with the complete denominator convention in the table/appendix.
- “All policies, failures and costs remain in Appendix…” repeated after each material sounds like an evidence report. Use a single compact citation to complete measurements, then continue the scientific argument.
- “Full studies” ambiguously combines distinct datasets, parent conditioning, backends and endpoints. Name the intervention once and keep cross-material comparisons descriptive.

## Suggested abstract architecture for root's rewrite

1. Problem: residual scores identify difficult predictions, but added interactions help only when they address the source of the error.
2. Method: cached residual controller with an exact extra-pair budget; paired graph exposure separates learning to use edges from placing them.
3. Principal observation: Goop/WaterDrop random-policy rollout gains in every paired seed coexist with conditional residual-placement results.
4. Decisive adverse observation: Sand improves its observed risk gap while worsening autonomous cached-risk error and gap in every seed; native base still beats random expansion.
5. Contribution/implication: evaluate intervention benefit and closed-loop behavior separately; residual calibration and realized edge counts alone do not establish useful adaptive computation.

Avoid suggesting the conditional allocation bound proves the observed controller effective. Keep the three-seed/exploratory scope and distinct WaterDrop continuation design in the body rather than making the abstract a protocol inventory.
