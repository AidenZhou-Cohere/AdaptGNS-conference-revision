# Conference editorial review after completed Sand integration

Reviewed manuscript SHA `d4ed535ce1ec78803d85ba37449c5fe6eec7d32c9e55303e7fff2cdec71c76fe`. Current title, read verbatim: **Adaptive Interaction Graphs for Particle Simulation**. No title change is recommended here. No canonical file, scientific result or D3 product was changed or generated.

The paper's strongest contribution is now the paired separation of **learning to use additional interactions**, **choosing them on a common observed history**, and **testing the choice under autonomous feedback**. Goop/WaterDrop provide constructive fixed-policy training effects; the complete adverse Sand study makes their limits credible. The uncapped/native graph distinction, retained Goop nulls, and physical/cost qualifications are already unusually clear. Remaining improvements should make this argument easier to recognize, rather than add more audit detail.

## 1. Align the abstract with the actual evidence scope

**Reviewer risk.** The abstract's final evidence sentence still says, verbatim, “New controlled CPU experiments evaluate alternative objectives and matched-budget policies on a small public WaterDrop subset.” That describes the pilot, but omits the full-architecture paired Goop/Sand studies and WaterDrop continuation that now drive the main paper. The opening “lightweight controller” can also sound like a measured efficiency claim; only the cached scoring design is established. The final absence of accuracy-runtime superiority remains correct and must stay.

**Recommendation.** Compare the separate optional abstract draft with the author's registered abstract before any edit. Foreground the graph-exposure intervention and the observed/autonomous separation, retain the adverse Sand result and undefined full-horizon comparisons, and describe caching directly instead of calling the controller lightweight. Do not imply that the three materials form one controlled material comparison. The title can remain unchanged.

**Evidence.** Current Experiments section; fixed observed/full-horizon tables; Goop, WaterDrop and Sand appendices; main Limitations. No additional computation is needed.

**Cost.** One paragraph replacement plus native layout check, but only after author comparison with the registered text. The local draft is available; the registration record has not been inspected. This is the highest-value change, but it is not an instruction to bypass that requirement.

## 2. State the constructive contribution before the result tour

**Reviewer risk.** The introduction begins as a new residual-based controller paper, but its strongest positive evidence is a training intervention and its strongest placement result is conditional or adverse. A reviewer may consequently read an unsuccessful selector with extensive diagnostics, instead of the controlled separation the paper actually establishes. The existing final conclusion sentence is a good correction; the introduction should make the same organizing idea explicit earlier.

**Recommendation.** Replace the current “We investigate this connection...” paragraph with a compact description of the two-stage intervention, for example:

> We pair base-only training with exposure to random, budgeted graph expansion, then freeze each simulator and compare allocation policies. This separates learning to use additional messages from choosing where to add them. Goop and Sand train both arms from initialization; WaterDrop uses paired continuations. Matched observed histories test placement, and full rollouts test feedback. Earlier objective and action-benefit controls explain why residual prediction alone need not identify useful interactions.

This names what the reader can reproduce without claiming that graph augmentation, faithful regression or the elementary top-budget bound is individually a new invention. Keep the following result paragraph, including Sand's adverse outcomes, intact.

**Evidence.** “Training for changing graphs” specifies precisely this intervention; all three admitted studies implement the two arms and fixed-policy comparisons. The current introduction and conclusion already support the empirical scope.

**Cost.** One paragraph substitution, no new table or number; a few lines may need trimming under the eight-page cap. No new experiment.

## 3. Make the two comparison axes explicit in the table captions

**Reviewer risk.** Table 1's “Mix - base (base)” and “Risk - random (base)” use “base” for different axes. Table 2 reports training effects at fixed policies, not absolute policy rankings. Its “Cached risk” also uses a different scoring history from Table 1's shortened “Risk.” The surrounding prose explains these facts, but a reviewer scanning the tables can still conflate a narrower placement gap, better-than-random allocation, and better-than-native-base simulation.

**Recommendation.** Use caption wording to identify the axes, without adding rows or changing the frozen display:

> Table 1: “Rows 1-2 compare training arms at fixed policies; rows 3-4 compare risk with random within each arm. Risk uses the preceding observed base graph. The interaction is the mixed-minus-base change in that placement gap.”

> Table 2: “Each entry compares training arms under the same evaluation policy; a negative entry does not establish that expansion beats the native base graph. Cached risk follows its own preceding selected graph.”

These can replace overlapping caption/prose phrases rather than accumulate as new text. Preserve all current scales, seed summaries, failure/null cells, policy/material order and the WaterDrop RMS not-predeclared label.

**Evidence.** The two frozen main tables and their defining contrasts; Sand improves dense exposure in all seeds while native base beats dense and random in all seeds; observed risk and autonomous cached risk are explicitly different regimes in Methods. This is a reading aid, not a new statistical interpretation.

**Cost.** Two caption edits and native layout check; no new scientific work or changed selection.

## Scope of this review

No numerical analysis, tests, scientific worker, raw array, model checkpoint or D3 result was opened or rerun. The complete admitted comparisons are treated as fixed evidence. This memo recommends at most the three editorial changes above; it does not certify conference readiness or replace the author's verification and registration comparison.
