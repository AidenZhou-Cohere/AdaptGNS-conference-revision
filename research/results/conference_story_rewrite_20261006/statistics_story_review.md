# Independent story review

Reviewed the current main text in `outputs/revised_manuscript.tex` and `work/conference_experiments_main.tex` against `before_revised_manuscript.tex` in this directory. This was a narrative/claim review only: no manuscript edits, tests, compilation, inference, or independent byte-equivalence certification.

**Assessment: substantially improved; no new substantive scientific-claim error found against the baseline.** The introduction now makes the physical intuition understandable before presenting its limitation. The experiments form an intelligible chain: historical accuracy/graph puzzle → objective-controlled common-state test → actual-action proxy check → self-message alternative → description of prediction changes → autonomous consequences and cost. This is a paper argument rather than a chronology of completed work.

The important qualifications survive without dominating the story: historical single-model comparisons, inspected-data reuse, three trained seeds, graph-input shift, exploratory follow-ups, null full-horizon means after failures, and the difference between observed-history risk timing and autonomous cached scoring. The positive high-risk-quartile observation and favorable dense comparison remain beside the adverse risk–random/speed evidence. The exposure decomposition is correctly descriptive, and its alignment identity is not presented as an independent causal explanation.

Small remaining refinements, in priority order:

1. In the historical paragraph, “Favorable comparisons ... remain in Appendix” still sounds like an editor describing retained material. State the favorable historical finding directly, with its audited metric/comparator and appendix citation. This also gives the original motivation more balance before the new controlled study.
2. The compact-study paragraph sits inside “What computation does the policy require?” despite not being a runtime result. Place its concise scope description near the controlled setup, or give it a short concluding context paragraph outside the runtime subsection. Let the practical experiment sequence end with cost.
3. Delete the redundant “All 2,550 histories contribute to this comparison.” The preceding sentence already specifies 425 histories per model; the total sounds like a completion report and is not an independent-replication count.
4. Clarify the aggregation sentence to say that frames are averaged within trajectories and trajectories equally within each seed, followed by means/sample SDs across three seeds. The current wording compresses the middle step.
5. Prefer “normalized coordinate MSE multiplied by $10^3$” to “in normalized units $\times10^3$” for the prediction-change gap. Values need not change.

**Known deferred issue:** the intentionally unchanged abstract still describes only a small-subset CPU study and leads with a reproducibility audit. It no longer represents the main paper's six full-model controlled argument. Resolve after the registered-text check; this review does not recommend editing it now. Title unchanged as requested. No evidence supports anticipating pending native or training-exposure follow-up results.
