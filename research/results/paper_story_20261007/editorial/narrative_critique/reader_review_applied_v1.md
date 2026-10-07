# Reader review of the applied first revision

**Approve the new narrative arc.** The abstract now describes the actual paper, Figure 1 makes its central distinction visible, and the experimental subsections each answer a different question. Moving the formal action-benefit interpretation after the experiments works: the mathematics now explains a puzzle the reader has seen. Keep this structure. The remaining improvements are local.

I read the current main source and experiment text and inspected the new standalone Figure 1 PNG. The figure is clear, the paired seed lines communicate heterogeneity, and the separate endpoint labels prevent treating its two scales as directly comparable. Source-level placement in the introduction is appropriate; this review did not inspect final manuscript pagination.

## Fix before finalizing

1. **The correlation paragraph has a wrong grammatical antecedent.** After “Here, realized action benefit is …”, “Its correlation with base error” appears to describe benefit, although the reported quantity is residual-score correlation. Replace the two sentences with:

   > Here, realized action benefit is base-graph loss minus expanded-graph loss on the same history and target. The residual score correlates with base error at $.214\pm.088$, but with its own sparse-action benefit at only $.001\pm.048$.

   This is the highest-impact prose fix because the current reading changes what the mechanism evidence says.

2. **Define the training interaction before asking the reader to interpret its sign.** “Autonomous interaction” appears in the introduction, and “favorable training interaction” appears before an explicit definition. Replace the introductory clause with ordinary language:

   > WaterDrop reverses the observed-test ordering, but exposure does not consistently improve its rollout performance relative to random across seeds.

   Then add once beside Table 1 or at the first result paragraph using the term:

   > The training interaction is the change in the risk-minus-random gap from base-only to mixed-graph training; negative values mean exposure improves risk relative to random.

   Table 1's “Change in gap” then has an immediate interpretation. This also prevents confusing an improvement in the gap with risk actually beating random.

3. **The action-benefit equation still mixes directed edges and unordered pairs.** The revised method defines $E_0$ as directed and $S$ as unordered, but later writes $E_0\cup S$. Define $D(S)=\{(i,j),(j,i):\{i,j\}\in S\}$ when appending the selected pairs, and write $E_0\cup D(S)$ in the benefit equation. This restores consistency with the method's existing two-orientation construction without changing the model or claim.

## Two small presentation edits

- Table 1's new semantic row groups are substantially easier to read. Its headline “Training helps expanded graphs…” is too broad for this observed-state table: the WaterDrop random training-effect mean is positive. Use **“Exposure lowers the observed risk-minus-random gap across the three materials.”** The body and row groups then show precisely which contrasts justify that statement. Keep the column scales and the explicit sign convention.
- Figure 1's caption phrase “does not solve where to send them” is awkward. A precise replacement is **“Graph exposure helps the simulator without making risk allocation better than random.”** The Goop-specific caption immediately bounds the claim. The current panel titles and paired mean labels already explain the figure well; do not add another legend or a dense protocol paragraph.

## Remove one dangling phrase and one repeated roadmap

The conclusion now begins with control of communication, so “The central difficulty is deciding what that need means” has no antecedent. Replace the first paragraph with:

> Adaptive interaction graphs make a particle simulator's communication pattern a controllable part of its computation. Our cached controller and explicit pair budget reveal a crucial distinction: residual prediction locates error, whereas useful allocation requires predicting the benefit of a graph change. Paired graph-exposure training separates this allocation question from the model's ability to use additional messages.

The introduction and the first experiment paragraph each spell out all three comparisons. Keep the introduction's reader roadmap; shorten the experiment opening to:

> We first hold the expansion policy fixed to test graph-exposure training, then compare placement on identical observed histories. Full rollouts test whether those findings survive the controller's own predicted states.

This removes repetition while preserving the question-led section headings. The Goop3D 25k paragraph remains an optional streamlining target: it introduces a fourth endpoint immediately before the rollout question. If moved out of this sequence, retain its adverse one-of-three fixed-random finding in the clearly labeled appendix and retain its distinct observed-only scope in the limitations.

## Preserve what is working

Keep the Sand native-base comparison, the three Goop guards and undefined effects, and the observed-base versus autonomous-selected scoring distinction. Table 2 is much easier to scan after replacing repeated zero counters with its exact failure caption. The simple physical-mismatch paragraph is useful and readable. No further reordering or larger theory expansion is needed for this reader pass.
