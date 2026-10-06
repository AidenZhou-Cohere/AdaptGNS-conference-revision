# Bounded adversarial review of the current conference argument

Read-only review of the current manuscript body and assembled manuscript after the argument map and earlier mechanism/narrative critiques. No partial native-cohort accuracy, continuation outcome, checkpoint or new experimental output was inspected. Title, abstract, numerical results and manuscript sources are unchanged.

## Overall judgment

The strongest defensible contribution is now identifiable: a controlled empirical test of whether residual ranking supplies useful graph-allocation information. The actual-action control weakens the dense-proxy objection; the self-loop control shows that a material absolute-accuracy change does not eliminate the observed risk–random ordering. Complete autonomous failures, physical excursions and paid scoring time prevent a favorable one-step result from being represented as a deployment advantage. Those are substantive strengths. The new introduction correctly leads with this empirical contribution rather than elementary algebra or an adopted training objective.

The main remaining weakness is how the evidence is connected. A reader still moves from full-model results to a descriptive exposure partition before encountering the graph-convention control, and then through a long compact-pilot sequence. That can make the paper read as accumulated analyses rather than a sequence of competing explanations. The strongest story is: **a score ranks residuals; its actual sparse action does not recover the same ranking; the risk–random deficit survives restored self-messages; neither that observed-state ordering nor fewer autonomous edges establishes a deployment advantage.** Training support remains an open explanation. The running native cohort and frozen continuation must remain separate and cannot be anticipated in the prose.

I found no mathematical error in the stated residual-moment proposition, top-budget perturbation proof, uniform additive-approximation extension or mixed-difference argument. Their assumptions are explicit and correctly limited. The concern is whether readers recognize when these bounds contain information, not a need to strengthen or enlarge the theorem section.

The fixes below are ordered by value. None changes a number, experimental protocol, title or abstract.

## 1. Make the experimental roadmap distinguish placement from the amount of expansion

**Locations:** `work/manuscript_body.tex:17` and `:128`; assembled manuscript around `:537` and `:649`. Also the main insert markers at body `:135–141`.

**Issue:** “The former freeze the simulator and match optional-pair counts to isolate placement” describes sparse-versus-sparse comparisons, but the same common-state experiment also compares base and dense with different budgets. The distinction between *where to add edges* and *whether/how much to expand* is already supported by the experiment; making it explicit gives the paper a clearer design and prevents an overly broad reading of “isolate placement.” This is more useful than another general sentence saying that correlations do not establish causality.

**Replace the first three sentences of the introduction's third paragraph with:**

```tex
We test these links through common-state comparisons and autonomous rollouts.
At a fixed state, equal-budget sparse policies compare where to place extra
pairs, while base and dense controls test the amount of expansion.
Autonomous rollouts then expose state feedback, changing future errors,
geometry, and cost.
```

Keep the existing final two sentences beginning “Base-preserving expansion cannot reduce…” unchanged.

**Replace the experiment section's opening paragraph with:**

```tex
The experiments separate three questions: does the score rank prediction
error, does its chosen graph improve a fixed-state prediction, and does that
improvement survive autonomous feedback at an acceptable cost?
Equal-budget sparse policies test placement; base and dense controls test
the amount of expansion. Objective and graph-convention controls test
competing explanations for the observed ordering. Correlations and a single
graph intervention do not certify the allocation bound's uniform score--gain
or interaction assumptions.
```

**Low-risk ordering change:** move the unchanged `GRAPH_BRIDGE_MAIN_INSERT` before `OPTIONAL_EXPOSURE_MAIN_INSERT`. The main sequence becomes full fixed study → actual-action control → graph-convention control → descriptive exposure partition. No paragraph, figure or table is removed or numerically altered. The exposure section should still state that it uses the original saved no-self-loop predictions; moving it does not transfer the bridge's convention to those arrays.

If later endpoint results require main-text space, move whole compact physical-control/benefit-head blocks to the supplement unchanged rather than compressing away adverse results or shrinking fonts. This is a conditional layout recommendation, not a request to restructure the pilot now.

## 2. Scope “native parity” to the object that was checked

**Locations:** `work/graph_bridge_main.tex`, paragraph ending the bridge subsection; assembled manuscript `:823–827`.

**Issue:** The unqualified sentence “native graph/feature parity checks pass” appears after a paragraph about all selectors. It can be read as saying that every supplied adaptive graph matches native inference. The native/supplied parity gate verifies the mandatory **base** path. The bridge's expanded policy graphs are intentional interventions; its global directed-edge ordering is also distinct from the native-prefix construction reserved for the autonomous/continuation controls. The argument map already makes this distinction, but the main paragraph does not.

**Replace the passage beginning “The cap is inactive…” with:**

```tex
The cap is inactive on all observed histories, and native/supplied
base-graph feature and prediction checks pass. The expanded graphs remain
controlled interventions. Thus restoring self-messages changes absolute
accuracy without removing the observed risk--random deficit. These controls
do not resolve the lack of expanded-graph training or autonomous feedback;
Appendix~\ref{sec:graph-bridge} gives all policies and the paired loop
interactions.
```

The “without removing” wording is deliberately narrower than “without explaining”: the experiment preserves the ordering while changing both predictions and, for risk, its scoring graph. It does not prove that self-messages have no effect on the gap's magnitude or mechanism. The appendix's paired loop interactions remain unchanged.

Optionally add one short appendix sentence after the description of the bridge construction: “Policy edge lists are globally ordered after selection; this screen does not test the separate native-base-prefix implementation used in the autonomous follow-up.” Do not insert any outcome claim about that running follow-up.

## 3. State the nonvacuity requirement of the affine risk–gain assumption

**Location:** `work/manuscript_body.tex:124–125`, after the existing actual-set regret paragraph; assembled manuscript around `:645–646`.

**Issue:** The affine risk–gain relation is not a consequence of accurate residual prediction; it is close to the proposition the empirical study is investigating. On a finite candidate set, some sufficiently large mismatch bound always exists. The present caveats say the assumptions are unverified, but do not quite explain why a formally valid regret bound can still be uninformative. This is the only worthwhile addition to the theorem discussion; the previous graph-transfer qualification and nonadditive appendix already address the genuine technical issues.

**Add these two sentences:**

```tex
On a finite candidate set, a sufficiently large mismatch bound always
exists. The result is informative only when score mismatch and set-interaction
error are small on the scale of the policy gains being compared; residual calibration
alone does not establish this condition.
```

No equation or proof needs changing. Do not claim that the measured rank correlations estimate `\epsilon`, `D_t`, `\zeta`, `\varepsilon_B` or `\gamma`. Do not add another theorem to compensate for the unmeasured assumptions.

## 4. Make the exposure identity locally readable and let the conclusion lead with the decisive controls

**Exposure location:** `work/optional_exposure_main.tex`; assembled manuscript `:802–806`.

**Issue:** The corrected paragraph properly distinguishes the identity from the additional positive-alignment observation, but `\Delta E`, `\Delta C` and `\Delta A` first appear there without local definitions. Their full definitions occur much later in the appendix. This is a notation-comprehension fix, not a request to reopen the already resolved algebraic or causal interpretation.

**Insert immediately before the displayed inline identity:**

```tex
Here $\Delta E$, $\Delta C$ and $\Delta A$ denote risk-minus-random
differences in normalized coordinate error, squared prediction change,
and residual alignment, respectively.
```

Keep the current identity, every gap value, every seed statement and the sentence distinguishing `C` from computational cost.

**Conclusion location:** body `:181`; assembled manuscript `:931`.

The current conclusion leads with the older dense-proxy comparison before its stronger actual-action and graph controls. A small reorder would connect the ending to the revised introduction. The following replacement retains all current findings, including the numerical failure count:

```tex
Residual prediction and graph allocation require different evidence.
In the full WaterDrop models, replacing dense-proxy labels with the chosen
sparse action's benefit does not recover strong risk ranking, and restoring
trained self-messages does not remove the observed risk--random deficit.
The compact pilot and full study both rank residuals more strongly than
dense-intervention benefit. These observed-state findings do not establish
deployment performance: the original full study retains eight NLL rollout
failures, faithful cached risk does not consistently beat cheap controls,
and historical WaterDrop gains at one horizon reverse over the full saved
rollout. A stronger allocation claim must connect its score to the signed
benefit of the chosen action and account for inexpensive alternatives,
failures, and end-to-end cost. Full-scale accuracy--runtime superiority
remains unestablished.
```

“Original full study” will also keep the existing failure count tied to the correct family when the separate native-cohort results are eventually integrated. This paragraph must be reconsidered using complete, audited follow-up outcomes, not partial accuracy.

## Resolved items that should stay resolved

Do not reopen the title/registered abstract, rebrand faithful regression or the elementary budget bound as novel algorithms, demand a new risk-calibration theorem, or reinterpret exposure groups causally. The current graph-conditioning qualification, nonadditive counterexample, noise-target appendix, fixed-seed/null rules, adverse outcomes and physical-boundary distinction are appropriate. More proof variants or more compact-model heuristics would dilute the strongest controlled question under this deadline.

This pass supports the concrete prose/scoping changes above; it does not establish conference readiness or supply new experimental evidence.
