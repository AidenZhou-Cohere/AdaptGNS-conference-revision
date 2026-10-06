# Scientific narrative review after the three full-model follow-ups

Read-only review of `work/manuscript_body.tex`, the full-evaluation, actual-action, graph-bridge and optional-exposure main inserts, and their relevant appendices. No experiments or new outcome extraction. Title and registered abstract are outside this review's edit scope.

The strongest current thread is the sequence of controls: residual correlation does not establish action value; replacing the dense proxy with actual sparse-action benefit does not recover strong ranking; restoring self-messages improves absolute accuracy while preserving the risk–random ordering. The exposure analysis adds information about the saved prediction changes. Keeping the full study before the pilot, retaining negative controls and failures, and explicitly separating teacher-forced from autonomous policies are strengths worth preserving. The new theory and decomposition are mathematically coherent; the refinements below sharpen their interpretation rather than repair a failed proof.

## 1. Lead the contribution with the controlled empirical question

**Location:** `work/manuscript_body.tex`, introduction paragraph beginning “Our contribution is a controlled analysis …” (line 19 in the reviewed body).

**Why:** The present list leads with an exact budget and an elementary top-budget perturbation bound. Reviewers may infer a stronger algorithmic/theoretical novelty claim than intended, while the strongest new evidence appears later in the paragraph. Faithful regression is already correctly attributed. The new contribution is the controlled investigation and its substantive empirical distinctions; there is no need to market the budget identity or the algebraic decomposition as novel mathematics.

**Replace the whole paragraph with:**

```tex
Our empirical contribution is a controlled test of residual-guided graph
allocation. Fixed-state comparisons and an explicit optional-pair budget
separate placement from changes in rollout geometry, while objective and
graph-convention controls test competing explanations. In the six-model
full-architecture study, residual scores track prediction error more strongly
than the benefit of their own sparse actions; random allocation remains more
accurate on observed histories after restoring trained self-messages.
A further exact decomposition describes where the saved policy gaps arise.
Historical reanalysis and autonomous evaluations retain horizon reversals,
failures and measured cost. The conditional analysis states the additional
score--gain and interaction assumptions needed for an allocation guarantee.
```

This preserves the existing evidence boundaries without claiming a new uncertainty objective, a new generic top-budget theorem, or a universal negative result about uncertainty-based allocation.

## 2. Make the cache's graph-conditioning error explicit in the bound

**Location:** the lagged-allocation derivation, immediately after defining fixed-convention risk `\rho_t` and before the two sup-norm assumptions (around lines 113–117).

**Why:** The actual autonomous cache is generated on the preceding selected graph. The displayed assumption remains mathematically valid, but a reader can mistake `\epsilon` for ordinary variance-estimation/calibration error on a reference base graph. If the theoretical risk uses that reference graph, transferring the cached score across graph inputs is another part of `\epsilon`. The graph-convention findings now make this distinction especially relevant. This is a clarification of the existing conditional assumption, not a new measured error term or a new guarantee.

**Insert after the initial risk/cache definition:**

```tex
The cache is produced on the preceding selected graph. If $\rho_t$ refers
to a reference graph, its estimation bound must also cover this change of
graph input. Let $\rho_{t-1}^{\mathrm{used}}$ denote residual risk on the graph
that produced the cache. Bounds
$\|q^{t-1}-\rho_{t-1}^{\mathrm{used}}\|_\infty\le\epsilon_{\mathrm{stat}}$
and
$\|\rho_{t-1}^{\mathrm{used}}-\rho_{t-1}\|_\infty\le\Gamma_{t-1}$
allow $\epsilon=\epsilon_{\mathrm{stat}}+\Gamma_{t-1}$ below.
Neither ordinary calibration nor the reported rank correlations bound this
graph-transfer term.
```

All existing equations can remain unchanged. If space is tight, retain only the first two and final sentences. This avoids implying that accurate risk prediction on one graph alone satisfies the cache-error assumption on another graph.

## 3. Separate the decomposition's additional observation from the identity

**Location:** `work/optional_exposure_main.tex`, from “Particles covered by both actions …” through “These are observed algebraic components.” Keep the gap values, the four-way definition and the final ten-block caveat unchanged.

**Why:** Once the gap is positive, `\Delta C>\Delta A` follows immediately from `\Delta E=\Delta C-\Delta A`; it is not an independent mechanism finding. The additional result is that alignment also increases (`\Delta A>0`), plus the location and signs of the unconditional contributions. Calling the quadratic term “cost” in this paper can also be confused with computational runtime. The equal-degree subgroup's positive primary contribution, already audited in all six models, prevents a simple “more incident edges caused the gap” reading. The current appendix's fractions and noncausal caveats are good and should remain.

**Replace those sentences with:**

```tex
The both-covered group provides the largest positive unconditional
contribution in every seed; its particle fraction is reported alongside the
contribution. Relative to random, risk increases both alignment with the base
residual and the squared normalized prediction change. The identity
$\Delta E=\Delta C-\Delta A$ makes $\Delta C>\Delta A$ equivalent to the
positive gap; the additional observation is $\Delta A>0$ in every seed.
The both-covered equal-degree subgroup also contributes positively in all
six models. These descriptions do not isolate excess local degree as a cause.
```

This retains the substantive finding without presenting an algebraic restatement as causal evidence. It also keeps the distinction between an unconditional contribution, its group's prevalence, and a conditional per-particle effect visible in the main text.

**Notation clarification within the same decomposition refinement:** The parent's identified collision is real: the Method uses $e_i$ for a residual vector, whereas the exposure appendix uses it for scalar coordinate squared error without redefining it. Also, dividing by the saved two-coordinate standard deviation is elementwise. Both are cheap presentation fixes with no numerical or protocol change.

Replace the appendix's introduction of the contribution with:

```tex
In this appendix, $y_i$, $b_i$ and $p_{ia}$ denote target, base-predicted and
action-predicted positions. Let $s=(s_1,s_2)$ be the saved acceleration
standard deviations, with $\oslash$ denoting elementwise division. Define
the scalar normalized coordinate squared error
$\ell_{ia}=\tfrac12\|(p_{ia}-y_i)\oslash s\|^2$.
A group's unconditional contribution is
$N^{-1}\sum_i I_{ig}(\ell_{iL}-\ell_{iR})$.
```

Later in the same appendix, use
`$r_i=(y_i-b_i)\oslash s$` and
`$\delta_{ia}=(p_{ia}-b_i)\oslash s$`, and replace the error identity with
`$\ell_{iL}-\ell_{iR}=(C_{iL}-C_{iR})-(A_{iL}-A_{iR})$`.
The alignment and quadratic-term formulas, all numbers, tables and array identities stay the same. This local declaration also prevents confusion with the Method's normalized-acceleration target $y_i$.

These are the three highest-value changes from this pass. No numerical table, experimental protocol, title, abstract or frozen scientific source needs to change.
