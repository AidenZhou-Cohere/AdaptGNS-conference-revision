# From additive score regret to actual set benefit

Status: explanatory derivation for the revised manuscript, independently proof-checked. These are conditional mathematical statements, not novelty claims or measured properties of the trained WaterDrop models. No model, dataset, or held-out record was accessed for this note.

## Scope and recommendation

The current manuscript's top-budget proposition is correct: it explicitly bounds regret for an **additive surrogate**, rather than the actual nonlinear value of selecting several edges. A brief bridge to the actual set objective is useful because even perfect singleton scores do not control joint benefit. The bridge requires an additional, unverified interaction assumption. It does not strengthen any empirical result or establish rollout stability.

Fix an observed state, the model, its mandatory base graph, and a finite optional-pair ground set \(C\). All comparisons below are on that same state and ground set. Let \(0\le B\le |C|\), and let

\[
F(S)=G_x(S)=\mathbb E[\mathcal E(f(x,E_r),Y)-\mathcal E(f(x,E_r\cup S),Y)\mid x].
\]

Thus \(F(\varnothing)=0\). The expectation is part of the fixed-state objective; it is not a rollout objective. No monotonicity or nonnegative-gain assumption is needed, because the comparator must select exactly \(B\) pairs.

## Uniform additive approximation

For arbitrary real pair scores \(g_e\), define \(A(S)=\sum_{e\in S}g_e\) and the interaction remainder \(R(S)=F(S)-A(S)\). Suppose estimated scores obey

\[
\max_{e\in C}|\widehat g_e-g_e|\le\delta.
\]

Let \(\widehat S\) maximize \(\widehat A(S)=\sum_{e\in S}\widehat g_e\) over the exactly-\(B\) subsets, and let \(S_F\) maximize \(F(S)\) over those subsets. Any maximizing tie choice is allowed. Then

\[
F(S_F)-F(\widehat S)
\le 2B\delta+
\left[\max_{|S|=B}R(S)-\min_{|S|=B}R(S)\right].
\tag{1}
\]

In particular, if

\[
\sup_{|S|=B}|F(S)-A(S)|\le\varepsilon_B,
\]

then

\[
\boxed{F(S_F)-F(\widehat S)\le 2B\delta+2\varepsilon_B.}
\tag{2}
\]

Only approximation on the exactly-\(B\) subsets is required for (2). A common constant offset in \(A\) or \(R\) cancels; consequently the oscillation bound (1) can be substantially sharper than a bound based on the absolute size of the remainder.

**Proof.** Insert \(A\) and \(\widehat A\) at the two sets:

\[
\begin{aligned}
F(S_F)-F(\widehat S)
={}&[A(S_F)-\widehat A(S_F)]
 +[\widehat A(S_F)-\widehat A(\widehat S)]\\
 &+[\widehat A(\widehat S)-A(\widehat S)]
 +R(S_F)-R(\widehat S).
\end{aligned}
\]

The middle estimated-score difference is nonpositive by the definition of \(\widehat S\). The first and third differences together are at most \(2B\delta\); the common edges actually cancel, giving the sharper \(2|S_F\setminus\widehat S|\delta\). Bound the remaining difference by its oscillation, or by \(2\varepsilon_B\). For \(B=0\) the two sets coincide and regret is zero. \(\square\)

The manuscript's affine risk-to-additive-gain and one-step drift assumptions imply

\[
\delta\le a(\epsilon+D_t)+\zeta.
\]

Under the **additional** current-state additive approximation assumption, (2) therefore gives

\[
F_t(S_{F,t})-F_t(\widehat S_t)
\le 2B\{a(\epsilon+D_t)+\zeta\}+2\varepsilon_{B,t}.
\tag{3}
\]

Neither the original perturbation proposition nor residual calibration supplies a bound on \(\varepsilon_{B,t}\).

## A sufficient bound from mixed finite differences

For this subsection, take the additive scores to be the actual singleton benefits:

\[
g_e=F(\{e\})-F(\varnothing).
\]

For \(e,f\notin T\), \(e\ne f\), define

\[
\Delta_f\Delta_e F(T)
=F(T\cup\{e,f\})-F(T\cup\{e\})-F(T\cup\{f\})+F(T).
\]

Suppose, uniformly over **every** such context \(T\) with \(|T|\le B-2\),

\[
|\Delta_f\Delta_e F(T)|\le\gamma.
\tag{4}
\]

Then for every set \(S\) of size \(k\le B\),

\[
\left|F(S)-F(\varnothing)-\sum_{e\in S}g_e\right|
\le {k\choose2}\gamma.
\tag{5}
\]

Consequently, for the manuscript's \(F(\varnothing)=0\), (2) implies

\[
\boxed{F(S_F)-F(\widehat S)\le 2B\delta+B(B-1)\gamma.}
\tag{6}
\]

**Proof.** Order \(S=\{e_1,\ldots,e_k\}\) and write \(P_j=\{e_1,\ldots,e_j\}\), with \(P_0=\varnothing\). Telescoping gives

\[
F(S)-F(\varnothing)-\sum_{j=1}^k g_{e_j}
=\sum_{j=1}^k[\Delta_{e_j}F(P_{j-1})-\Delta_{e_j}F(\varnothing)].
\]

For a given \(j\), telescope its bracket over \(e_1,\ldots,e_{j-1}\). Each term is one mixed difference at a context of size at most \(j-2\le B-2\), so the bracket has absolute value at most \((j-1)\gamma\). Summing gives \(\gamma\sum_{j=1}^k(j-1)={k\choose2}\gamma\). The cases \(k\le1\) are exact and need no mixed-difference assumption. \(\square\)

If the mixed differences have one sign, the oscillation form is sharper. Under \(0\le\Delta_f\Delta_eF(T)\le\gamma\), all exactly-\(B\) remainders lie in \([0,{B\choose2}\gamma]\); under \(-\gamma\le\Delta_f\Delta_eF(T)\le0\), they lie in \([-{B\choose2}\gamma,0]\). Either condition gives \(2B\delta+{B\choose2}\gamma\). Neither one-sided condition is assumed for the actual model.

Controlling only baseline pair interactions \(\Delta_f\Delta_eF(\varnothing)\) is insufficient when \(B\ge3\). For \(F(S)=M\,\mathbf1\{\{a,b,c\}\subseteq S\}\), all singleton benefits and baseline pair interactions are zero, yet the size-three benefit is \(M\); for example \(\Delta_b\Delta_cF(\{a\})=M\). Condition (4) must cover nonempty contexts.

## Perfect singleton scores can fail jointly

Let \(C=\{a,b,c,d\}\), \(B=2\), and \(M>2\). Define

\[
F(S)=\mathbf1\{a\in S\}+\mathbf1\{b\in S\}
       +M\,\mathbf1\{\{c,d\}\subseteq S\}.
\]

This set function is nonnegative, monotone, and zero on the empty set. The exact singleton gains are \((1,1,0,0)\), so with \(\delta=0\), top-singleton selection uniquely chooses \(\widehat S=\{a,b\}\), obtaining value \(2\). The optimal size-two set \(\{c,d\}\) has value \(M\). The actual regret \(M-2\) can be arbitrarily large despite zero score error and zero additive-surrogate regret.

To match an exact 25% optional-pair budget, add four dummy pairs that contribute zero. Then \(|C|=8\) and \(B=\lfloor0.25|C|\rfloor=2\), with the same conclusion. This is a set-function counterexample, not an asserted graph or WaterDrop construction.

It can also be interpreted as a reduction of a nonnegative scalar loss: take baseline loss \(L=M+2\) and loss after adding \(S\) to be \(L-F(S)\). This shows that the failure does not require negative losses or harmful edges; it does not prove that a particular trained message-passing network realizes the construction.

## Suggested concise manuscript bridge

The current text already states its limitations correctly. A short appendix insertion or two sentences after the existing proposition would be proportionate:

> For the actual fixed-state set benefit, let \(S_F\) maximize \(G_x(S)\) over exactly \(B\) optional pairs. If the additive surrogate additionally satisfies \(\sup_{|S|=B}|G_x(S)-\sum_{e\in S}g_e|\le\varepsilon_B\), then \(G_x(S_F)-G_x(\widehat S)\le2B\delta+2\varepsilon_B\), by inserting the surrogate at the two sets. This extra approximation is unverified: even perfect singleton gains can miss a complementary pair with zero singleton values and large joint benefit.

The finite-difference sufficient condition belongs in an appendix if space permits. It is a diagnostic of missing assumptions, not a reason to present the allocation theorem as evidence of effective risk-directed control. Current all-optional-pairs interventions and residual-risk correlations cannot identify a uniform \(\varepsilon_B\) or \(\gamma\) on exactly-\(B\) subsets. Dense intervention benefit is one set-level observation and does not establish singleton additivity.

## Verification

- An independent agent review checked (2), the telescoping constant in (5), the requirement to cover every context through size \(B-2\), and the exact-cardinality scope. Human author verification remains outstanding.
- Companion `research/nonadditive_allocation_toy.py` uses finite synthetic set functions only to enumerate the counterexamples and verify the inequalities on bounded quadratic interactions. It does not use any model, simulator, dataset, or test split.

Reproduce the finite synthetic checks from the repository root with:

```sh
python -m research.nonadditive_allocation_toy --output research/results/nonadditive_allocation_toy_check.json
```

The saved result records the source SHA-256 and Python version. Its repository-relative path is `research/results/nonadditive_allocation_toy_check.json`.
