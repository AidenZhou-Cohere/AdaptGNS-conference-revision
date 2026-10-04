# Exact budgets and permutation symmetry

This explanatory note does not change the frozen [pair selector](../budget_graph.py) or claim novelty. Tie frequencies and effects on current simulation accuracy, stability or runtime have not been measured.

Let \(X\) include all permitted selector information, including candidate pairs, node scores and controller memory. Unordered pair sets relabel equivariantly, and the exact integer budget \(B(X)\) is invariant. Particle array indices are labels, not distinguishing physical features. A deterministic selector is equivariant when \(S(\pi X)=\pi S(X)\).

## Sharp obstruction

The input stabilizer \(G_X=\{\pi:\pi X=X\}\) acts on optional pairs, with orbits \(O_1,\ldots,O_m\). An equivariant deterministic exact-\(B\) selector exists on the relabeling orbit of \(X\) if and only if \(B\) is a sum of sizes of a subset of these orbits.

For necessity, every stabilizer element satisfies \(S(X)=S(gX)=gS(X)\), so the selected set must be a union of orbits. For sufficiency, choose a union of size \(B\) at a representative \(X_0\) and define \(S(\pi X_0)=\pi S(X_0)\). Stabilizer invariance makes this independent of the permutation chosen. This establishes existence, not efficiency, continuity or locality.

For top-priority selection, retain pairs strictly above the cutoff and apply the same orbit-size condition to the remaining budget within the cutoff tie block. A tie block can contain multiple stabilizer orbits: equal priorities are weaker than full input symmetry.

A four-leaf optional star with indistinguishable leaves has one pair orbit of size four. An exact 25% budget selects one pair and is impossible for a deterministic equivariant selector on that input. Distinct absolute coordinates can destroy the stabilizer under pure node relabeling, so a visually symmetric drawing alone is insufficient. The full permitted input and symmetry action must be stated.

## Max-score ties do not always force randomization

For \(w_{ij}=\max(s_i,s_j)\), a node gives the same priority to all incident optional pairs whose other endpoint has lower score. With distinct node scores, these sets form tie blocks of sizes
\[
d_i^-=|\{j:\{i,j\}\in E_+,\ s_j<s_i\}|.
\]
The budget can split a block even without equal node scores or rounding. The frozen stable score sort inherits lexicographic particle-index ordering and can therefore break equivariance.

This is not always unavoidable. If the star's scores are \((5,1,2,3,4)\), all four max priorities tie, but the leaf risks distinguish the pairs. Ordering by \((\max(s_i,s_j),\min(s_i,s_j))\) uniquely orders unordered pairs whenever node scores are distinct, preserving the primary priority and permutation equivariance. It does not resolve genuinely indistinguishable pairs. This alternative was not installed or evaluated.

## Random ties with explicit coupling

Assign iid continuous keys \(U_e\) to unordered pairs, independently of the input, and use them only as secondary tie breakers. Equivalently, if the cutoff block has \(t\) pairs and \(b\) remaining slots, sample a uniform \(b\)-subset. The budget is exact, each cutoff pair has inclusion probability \(b/t\), and all strict priority rankings are preserved. Every realization has the same summed priority, but can have different simulation error.

Under the transported-key coupling \((\pi U)_{\pi e}=U_e\), corresponding pairs have identical scores and keys, so almost surely
\[
S(\pi X,\pi U)=\pi S(X,U).
\]
Since the iid key field retains its law after relabeling, this yields equivariance in distribution. Randomness augments the input, so this does not contradict the deterministic obstruction.

Restarting the same scalar PRNG seed on a newly sorted pair array generally assigns different keys to corresponding physical pairs. It does not provide that pathwise coupling. Finite-key collisions require symmetric resolution or a sampled ordering without replacement; an index fallback weakens an exact guarantee. Adding jitter to scores can reorder unequal priorities and changes the policy.

With fresh independent keys per time step, the statement holds conditionally at each forecast. Coupling keys for every pair and time yields coupled pathwise rollouts if the remaining controller and dynamics are equivariant. Persistent pair keys may preserve the joint trajectory-law symmetry, but can become dependent on current predicted states; conditional uniform tie sampling then does not follow automatically.

Random cutoff ties differ from the existing `random25` control, which samples across all optional pairs. Any change needs its own declared evaluation. These statements address selection symmetry, not calibration, useful allocation or physical accuracy.
