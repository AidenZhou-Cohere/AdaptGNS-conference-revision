# Conference-story critique after admitted WaterDrop110k integration

Reviewed manuscript SHA256:
`0a5e83103a7d5c9f9ce9969d6f603099c7486f2e668656b263017d6a72295f42`.
Scope: admitted Goop2D100k and WaterDrop110k text; continuation-state entry
20:36 UTC. D3 observations below are conditional integration advice, not results
admission. No D3 arrays, summaries or paired outputs were inspected; no analysis,
experiment, remote action or manuscript edit was performed.

## Reviewer judgment

The strongest contribution is now a controlled distinction between **learning
to use additional interactions, placing them using residual risk, and obtaining
an autonomous benefit**. Goop and WaterDrop are useful counterpoints: exposure
improves fixed-random full rollouts in both, but their observed-test placement
orderings differ. WaterDrop's reversal and Goop's inexpensive kinematic controls
prevent an overly simple “uncertainty always fails” story. The unchanged
controller, paired histories and explicit adverse outcomes give this empirical
argument credibility.

The principal weakness is that readers can still conflate endpoints. WaterDrop's
observed-test mix-minus-base random contrast is slightly positive on average,
while its full-rollout contrast improves in every seed. These are compatible
findings. Likewise, a favorable training interaction need not mean risk beats
random, and observed placement does not establish cached autonomous placement.
The main contribution should remain this demonstrated separation; the conditional
bound and established faithful loss support it rather than supply a new effective
allocation guarantee.

## Two highest-value clarity changes

1. **Organize the results explicitly by the three questions.** Keep the existing
   Goop figure and observed-history contrast table. Replace “The observed-state
   comparison separates that training effect from placement” with wording such
   as “The matched-history comparison separately tests exposure and placement
   before autonomous feedback.” Split the existing WaterDrop paragraph after
   its observed-test/validation result; move its autonomous results into “Does
   the effect survive autonomous feedback?” beside Goop. This makes the positive
   random-policy full-rollout findings and the weaker risk-placement findings
   readable without mentally switching horizons inside one paragraph. Keep
   “observed test” explicit: the WaterDrop reversal does not hold on average on
   validation. In the eventual conclusion, name Goop and WaterDrop explicitly
   when describing their fixed-random full-rollout gains; do not imply that
   every later dataset establishes that result.

2. **Integrate D3 at the level its final audit supports.** If all prescribed
   observed histories are confirmed complete, add one short paragraph reporting
   the fixed-policy exposure contrast, mixed-arm risk-minus-random contrast and
   their interaction, with a reference to the full appendix. Preserve each
   contrast's seed signs and the distinction between observed validation and
   observed test. State 25k fresh training/H295 separately from Goop2D100k/H395
   and WaterDrop's parent-conditioned110k/H995; neither pool scales nor attribute
   differences to dimensionality. Do not add another main-text table or a new
   methodological claim.

   In the autonomous subsection, give the coverage limitation one direct
   sentence. For example, **only after confirmation**: “The Goop-3D observed-history
   comparisons are complete, but the fixed operational quotas leave autonomous
   outcomes incomplete; full-horizon means and paired training effects remain
   undefined wherever a required trajectory is missing.” If any complete
   policy/seed groups exist, retain their admitted results without averaging
   incomplete groups or presenting them as the complete study. Operational
   timeout is an adverse compute/coverage outcome, not by itself a physical
   instability or an efficacy loss. Scientific guards, timed-out current cells,
   other incomplete cells and never-started cells must remain distinct in the
   appendix/results. No partial-prefix mean, survivor-only aggregate or earlier
   horizon should replace the missing declared endpoint.

The 4728-cell D3 ledger mixes rollout and diagnostic cell types; its aggregate
completion count is not a rollout denominator or a useful main-text headline.
Keep operational history, source hashes and detailed coverage outside the main
narrative. Fund at most roughly100 new D3 words by removing duplicated Goop
failure prose from the figure caption and repeated synthesis, while retaining
the visible failure table and nearby explanation. Preserve the eight-page cap
and the existing explanatory figure; more tables would obscure the argument.

The current abstract still emphasizes the small CPU pilot rather than the
admitted paired studies. The existing separate candidate at
`work/deadline_research_20261005/waterdrop110k_integration_candidate_code_audit_v1/candidate_abstract_dated_20261006.md`
better matches the current evidence, but title and abstract should remain
unchanged until the author verifies the registered text. Sand and D3's pending
scientific results should not enter that candidate or manuscript by inference.
