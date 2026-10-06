# Goop global graph-action gate: prospective protocol v1

**Frozen prospective scientific contract; not permission to execute.** Root selected this candidate before the new gate test outcomes. Use only the three already planned Goop2D **mix**, faithful, seed0/1/2 mean models at exactly100000 updates. Hash-bind all three checkpoints and the original graph-support cohort/audit. Mean and variance parameters remain unchanged. This is a separate follow-up; preserve every original arm/policy result, failed attempt and earlier protocol. Do not select the arm, seeds, features, label schedule, ridge grid, controls, endpoint or outcome subset using test results.

## Decision and estimand

At each current six-frame history x, choose between two existing actions: native capped base graph, or native base plus exactly floor(0.25*annulus_count) random optional unordered pairs, emitted both ways without recapping. Use the frozen strict-r/1.267r annulus, native directed prefix, cap128/self behavior, and all original numerical guards. A zero optional budget returns the identical base graph. No individual pair utility is estimated.

For a clean observed training history, label

`b(x,u) = mean_particles,coordinates[(f_base(x)-y)^2] - mean_particles,coordinates[(f_random25(x,u)-y)^2]`.

Positive b means this complete random graph action improved one-step position MSE. Both actions use the identical frozen model, input history and next-position target, with no noise. Compute predictions before reading the target into the scoring function. Label arithmetic is float64 on the returned prediction and truth values. With one randomized action per state, the fitted state-only predictor targets expected action benefit under that randomization; it does not identify marginal edge utility.

The deployed rule requests random25 iff predicted b is strictly positive. It receives only current history and fixed bounds. No future target, current prediction, variance output, graph-action result, processor activation, previous network cache, source ID, forecast index or RNG draw is a predictor feature. There is one simulator forward per forecast and no discarded score forward or neural warmup. This removes an input/cache mismatch; teacher-forced versus autonomous state-distribution shift remains a limitation to measure.

## Exact feature contract

Input has shape[6,N,2], N>0, with all finite coordinates. Apply the unchanged scientific state guard before feature extraction. Convert coordinates to float64 for these calculations. Let `v_i=x[5,i]-x[4,i]`, `a_i=x[5,i]-2*x[4,i]+x[3,i]`; these are per-stored-step differences, without dividing by dt. Let `s_i=||v_i||_2`, `c_i=||a_i||_2`, and `d_i=min(x[5,i,0]-.1,.9-x[5,i,0],x[5,i,1]-.1,.9-x[5,i,1])`. Quantiles use NumPy's explicit `method='linear'`. Fixed feature order is:

1. `log1p_particle_count = log(1+N)`.
2. `speed_mean = mean(s_i)`.
3. `speed_p90 = quantile(s_i,.9)`.
4. `speed_max = max(s_i)`.
5. `acceleration_difference_mean = mean(c_i)`.
6. `acceleration_difference_p90 = quantile(c_i,.9)`.
7. `acceleration_difference_max = max(c_i)`.
8. `signed_wall_distance_mean = mean(d_i)`.
9. `signed_wall_distance_p10 = quantile(d_i,.1)`.
10. `outside_particle_fraction = mean(d_i<0)`.
11. `maximum_coordinate_excursion = max(max(-d_i,0))`.

There are no graph-derived features or duplicate neighbor queries. Feature work is O(N). Boundary metrics are geometric diagnostics; none is a physical conservation claim. No clipping or outcome-dependent feature removal is allowed. Reject nonfinite features and preserve the failed input identity.

## Train labels, ridge fit and validation selection

Use all1000 admitted official Goop train sources in original order at target frames `[6,62,118,174,231,287,343,400]` (eight floor-spaced indices spanning6..400). Current history is positions[target-6:target]; truth is positions[target]. There are8000 required states per model,24000 overall, and48000 paired-action simulator forwards. Every scheduled base/random label must be committed and finite, with original history/graph/prediction/source/model hashes and both measured action costs. **Any guarded, failed, missing or uncommitted scheduled label stops the gate study before fitting or validation selection.** Do not fit on survivors, invent a numeric failure label, replace a source, or retry a numerical failure. Retain the partial collection and its failure ledger.

For each seed separately, use all8000 complete train states with equal weight. Let feature mean `mu_j` and population standard deviation `sigma_j` use denominator8000; `scale_j=max(sigma_j,1e-12)`, `z_ij=(x_ij-mu_j)/scale_j`. Let label mean `m` and population standard deviation `s`, with `label_scale=max(s,1e-12)` and `t_i=(b_i-m)/label_scale`. Retain all11 columns, including constant columns. For each lambda in **`[0.001,0.1,10.0]`**, solve in float64

`w_lambda = argmin_w (1/n)*sum_i (z_i @ w - t_i)^2 + lambda*||w||_2^2`.

Use the direct11×11 solve `(Z.T@Z/n + lambda*I) w = Z.T@t/n`. The unpenalized intercept is represented by training centering; prediction in original MSE units is `m + label_scale*(z@w)`. No optimizer epochs, random initialization, feature search or alternate loss is permitted. Serialize the raw train normalizers, coefficients, lambda, label/feature schemas and input hashes. Nonfinite solve/output stops the study. Verify frozen simulator checkpoint bytes before/after fitting.

Validation uses all30 validation sources at targets `[7,105,203,301,400]`, exactly150 paired-action states per model. Use clean observed histories and the same feature/label definitions. All450 paired validation states must be complete and finite; otherwise stop before selection. Validation outcomes never enter ridge fitting or feature/label normalization.

For each lambda, compute the actual selected-action coordinate MSE on every validation state: random25 error iff the corresponding fitted head predicts b>0, otherwise base error. Average the five frames within source,30 sources within seed, then three seeds equally. Select **one common lambda** minimizing this decision loss. Break exact float64 equality by the larger lambda; no tolerance band or manual preference. Threshold remains exactly0. Hash-freeze all three selected heads, the selection receipt, the complete train/validation ledgers and the prospective test driver before any gate test access. Original validation was already used elsewhere; this is exploratory selection, not a fresh independent confirmation split.

For each seed, set `p_seed` to the fraction of its150 validation states for which the selected head requested expansion (`predicted_b>0`), including zero-budget requests. Freeze p with the head. Do not recalibrate it on test states or outcomes.

## RNG domains and four matched-protocol policies

Use NumPy `SeedSequence` with these integer lists; preserve material, package versions and selected-pair hashes. No policy identity enters pair RNG material. The frozen graph helper performs the actual uniform-without-replacement selection.

- Train-label optional-pair RNG: `[20261006,27101,model_seed,source_index,target_frame,1701]`.
- Validation paired-action optional-pair RNG: `[20261006,27102,model_seed,source_index,target_frame,1701]`.
- Test optional-pair RNG at each forecast step1..395: `[20261006,27103,model_seed,source_index,forecast_step,1701]`.
- Test randomized-gate decision draw: `[20261006,27103,model_seed,source_index,forecast_step,2909]`; request expansion iff the first uniform draw is `<p_seed`.
- Train-only capacity rollout optional-pair RNG: `[20261006,27104,model_seed,source_index,forecast_step,1701]`; randomized-gate draw uses the same material with final2909.

Instantiate RNG independently for every declared state/step, so skipped expansion never advances or shifts later random choices. Policies on differing autonomous states share random material, not a fictitious identical graph. Four fixed policies are **base**, **random25**, **learned_global_gate**, **validation_rate_random_gate**. All use the same mean checkpoint and compatible graph helper. Re-run both constants under this new RNG/source/measurement protocol; do not import incompatible historical baseline rows.

The random gate is **validation-rate-matched, not test-cost-matched**. Autonomous histories, annulus sizes, realized decision rates and retained-edge counts may differ. Report requested expansion fraction, zero-budget fraction, actual nonzero expansion fraction, added pairs/edges and end-to-end runtime. A matched-rate contrast does not by itself establish equal deployed compute; no per-test matching or recalibration is allowed.

## Full evaluation and failure accounting

Evaluate all30 admitted test sources for all395 autonomous forecasts after the initial six stored frames, allthree selected heads/seeds and allfour policies: **360 full trajectory-policy outcomes and142200 forecast forwards**. Native parity/setup calls, fitting, label generation and serialization are additional and separately measured. At each step produce the action, graph and forecast before scoring against truth; only the predicted state advances history. Never feed the target into the deployed gate or next history.

Primary scalar comparisons are learned_global_gate minus each of base, random25 and validation_rate_random_gate full-H395 MSE. Mean coordinates/particles within forecast, all395 forecasts within source,30 sources equally within seed, then allthree seed effects equally. Report every seed value, mean and sampleSD, all absolute policy values, physical boundary excursions, realized costs, guard categories and missing work. H200 is a separate pointwise secondary endpoint and may survive an accepted prefix that later fails; it never substitutes for full-H395 error.

Retain all360 planned cells as committed-complete, committed-guard-failed, timed-out current, uncompleted after started invocation or never started. Undefined full-horizon values and any required incomplete paired mean remain null. Do not compare quota survivors as full-source performance or relabel timeout as numerical instability. No automatic retries, seed replacement, shortened endpoint, source omission or policy deletion.

Always-base collapse is a possible negative result, not successful adaptation. Beating always-expanded random25 while merely matching base does not establish an accuracy gain from graph adaptation. Beating the state-independent gate is evidence that conditioning helps under the reported realized costs; beating both constant actions is the stronger constructive outcome. A null result remains informative within this feature class, one-step target and autonomous distribution shift; it does not establish that action-value learning generally fails.

Report validation one-step paired-action oracle regret and signed-benefit calibration only as diagnostics. An oracle selecting from already computed validation action outcomes is not a deployable policy. No test oracle is used for tuning or performance claims.

## Complete measured cost gate and operational release

No scientific execution is authorized by this file. Root first admits source/unit/parity review and a **train-only** preparation phase; this does not admit any gate test access. A fixed64-state label capacity sample per seed uses flattened schedule indices `floor(i*(8000-1)/63)`, i=0..63. All its paired actions, source checks, feature work and serialization must complete. Use the slowest observed paired-state wall rate, measured input/load overhead and an explicit1.35 engineering margin to budget the complete8000-state label generation per model. This is a planning allowance, not a worst-case bound. Preserve all capacity outcomes; no failed sample is dropped or replaced.

The prospective complete workload includes full train-label collection, float64 fitting, all450 validation states/common-lambda selection, immutable head/publication audits, and all360 test outcomes. After fitting/freezing, the **actual four deployed policies** must complete train-only full-H395 capacity rollouts on three training sources selected by manifest particle count: minimum, lower median and maximum, ties by original source index (if an index repeats, take the next unused source in that sorted order). This gives36 train capacity trajectory-policy outcomes across three seeds, with normal graph/state guards and production publication enabled. It is not a new test set. Preserve full outcomes. Any failure, missing outcome, or nonfinite cost leaves the complete rollout forecast undefined and prevents test admission under this protocol; no failed gate is reinterpreted as a passing forecast.

For seed s and policy p, let `r_sp` be the maximum observed full trajectory wall duration across those three sources, including graph/feature/forward/trace/publication work. Measure per-model process/load setup separately. Planned test allocation per seed is `R_s = measured_setup_s + 1.35*30*sum_p(r_sp)`. Root must include actual source/cohort/head revalidation, transfers and binary-hash costs, not just network calls. This three-source extrapolation is limited engineering evidence; it does not guarantee held-out dynamics or success. Enforce reviewed whole-invocation allocations and preserve incomplete outcomes if actual cost exceeds them.

Use B GPUs0/1 only after the original Goop children and their parent monitoring loop are conclusively reaped; never live-handoff or disturb Sand GPUs2/3. Fixed two-wave placement: GPU0 runs seed0 then seed2; GPU1 runs seed1. One process/model, two CPU threads, same immutable native CUDA/package/environment lineage. The concurrent test allocation is `max(R_0+R_2,R_1)` plus bounded per-invocation cleanup. Outside it reserve900seconds for remaining input/preflight work,2700seconds for complete scalar/array-byte collection, and3600seconds for independent analysis/manuscript integration. Charges already completed before final release are recorded as spent, not silently charged twice or omitted from total study cost. If measured work does not fit, decline the study as a whole; do not reduce seeds/policies/sources/horizon.

Root's separate final test release must bind this protocol, all helper/driver/checkpoint/head/data/selection/capacity hashes, the exact remaining-cost arithmetic and clock/owned-process evidence. **Complete compute and analysis by October7 04:00UTC**, preserve04:00–08:00 for writing/author checks and the08:00 no-new-work cutoff. Existing Goop/Sand/D3 completion and their reserved analysis must not be displaced by this candidate. Final public accounting distinguishes arrays byte-verified and retained on the VM from scalar JSON transferred locally. No conference-readiness or submission claim follows from completing this study.
