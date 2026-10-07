# Source-only observed checker diagnosis

The frozen checker contains a reproducible floating-point defect in its three-seed sample-standard-deviation calculation. This is a source finding with synthetic witnesses. The exact failed leaf from the real cache is being independently diagnosed by the reviewer; this note does not infer it from the traceback alone.

The unchanged summarizer `stats` uses `statistics.stdev(values)`. Python computes the sample variance from exact ratios of the input numbers before returning a floating result. The unchanged checker `seed_values` instead computes `m = math.fsum(items)/3`, then `sqrt(math.fsum((x-m)**2)/2)`. The latter centers around an already-rounded floating mean.

When all three input values are identical, the true sample SD is exactly zero. But `fsum([x,x,x])/3` can differ from `x` by one floating-point unit. In that case the checker computes a false positive SD. For three samples, if the rounded mean differs from the exact mean by delta, centering at that mean adds `3*delta^2/2` to the sample variance. Near-zero SD is therefore especially vulnerable, even though the published mean is accurate to floating precision.

Concrete synthetic witnesses under the unchanged `rel_tol=1e-10, abs_tol=1e-12` comparison:

| Three identical seed values | Summarizer SD | Frozen checker SD | Result |
| --- | ---: | ---: | --- |
| 100000.1 | 0 | 1.7822383544867465e-11 | Rejected |
| 1651982.6 | 0 | 2.8515813671787943e-10 | Rejected |
| 100000000.1 | 0 | 1.8250120749944284e-08 | Rejected |

The source-level frame-to-trajectory and trajectory-to-model reductions agree: both the successor merge's arithmetic helper and the independent checker use `math.fsum(values)/len(values)`, preserve fixed denominators, and propagate undefined values. The summarizer's `statistics.fmean` and the checker's `average` also use that same floating sum/division for the three-seed mean. The variance centering is the concrete discrepancy isolated here.

## Minimal correction proposal

Keep the frozen checker, summarizer, original failure and all 2,568 passing row checkpoints unchanged. Introduce a separately hashed checker successor that changes only the independent sample-variance calculation inside `seed_values`.

For the three finite seed values, represent each input exactly as a `Fraction` and compute the unbiased sample variance as:

`((x0-x1)^2 + (x0-x2)^2 + (x1-x2)^2) / 6`.

This is algebraically the same sample variance with denominator n-1=2; it avoids centering at a rounded mean. The proposal uses a 60-digit Decimal square root of that exact nonnegative rational, then converts the result to float. The Decimal square root also avoids overflow/underflow from converting the variance to float before its square root. It is independent of the summarizer's `statistics.stdev` implementation.

Do not change seed values, counts, null rules, family-completion rules, fixed source denominators, contrasts, means, or the existing comparison tolerances. The proposal passes ten synthetic cases, including identical large values, small nonzero spread, ordinary contrasts, undefined seeds and variance underflow/overflow stress. No real array, source trajectory or model was loaded. The original checker exhibits three mismatches and one overflow exception in these probes; all outcomes are retained in the probe JSON.

`proposed_exact_seed_stats.py` is only an isolated proposal, not an edit to the frozen checker. `source_probe.py` extracts only the frozen pure reduction functions by AST and compares them on invented values. `synthetic_source_probe.json` records exact source hashes and every old/new result.

Once the reviewer confirms the actual cached mismatch is this SD reduction, a successor can rebuild merge/summary/check products from the hash-verified passing row cache and original scalar metadata. It must retain the original failed invocation and identify the checker correction explicitly. It must not repeat saved-array audits or weaken the equality tolerance merely to pass.


## Independent actual-cache confirmation

The independent reviewer subsequently checked the retrieved 2,568-row cache and reported 5,528 statistic leaves compared under the original tolerance, with 36 mismatches. Every mismatch was a sample SD for a graph-count metric with three bit-identical seed means; exact variance and the summarizer SD were both zero. No accuracy mean or contrast mismatch was found.

The first reported case is `same_state_valid / base / graph/base/candidate_pairs`, with seed values `[102089.95333333334, 102089.95333333334, 102089.95333333334]`; the rounded checker mean differs by one unit and produces false SD `1.7822383544867465e-11`. The independent diagnosis is `../review_v1/observed_arithmetic_diagnosis_v1/diagnosis.json`, SHA256 `02685d533fff4f3e43da3eb7354003b315bf5f376fbcb5971f84b40993821375`.

This confirmation supports a separately versioned exact-variance checker correction and finalization from the existing cache, while keeping the failed original check visible.
