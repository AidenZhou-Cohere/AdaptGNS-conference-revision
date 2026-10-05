# Audited actual-action benefit findings

This is an exploratory postprocess specified after the fixed experiment. It evaluates the same 1,782 observed-history cases and all six frozen checkpoints; it is neither a new independent test nor autonomous cached-risk inference. All means below are equal frames within trajectories, equal trajectories within seeds, then three seed means. Sample SD is descriptive.

An independent implementation reopened every original/derived archive and reconstructed signed errors, the alignment-minus-perturbation identity, dense/actual sign cross-tabs, whole-frame oracle choices, optional-degree statistics, tied risk quartiles, all trajectory means, seed summaries, and paired differences. It passed 79,192,129 element/structure checks across all 1,782 frames. All 3,582 original input files and 3,568 analysis files were unchanged. Maximum arithmetic difference was 7.11e-15. The 65.54-second audit is verification time, not policy runtime. There were no failed audit attempts in this audit family.

## Mechanistic implications

1. **Changing the benefit label does not produce a positive average rank association.** Previous-risk versus its actual selected action has mean Spearman −0.0212 for faithful and −0.0518 for NLL. Faithful seed signs differ; all three NLL values are negative. These describe the declared whole-graph action, not causal risk utility or arbitrary-edge marginal values.
2. **Dense benefit is materially different from the actual action.** Dense and risk25 benefit signs disagree on mean within-frame particle fractions 30.20% / 31.30% (faithful/NLL); strictly opposite nonzero signs account for 28.81% / 28.91%. Thus dense is an imperfect label even though average risk rank correlations happen to be similar.
3. **A one-frame observed-history lag is not the main apparent limitation here.** Current/previous risk Spearman agreement is approximately 0.9888 / 0.9913. The actual-action correlation changes by only −0.00041 / +0.00117 when using current rather than previous risk. This does not rule out autonomous selected-graph cache drift or rollout distribution shift.
4. **Weak Spearman hides nonmonotone/tail structure.** The highest previous-risk quartile has positive mean actual-risk benefit for both objectives, but the NLL seed-1 quartile mean is negative. Random25 produces greater mean benefit in that same high-risk group in all six individual models. Middle quartiles are harmed by risk25 in every seed. The evidence does not justify saying risk contains no useful location information.
5. **Always spending the optional budget can be harmful.** Base is strictly better than all three budgeted actions on 35.91% / 35.80% of frames on average. Adding base to the hindsight menu lowers normalized coordinate MSE by 0.0002152 / 0.0002177. This is a target-using upper envelope over four saved whole-frame choices, not a deployable abstention policy, a cost-matched dense comparison, or an optimum over arbitrary edge subsets.
6. **Allocation structure is an unresolved mechanism.** Risk25 gives optional edges to 43.47% / 41.79% of particles versus random25’s 63.13%, with optional-degree Gini 0.7190 / 0.7251 versus 0.5262. Yet speed25 is still more concentrated (32.96% coverage, Gini 0.7784), while its faithful autonomous result was stronger. Concentration alone is not an established causal explanation.

## Seed values

| Objective | Diagnostic | Seeds 0, 1, 2 | Mean ± sample SD |
|---|---|---|---|
| faithful | Risk / own-action benefit Spearman | -0.007221754, 0.01895811, -0.07537869 | -0.02121411 ± 0.0487 |
| faithful | Current / previous risk Spearman | 0.989309, 0.9885213, 0.9885281 | 0.9887861 ± 0.0004528 |
| faithful | Dense/actual sign disagreement fraction | 0.3074275, 0.2957689, 0.3028306 | 0.302009 ± 0.005873 |
| faithful | Dense/actual opposite nonzero fraction | 0.2895935, 0.2857721, 0.2890257 | 0.2881304 ± 0.002062 |
| faithful | Risk25 normalized vector benefit | 0.0004325272, 0.0008733655, -0.0006055995 | 0.0002334311 ± 0.0007593 |
| faithful | Random25 normalized vector benefit | 0.001621054, 0.001617236, 0.0003782955 | 0.001205529 ± 0.0007164 |
| faithful | Risk25 whole-frame harmful fraction | 0.5690236, 0.4646465, 0.6363636 | 0.5566779 ± 0.08652 |
| faithful | Base strictly best frame fraction | 0.3535354, 0.2424242, 0.4814815 | 0.359147 ± 0.1196 |
| faithful | Hindsight benefit from allowing base | 0.0001916197, 0.0001388888, 0.0003149653 | 0.0002151579 ± 9.037e-05 |
| faithful | Risk25 benefit in highest risk quartile | 0.0037017, 0.004995755, 0.0008132257 | 0.003170227 ± 0.002141 |
| faithful | Random25 benefit in highest risk quartile | 0.00609207, 0.005850534, 0.002714915 | 0.004885839 ± 0.001884 |
| nll | Risk / own-action benefit Spearman | -0.03828926, -0.05267574, -0.06438815 | -0.05178438 ± 0.01307 |
| nll | Current / previous risk Spearman | 0.9915191, 0.9909845, 0.9913537 | 0.9912858 ± 0.0002737 |
| nll | Dense/actual sign disagreement fraction | 0.3095955, 0.3113336, 0.3181528 | 0.3130273 ± 0.004523 |
| nll | Dense/actual opposite nonzero fraction | 0.2889505, 0.2844829, 0.2939486 | 0.2891273 ± 0.004735 |
| nll | Risk25 normalized vector benefit | 0.0008219212, -0.001053257, -0.0002998545 | -0.0001770636 ± 0.0009436 |
| nll | Random25 normalized vector benefit | 0.002312356, 0.0008259188, 0.001490838 | 0.001543038 ± 0.0007446 |
| nll | Risk25 whole-frame harmful fraction | 0.4882155, 0.6329966, 0.5993266 | 0.5735129 ± 0.07576 |
| nll | Base strictly best frame fraction | 0.2861953, 0.4006734, 0.3872054 | 0.3580247 ± 0.06257 |
| nll | Hindsight benefit from allowing base | 0.000152866, 0.0003042616, 0.000195828 | 0.0002176519 ± 7.802e-05 |
| nll | Risk25 benefit in highest risk quartile | 0.00515538, -0.0007401979, 0.001896443 | 0.002103875 ± 0.002953 |
| nll | Random25 benefit in highest risk quartile | 0.008874806, 0.00368183, 0.004824857 | 0.005793831 ± 0.002729 |

## Next high-value experiment

After the graph-convention bridge resolves active self-loop/cap effects, test a small fixed budget grid with repeated uniform-random and risk-permutation controls. Preserve the max-endpoint scoring mechanism while permuting particle scores (with simple predeclared geometry/degree strata), and measure how much optional-degree concentration changes. The question is whether useful high-risk regions are being over-perturbed or poorly connected by this score-to-edge map, rather than whether a lucky new selector can improve one summary. Permuted risk is a better structure control than uniform random edges but does not guarantee identical degree sequences; report the remaining differences.
Use all six checkpoints and all fixed observed states, preserve every draw, and compare current/previous risk and random controls on the same budgets. Do not select a test-optimal budget and call it independent confirmation. A fixed at-most-budget abstention candidate needs training-only decision fitting and a separate documented evaluation. Existing teacher-forced evidence cannot establish its autonomous benefit.

## Provenance and limits

Analysis result SHA256: `c9a608d3f17cfc84001020b27048ef448958daa2705d135be5f4492e8bbdc008`.
Frozen analyzer SHA256: `e8f167646986e08b650e58b4388cd56e5277ec200708eb3d9788eaedf39c34b3` (source/protocol freeze commit `6ed731813373e01e8638bb1495da3c902e8c631c`).
All original arrays and locked results remain unmodified. The optional-degree and risk-stratum results are observational descriptions of the actual saved interventions. No p-values, new policy inference, training, speed claim, independent confirmation, or submission-readiness claim is made.
