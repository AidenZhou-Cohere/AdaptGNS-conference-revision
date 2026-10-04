# Residual risk versus observed physical descriptors

**Completed exploratory postprocess of all 15 compact-pilot models; no model calls or new training.** The same 36 validation and 36 previously inspected test frames were reused for every model. These are kinematic descriptors, not ground-truth physical complexity or epistemic uncertainty.

Strain and vorticity come from a local affine velocity fit using only observed positions and displacements before the prediction target. The radius is 0.015, with at least three neighbors, spatial rank two and Gram condition at most 10⁶. Units are per stored frame; constant positive time rescaling leaves ranks unchanged. Adjusted coefficients correlate rank residuals after regressing on ranked neighbor count, speed and signed box-plane clearance. This descriptive adjustment does not establish causality.

Coefficients are averaged within each trajectory, then equally across three trajectories and five training seeds. Values after ± are sample seed SD, not confidence intervals. No particle-level p-values are computed. A required undefined frame makes the unconditional aggregate undefined. Conditional defined-frame results below still require all three trajectories and all five seeds.

## Geometry coverage

| Split | Unique frames | Eligible particle–frame pairs / total | Fit coverage range across frames | Outside-box pairs | No-neighbor pairs |
|---|---:|---:|---:|---:|---:|
| valid | 36 | 15,908 / 16,572 (95.99%) | 90.32%–98.96% | 389 | 37 |
| test | 36 | 17,506 / 18,096 (96.74%) | 88.05%–99.66% | 596 | 53 |

Masks are identical across models. Per-frame counts and the mean/median risk of included and excluded particles are preserved in JSON. Particle–frame pairs are repeated observations, not independent samples. Box clearance is not a free-surface detector.


## Unconditional associations

| Descriptor | faithful/valid | faithful/test | nll/valid | nll/test | beta_nll/valid | beta_nll/test |
|---|---:|---:|---:|---:|---:|---:|
| Strain norm | 0.312 ± 0.041 | 0.236 ± 0.021 | 0.442 ± 0.037 | 0.397 ± 0.057 | 0.371 ± 0.020 | 0.310 ± 0.011 |
| Absolute vorticity | 0.181 ± 0.032 | 0.098 ± 0.009 | 0.212 ± 0.036 | 0.213 ± 0.052 | 0.191 ± 0.022 | 0.157 ± 0.015 |
| Strain, adjusted ranks | 0.160 ± 0.027 | 0.167 ± 0.025 | 0.311 ± 0.033 | 0.283 ± 0.059 | 0.232 ± 0.028 | 0.196 ± 0.014 |
| Vorticity, adjusted ranks | 0.084 ± 0.026 | 0.041 ± 0.019 | 0.077 ± 0.024 | 0.061 ± 0.053 | 0.067 ± 0.020 | 0.026 ± 0.016 |
| Absolute divergence | 0.094 ± 0.048 | 0.114 ± 0.063 | 0.209 ± 0.006 | 0.223 ± 0.012 | 0.179 ± 0.024 | 0.195 ± 0.045 |
| Neighbor velocity RMS difference | 0.281 ± 0.025 | 0.217 ± 0.011 | 0.406 ± 0.037 | 0.365 ± 0.054 | 0.333 ± 0.022 | 0.275 ± 0.010 |
| Base error (same fit mask) | 0.237 ± 0.145 | 0.254 ± 0.150 | 0.408 ± 0.060 | 0.411 ± 0.075 | 0.326 ± 0.036 | 0.327 ± 0.052 |
| Dense benefit (same fit mask) | 0.008 ± 0.179 | 0.025 ± 0.193 | -0.037 ± 0.042 | -0.051 ± 0.045 | -0.050 ± 0.059 | -0.042 ± 0.058 |
| Neighbor count (all particles) | -0.343 ± 0.229 | -0.254 ± 0.262 | -0.286 ± 0.021 | -0.221 ± 0.024 | -0.392 ± 0.103 | -0.340 ± 0.134 |
| Speed (all particles) | 0.542 ± 0.029 | 0.432 ± 0.032 | 0.494 ± 0.030 | 0.443 ± 0.034 | 0.485 ± 0.053 | 0.392 ± 0.055 |
| Signed wall clearance (all particles) | 0.314 ± 0.078 | 0.436 ± 0.077 | 0.240 ± 0.037 | 0.232 ± 0.027 | 0.220 ± 0.072 | 0.217 ± 0.075 |
| Observed acceleration (all particles) | 0.338 ± 0.056 | 0.354 ± 0.045 | 0.536 ± 0.036 | 0.554 ± 0.014 | 0.439 ± 0.055 | 0.467 ± 0.049 |
| Base error (all particles) | 0.241 ± 0.145 | 0.265 ± 0.152 | 0.414 ± 0.056 | 0.419 ± 0.074 | 0.330 ± 0.037 | 0.336 ± 0.052 |
| Dense benefit (all particles) | 0.007 ± 0.178 | 0.024 ± 0.195 | -0.029 ± 0.038 | -0.041 ± 0.042 | -0.030 ± 0.059 | -0.022 ± 0.055 |

## Defined-frame conditional associations

| Descriptor | faithful/valid | faithful/test | nll/valid | nll/test | beta_nll/valid | beta_nll/test |
|---|---:|---:|---:|---:|---:|---:|
| Strain norm | 0.312 ± 0.041 | 0.236 ± 0.021 | 0.442 ± 0.037 | 0.397 ± 0.057 | 0.371 ± 0.020 | 0.310 ± 0.011 |
| Absolute vorticity | 0.181 ± 0.032 | 0.098 ± 0.009 | 0.212 ± 0.036 | 0.213 ± 0.052 | 0.191 ± 0.022 | 0.157 ± 0.015 |
| Strain, adjusted ranks | 0.160 ± 0.027 | 0.167 ± 0.025 | 0.311 ± 0.033 | 0.283 ± 0.059 | 0.232 ± 0.028 | 0.196 ± 0.014 |
| Vorticity, adjusted ranks | 0.084 ± 0.026 | 0.041 ± 0.019 | 0.077 ± 0.024 | 0.061 ± 0.053 | 0.067 ± 0.020 | 0.026 ± 0.016 |
| Absolute divergence | 0.094 ± 0.048 | 0.114 ± 0.063 | 0.209 ± 0.006 | 0.223 ± 0.012 | 0.179 ± 0.024 | 0.195 ± 0.045 |
| Neighbor velocity RMS difference | 0.281 ± 0.025 | 0.217 ± 0.011 | 0.406 ± 0.037 | 0.365 ± 0.054 | 0.333 ± 0.022 | 0.275 ± 0.010 |
| Base error (same fit mask) | 0.237 ± 0.145 | 0.254 ± 0.150 | 0.408 ± 0.060 | 0.411 ± 0.075 | 0.326 ± 0.036 | 0.327 ± 0.052 |
| Dense benefit (same fit mask) | 0.008 ± 0.179 | 0.025 ± 0.193 | -0.037 ± 0.042 | -0.051 ± 0.045 | -0.050 ± 0.059 | -0.042 ± 0.058 |
| Neighbor count (all particles) | -0.343 ± 0.229 | -0.254 ± 0.262 | -0.286 ± 0.021 | -0.221 ± 0.024 | -0.392 ± 0.103 | -0.340 ± 0.134 |
| Speed (all particles) | 0.542 ± 0.029 | 0.432 ± 0.032 | 0.494 ± 0.030 | 0.443 ± 0.034 | 0.485 ± 0.053 | 0.392 ± 0.055 |
| Signed wall clearance (all particles) | 0.314 ± 0.078 | 0.436 ± 0.077 | 0.240 ± 0.037 | 0.232 ± 0.027 | 0.220 ± 0.072 | 0.217 ± 0.075 |
| Observed acceleration (all particles) | 0.338 ± 0.056 | 0.354 ± 0.045 | 0.536 ± 0.036 | 0.554 ± 0.014 | 0.439 ± 0.055 | 0.467 ± 0.049 |
| Base error (all particles) | 0.241 ± 0.145 | 0.265 ± 0.152 | 0.414 ± 0.056 | 0.419 ± 0.074 | 0.330 ± 0.037 | 0.336 ± 0.052 |
| Dense benefit (all particles) | 0.007 ± 0.178 | 0.024 ± 0.195 | -0.029 ± 0.038 | -0.041 ± 0.042 | -0.030 ± 0.059 | -0.022 ± 0.055 |

Defined frame counts per seed (out of 36 scheduled frames):

| Descriptor | faithful/valid | faithful/test | nll/valid | nll/test | beta_nll/valid | beta_nll/test |
|---|---|---|---|---|---|---|
| Strain norm | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 |
| Absolute vorticity | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 |
| Strain, adjusted ranks | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 |
| Vorticity, adjusted ranks | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 |
| Absolute divergence | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 |
| Neighbor velocity RMS difference | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 |
| Base error (same fit mask) | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 |
| Dense benefit (same fit mask) | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 |
| Neighbor count (all particles) | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 |
| Speed (all particles) | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 |
| Signed wall clearance (all particles) | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 |
| Observed acceleration (all particles) | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 |
| Base error (all particles) | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 |
| Dense benefit (all particles) | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 | 36,36,36,36,36 |

## Interpretation limits

Correlation with deformation is a spatial association. Useful edge allocation must be assessed by the actual matched-budget intervention and rollout accuracy, failures and measured cost. Positive whole-dense benefit is neither necessary nor sufficient for useful selected-edge allocation. Dense benefit is base error minus dense-graph error; it is not single-edge marginal value and shares the base error algebraically. Risk–error and risk–benefit comparisons on the identical fit mask are retained above, alongside all-particle comparisons. No proxy, objective, seed or split was selected by its outcome.

These compact networks use a short training budget and a small convenience sample. The original pilot predictions and both splits had already been inspected. The running six-model 100,000-update experiment and its five locked policies are unchanged. Applying this postprocess to full-model same-state artifacts would be a separate exploratory diagnostic after the existing completion/test gate.

## Reproduction and provenance

CPU postprocessing took 2.943 seconds; this is not model or policy latency.
The input identity manifest pins the summary from commit d061cef; the analyzer verifies its data/raw-array hashes and frame/particle joins before using them. No checkpoint or reserved full-test source was opened.

- Result SHA256: `9a3856d0a7906b3023e473e62bc7269d5139651e255b425994d264cb7ac68b8b`
- Analysis protocol SHA256: `a60a2e625aefd937a88424792882db0913565b09dbbef2afe015cad8071cf653`
- Geometry NPZ SHA256: `3ec040367eb6ca27941244f4b93675daa1e580d3d6fd23d3d4bdf283b6f9f67f` (retained locally under work/)
- Figure: `physical_complexity_pilot.png`

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m research.analyze_physical_complexity \
  --data-dir data-pilot --output-prefix work/physical_complexity/analysis
python -m research.report_physical_complexity \
  --input work/physical_complexity/analysis.json --output-prefix research/results/physical_complexity_pilot
```
