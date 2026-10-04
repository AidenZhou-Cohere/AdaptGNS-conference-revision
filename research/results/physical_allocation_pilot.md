# Cheap physical allocation controls on the compact pilot

**Completed separate exploratory comparison: all 15 fixed models, both previously inspected splits, 36 frames per split/model, six replayed controls and two new policies.** No model was trained or selected. The full-architecture training and its locked five-policy evaluation are unchanged.

The new `inverse_count25` score is negative mandatory-base degree. `velocity_rms25` scores each particle by the RMS last-observed displacement difference over its base neighbors; isolated particles receive score zero by the declared convention and remain counted. The original pilot graph and pair ordering are retained, including its float32 base-radius comparison. This differs from the earlier physical correlation analysis’s strict float64 neighborhood. Every particle participates.

Both methods retain all base pairs and exactly floor(0.25 × available annulus pairs). Optional-pair priority is the maximum endpoint score, with the original deterministic ID tie rule. No target is used to construct a score. All six original control errors and edge counts were replay-checked before accepting comparisons. Previous-risk scores use the preceding observed base graph, not an autonomous controller’s cached own-graph score.

## One-step normalized-acceleration coordinate MSE

Lower is better. Particles are averaged within each frame, then equally over twelve frames per trajectory, three trajectories per split, and five training seeds. ± denotes sample seed SD; it is not a confidence interval over the trajectory population. Missing required frames or seeds are never replaced with survivor means.

### valid

| Policy | Faithful | NLL | Beta-NLL |
|---|---:|---:|---:|
| base | 0.2141 ± 0.0151 | 0.2727 ± 0.0080 | 0.2319 ± 0.0134 |
| dense | 0.2146 ± 0.0166 | 0.2708 ± 0.0084 | 0.2326 ± 0.0155 |
| random25 | 0.2139 ± 0.0159 | 0.2716 ± 0.0080 | 0.2311 ± 0.0142 |
| speed25 | 0.2148 ± 0.0164 | 0.2722 ± 0.0083 | 0.2335 ± 0.0155 |
| current_risk25 | 0.2149 ± 0.0165 | 0.2718 ± 0.0082 | 0.2334 ± 0.0156 |
| lagged_base_risk25 | 0.2149 ± 0.0164 | 0.2718 ± 0.0082 | 0.2334 ± 0.0155 |
| inverse_count25 | 0.2147 ± 0.0161 | 0.2710 ± 0.0081 | 0.2306 ± 0.0147 |
| velocity_rms25 | 0.2143 ± 0.0162 | 0.2708 ± 0.0080 | 0.2332 ± 0.0156 |

### test

| Policy | Faithful | NLL | Beta-NLL |
|---|---:|---:|---:|
| base | 4.0611 ± 0.2229 | 3.6502 ± 0.0164 | 3.7239 ± 0.1280 |
| dense | 4.0164 ± 0.1875 | 3.6210 ± 0.0215 | 3.6996 ± 0.1325 |
| random25 | 4.0428 ± 0.2117 | 3.6414 ± 0.0161 | 3.7141 ± 0.1315 |
| speed25 | 4.0708 ± 0.2252 | 3.6402 ± 0.0182 | 3.7287 ± 0.1317 |
| current_risk25 | 4.0132 ± 0.1917 | 3.6414 ± 0.0166 | 3.7067 ± 0.1363 |
| lagged_base_risk25 | 4.0297 ± 0.1963 | 3.6542 ± 0.0181 | 3.7327 ± 0.1285 |
| inverse_count25 | 4.0604 ± 0.2193 | 3.6413 ± 0.0178 | 3.7206 ± 0.1299 |
| velocity_rms25 | 4.0425 ± 0.2163 | 3.6301 ± 0.0171 | 3.6980 ± 0.1311 |

## Paired percent differences

Each entry is the five-seed mean ± sample SD of 100 × (physical-policy MSE − control MSE) / control MSE, calculated separately within each seed. Negative favors the physical policy. This is not the percent difference between group means. All six controls, all objectives and both splits are retained; no p-values or best-case selection.

### faithful / valid

| Physical policy | base | dense | random25 | speed25 | current_risk25 | lagged_base_risk25 |
|---|---:|---:|---:|---:|---:|---:|
| inverse_count25 | 0.249 ± 1.057% | 0.065 ± 0.339% | 0.336 ± 0.348% | -0.058 ± 0.518% | -0.086 ± 0.514% | -0.094 ± 0.504% |
| velocity_rms25 | 0.070 ± 1.199% | -0.115 ± 0.186% | 0.156 ± 0.541% | -0.238 ± 0.259% | -0.267 ± 0.223% | -0.274 ± 0.221% |

### faithful / test

| Physical policy | base | dense | random25 | speed25 | current_risk25 | lagged_base_risk25 |
|---|---:|---:|---:|---:|---:|---:|
| inverse_count25 | -0.013 ± 0.247% | 1.069 ± 1.156% | 0.429 ± 0.463% | -0.250 ± 0.209% | 1.154 ± 1.167% | 0.740 ± 0.839% |
| velocity_rms25 | -0.450 ± 0.270% | 0.626 ± 0.978% | -0.011 ± 0.238% | -0.685 ± 0.505% | 0.710 ± 0.987% | 0.299 ± 0.713% |

### nll / valid

| Physical policy | base | dense | random25 | speed25 | current_risk25 | lagged_base_risk25 |
|---|---:|---:|---:|---:|---:|---:|
| inverse_count25 | -0.630 ± 0.351% | 0.097 ± 0.334% | -0.226 ± 0.143% | -0.418 ± 0.085% | -0.289 ± 0.061% | -0.285 ± 0.062% |
| velocity_rms25 | -0.701 ± 0.263% | 0.026 ± 0.399% | -0.298 ± 0.075% | -0.489 ± 0.266% | -0.359 ± 0.244% | -0.355 ± 0.244% |

### nll / test

| Physical policy | base | dense | random25 | speed25 | current_risk25 | lagged_base_risk25 |
|---|---:|---:|---:|---:|---:|---:|
| inverse_count25 | -0.243 ± 0.090% | 0.562 ± 0.142% | -0.002 ± 0.060% | 0.031 ± 0.096% | -0.001 ± 0.180% | -0.353 ± 0.041% |
| velocity_rms25 | -0.550 ± 0.170% | 0.252 ± 0.213% | -0.309 ± 0.181% | -0.276 ± 0.302% | -0.309 ± 0.177% | -0.659 ± 0.219% |

### beta_nll / valid

| Physical policy | base | dense | random25 | speed25 | current_risk25 | lagged_base_risk25 |
|---|---:|---:|---:|---:|---:|---:|
| inverse_count25 | -0.561 ± 0.954% | -0.820 ± 0.406% | -0.192 ± 0.420% | -1.192 ± 0.323% | -1.155 ± 0.309% | -1.157 ± 0.291% |
| velocity_rms25 | 0.511 ± 1.259% | 0.247 ± 0.299% | 0.883 ± 0.731% | -0.129 ± 0.244% | -0.092 ± 0.185% | -0.094 ± 0.192% |

### beta_nll / test

| Physical policy | base | dense | random25 | speed25 | current_risk25 | lagged_base_risk25 |
|---|---:|---:|---:|---:|---:|---:|
| inverse_count25 | -0.091 ± 0.168% | 0.572 ± 0.588% | 0.176 ± 0.170% | -0.216 ± 0.134% | 0.380 ± 0.557% | -0.325 ± 0.295% |
| velocity_rms25 | -0.696 ± 0.812% | -0.040 ± 0.155% | -0.432 ± 0.506% | -0.820 ± 0.755% | -0.231 ± 0.201% | -0.928 ± 0.929% |

## Geometry, ties and replay

The following counts describe 36 unique observed frames in each split, shared by all models. Split cutoff ties mean the budget includes some but not all optional pairs with the cutoff score. Such selection depends on particle IDs under the fixed tie rule.

| Split | Isolated particle–frame pairs | inverse_count25 split-tie frames | velocity_rms25 split-tie frames |
|---|---:|---:|---:|
| valid | 37 | 35 / 36 | 27 / 36 |
| test | 53 | 36 / 36 | 21 / 36 |

All 6,480 frame-policy replay comparisons passed the fixed tolerance (rtol 2e−5, atol 1e−7) and exact edge counts. Maximum absolute MSE difference from the original controls was 1.11972012e-05. Pair overlaps, full cutoff-tie records, isolated counts, per-seed absolute differences and raw-array hashes remain in the scalar outputs.

## Scope and reproducibility

These are one-step interventions on observed histories from a small convenience sample. They were designed after inspecting earlier pilot and physical-correlation results. They do not establish long-horizon stability, independent confirmation, a full-architecture advantage or deployment speedup. The labels “physical” and “sparse” refer to the defined score heuristics, not a causal interpretation or detected free surface.

The complete single-CPU-thread attempt took 37.737 seconds. Timings include shared/reused work and concurrent full-model training; they are operational measurements, not a fair optimized policy-speed comparison. No new frame was allowed to start after the declared 300-second limit. A fresh output directory and preserved failure records prevent silent overwrites or automatic retries.

- Result SHA256: `9921346dd542276dfbf3557c8f4b246e0c06da8e862e33c95becd8e9572e2594`
- Protocol SHA256: `0d2a35e6422f5e732984f18abf44da17b6f23a87dba1bd1b7e746274be0f02ce`
- Input-identity SHA256: `a195fb894d0de35a8d312421cf06817a9bd97f2cf9167c6aad262647b3b105c2`

```sh
python -m research.physical_allocation_pilot \
  --data-dir data-pilot --output-dir work/physical-allocation-attempt
python -m research.report_physical_allocation \
  --input work/physical-allocation-attempt/summary.json \
  --output-prefix research/results/physical_allocation_pilot
```
