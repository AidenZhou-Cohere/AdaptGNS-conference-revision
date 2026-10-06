# Existing self-loop bridge correlations

These are already-audited results, newly extracted for reporting. All 2,550 histories remain represented. Values are dimensionless within-frame particle Spearman correlations, averaged equally within trajectory and then equally across trajectories; the final mean and sample SD use three training seeds. Own benefit is signed base-minus-selected-action normalized vector squared error. No new inference or raw-array calculation occurred.

| Objective | Split | Risk rule | Label | Seed 0 | Seed 1 | Seed 2 | Mean +/- sample SD | Undefined frames |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| faithful | valid | Previous | base_residual | +0.319893 | +0.365709 | +0.337899 | +0.341167 +/- 0.023083 | 0 |
| faithful | valid | Previous | own_sparse_benefit | -0.044496 | -0.021954 | -0.075320 | -0.047257 +/- 0.026790 | 0 |
| faithful | valid | Current | base_residual | +0.322746 | +0.368791 | +0.339293 | +0.343610 +/- 0.023324 | 0 |
| faithful | valid | Current | own_sparse_benefit | -0.041114 | -0.022549 | -0.079669 | -0.047777 +/- 0.029138 | 0 |
| faithful | test | Previous | base_residual | +0.268871 | +0.377073 | +0.329045 | +0.324996 +/- 0.054215 | 0 |
| faithful | test | Previous | own_sparse_benefit | -0.024837 | +0.014110 | -0.078924 | -0.029883 +/- 0.046722 | 0 |
| faithful | test | Current | base_residual | +0.267573 | +0.376517 | +0.331179 | +0.325090 +/- 0.054727 | 0 |
| faithful | test | Current | own_sparse_benefit | -0.025161 | +0.015936 | -0.079885 | -0.029703 +/- 0.048071 | 0 |
| nll | valid | Previous | base_residual | +0.396997 | +0.420821 | +0.401291 | +0.406370 +/- 0.012698 | 0 |
| nll | valid | Previous | own_sparse_benefit | -0.046409 | -0.083775 | -0.106258 | -0.078814 +/- 0.030232 | 0 |
| nll | valid | Current | base_residual | +0.401431 | +0.425096 | +0.404462 | +0.410330 +/- 0.012877 | 0 |
| nll | valid | Current | own_sparse_benefit | -0.045783 | -0.084526 | -0.106404 | -0.078904 +/- 0.030699 | 0 |
| nll | test | Previous | base_residual | +0.357229 | +0.404962 | +0.395422 | +0.385871 +/- 0.025259 | 0 |
| nll | test | Previous | own_sparse_benefit | -0.062974 | -0.072572 | -0.090868 | -0.075471 +/- 0.014171 | 0 |
| nll | test | Current | base_residual | +0.356464 | +0.407661 | +0.395628 | +0.386584 +/- 0.026770 | 0 |
| nll | test | Current | own_sparse_benefit | -0.062907 | -0.071626 | -0.090203 | -0.074912 +/- 0.013941 | 0 |

Verification: 14,576 source/summary/coverage/scalar correspondence checks passed. The counts measure bookkeeping consistency, not scientific replication. The original strict array audit and separate scalar audit remain the computational evidence.

The bridge globally orders each selected graph. These correlations do not measure the autonomous cached-own-graph policy, and the inspected test histories do not supply independent confirmation.
