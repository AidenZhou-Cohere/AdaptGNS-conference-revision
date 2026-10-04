# The implemented noise changes the conditional risk target

This explanatory derivation is not a novelty claim or a measurement of an effect on WaterDrop. It uses the frozen GNS target correction and a separate synthetic Gaussian example. No model, GPU or reserved test data is used.

## Exact target and risk identity

The input history is \(\widetilde x_k=x_k+\xi_k\). The code adjusts the next-position target to \(x_{t+1}+\xi_t\), so its acceleration target is
\[
Y_t=(x_{t+1}+\xi_t)-2(x_t+\xi_t)+(x_{t-1}+\xi_{t-1})
=A_t-\eta_t,
\qquad \eta_t=\xi_t-\xi_{t-1}.
\]
Here \(A_t=x_{t+1}-2x_t+x_{t-1}\) is clean acceleration in stored-frame displacement units. Five independent increments of variance \(s^2/5\) are integrated into velocity noise, so the final \(\eta_5\) has variance \(s^2\) per coordinate.

For any fixed predictor on noisy history, let \(R=A_t-\mu(\widetilde H)\). With finite second moments,
\[
\mathbb E[\|Y_t-\mu\|^2\mid\widetilde H]
=\mathbb E[\|R\|^2\mid\widetilde H]
+\mathbb E[\|\eta_t\|^2\mid\widetilde H]
-2\mathbb E[R^\top\eta_t\mid\widetilde H].
\]
Fixed affine normalization applies the same scaling to the residual and noise terms. With a fixed mean, an unconstrained isotropic Gaussian head optimally predicts this residual second moment divided by dimension; a variance floor or restricted head representation modifies that ideal optimum.

Noise independence from the clean trajectory before conditioning does not justify deleting the cross term. The noisy history, and hence the predictor, contains information about the noise. Clean evaluation additionally changes the input. Writing \(\Delta_\mu=\mu(\widetilde H)-\mu(H)\) and \(R_{\rm clean}=A_t-\mu(H)\), the exact relation is
\[
Y_t-\mu(\widetilde H)=R_{\rm clean}-\Delta_\mu-\eta_t.
\]
A training-calibrated head therefore cannot generally be converted to clean residual risk by subtracting marginal noise variance.

## Gaussian six-frame example

Take a fixed initial position, zero clean acceleration and \(V\sim\mathcal N(0,\tau^2)\). Observe five noisy velocities
\[
U_k=V+\sum_{j=1}^kz_j,\qquad
z_j\overset{\rm iid}{\sim}\mathcal N(0,\delta),\quad \delta=s^2/5.
\]
Noise is independent of \(V\). The target is \(Y=-\eta_5=V-U_5\). Since \(U_1=V+z_1\) and \(U_k-U_{k-1}=z_k\) for \(k\ge2\), only \(U_1\) provides information about \(V\). For positive \(\tau,s\), put
\[
a=\frac{\tau^2}{\tau^2+\delta},\qquad
b=1-a,\qquad
h=\frac{\tau^2\delta}{\tau^2+\delta}.
\]
Gaussian conditioning yields
\[
V\mid U\sim\mathcal N(aU_1,h),\qquad
\mu(U)=aU_1-U_5,\qquad q(U)=h.
\]
This mean and variance exactly describe the augmented target. The continuous Gaussian Bayes mean on a noiseless history \(U_k=v\) is \(-bv\). The clean residual second moment is \(b^2v^2\), and over the clean velocity population it is
\[
\mathbb E_V R_{\rm clean}^2=b^2\tau^2=bh.
\]
The unchanged head \(h\) exceeds this aggregate mean by a factor \(1/b=1+5\tau^2/s^2\), although individual large-velocity residuals can exceed the head. True clean target variance is zero; fixed-mean residual MSE is a different quantity.

The cross term is explicit: \(c(U)=U_5-aU_1=\mathbb E[\eta_5\mid U]\), and the residual against the clean target on noisy inputs is \(R=c(U)\). Thus
\[
\mathbb E[(Y-\mu)^2\mid U]=c^2+(h+c^2)-2c^2=h.
\]
Here clean acceleration is constant, so conditional covariance of acceleration and noise is zero, while the **uncentered residual–noise product** remains nonzero. Subtracting \(s^2\) from \(h<s^2\) would give a negative result.

Exact clean histories are a measure-zero diagonal under noisy training. The clean-input formula uses the unique continuous Gaussian Bayes version, not every arbitrary measurable training minimizer. The displayed ratios require positive variances; degenerate zero-variance cases and a positive head floor need separate interpretation. This example neither predicts a WaterDrop calibration error nor establishes that augmentation harms simulation.

## Reproduction

From the repository root:

```sh
python -m research.noise_target_toy --output research/results/noise_target_toy_check.json
```

The [CPU-only script](../noise_target_toy.py) checks the 5×5 Gaussian covariance calculation and 250,000 synthetic draws at seed 20261004, \(\tau^2=4\), \(s^2=1\). The analytic training variance is 0.19047619 and clean MSE is 0.00907029. The verified sample values are 0.19035431 and 0.00908417. All five checked moments lie within 1.41 estimated Monte Carlo standard errors of their analytic values; matrix coefficients agree within \(4.72\times10^{-15}\). The [result JSON](../results/noise_target_toy_check.json) records source SHA256, Python and NumPy versions. These checks support the algebra only.
