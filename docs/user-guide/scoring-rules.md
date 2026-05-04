# Scoring Rules

A scoring rule defines the loss function used during the boosting loop.
It controls the gradients, the natural gradient transformation, and the
recorded training/validation loss. ngboost-lightning provides two scoring
rules plus a censored variant for survival data.

## Available Scoring Rules

### LogScore (default)

Negative log-likelihood (NLL). Equivalent to maximum likelihood estimation.
This is the default for all estimators.

```python
from ngboost_lightning import LightningBoostRegressor

# These are equivalent:
reg = LightningBoostRegressor(n_estimators=200)
reg = LightningBoostRegressor(scoring_rule=None, n_estimators=200)
```

LogScore works with all 12 distributions.

### CRPScore

The Continuous Ranked Probability Score (CRPS) is a proper scoring rule that
measures the quality of the entire predictive CDF — not just the density at
the observed value. CRPS tends to produce better-calibrated predictive
distributions, particularly in the tails.

```python
from ngboost_lightning import LightningBoostRegressor, CRPScore

reg = LightningBoostRegressor(scoring_rule=CRPScore(), n_estimators=200)
reg.fit(X_train, y_train)
```

!!! note
    CRPScore is only available for distributions that implement
    `crps_score`, `crps_d_score`, and `crps_metric`. Currently supported:
    Normal, LogNormal, Exponential, Gamma, Poisson, and Laplace.

### CensoredLogScore

Used automatically by [`LightningBoostSurvival`](survival.md). Computes log-likelihood
for uncensored observations and log survival function for censored observations:

$$\ell_i = \begin{cases} \log f(t_i) & \text{if event observed} \\ \log S(t_i) & \text{if censored} \end{cases}$$

You do not need to create this manually — the survival estimator handles it.

## LogScore vs CRPScore

| Property | LogScore | CRPScore |
|---|---|---|
| Optimizes | Likelihood at observed value | Full predictive CDF quality |
| Calibration | Good density calibration | Better quantile/interval calibration |
| Sensitivity | More sensitive to outliers | More robust to outliers |
| Distributions | All 12 | 7 (continuous with closed-form CRPS) |
| Speed | Faster (simpler gradients) | Slightly slower |

**When to use CRPScore:**

- You care about prediction intervals and quantile accuracy.
- Your evaluation metric is calibration-based (e.g. coverage of prediction intervals).
- Your data has outliers that might dominate NLL.

**When to use LogScore:**

- You care about density estimation (e.g. probability of specific outcomes).
- You need the widest distribution support.
- Default choice when unsure.

## How Scoring Rules Work

Each iteration of the boosting loop:

1. The scoring rule computes per-sample scores via `score(dist, y)`.
2. Gradients are computed via `d_score(dist, y)`.
3. The Fisher Information matrix is obtained via `metric(dist)`.
4. The natural gradient is `metric_inverse @ gradient`.
5. One LightGBM tree per parameter is fit to its natural gradient component.

The scoring rule is a strategy object that plugs into this loop. Both
`LogScore` and `CRPScore` implement the same
[`ScoringRule`](../api/scoring.md) protocol.
