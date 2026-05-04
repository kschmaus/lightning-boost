# Distributions

ngboost-lightning supports 12 probability distributions spanning regression,
classification, count data, and survival analysis. Every distribution uses
unconstrained internal parameterization (log-link or logit) so boosted tree
predictions can take any real value.

## Distribution Table

| Distribution | Class | Parameters | CRPS | Use Case |
|---|---|---|---|---|
| Normal | `Normal` | mean, log_scale | Yes | General regression (default) |
| Log-Normal | `LogNormal` | mu, log_sigma | Yes | Positive, right-skewed targets |
| Exponential | `Exponential` | log_rate | Yes | Positive targets, waiting times |
| Gamma | `Gamma` | log_alpha, log_beta | Yes | Positive targets (non-diagonal Fisher) |
| Poisson | `Poisson` | log_rate | Yes | Count data |
| Laplace | `Laplace` | loc, log_scale | Yes | Heavy-tailed, robust regression |
| Student's T | `StudentT` | loc, log_scale, log_df | No | Heavy tails with learnable degrees of freedom |
| Weibull | `Weibull` | log_scale, log_concentration | No | Survival / time-to-event data |
| Half-Normal | `HalfNormal` | log_scale | No | Positive targets, folded normal |
| Cauchy | `Cauchy` | loc, log_scale | No | Extreme outliers |
| Bernoulli | `Bernoulli` | logit | No | Binary classification (default) |
| Categorical | `k_categorical(K)` | K-1 logits | No | Multiclass classification |

The **CRPS** column indicates whether the distribution supports training with
[`CRPScore`](scoring-rules.md). Distributions without CRPS support can only
use `LogScore` (negative log-likelihood).

## Choosing a Distribution

**Continuous regression:**

- Start with `Normal` — it works well for most problems.
- Use `LogNormal` if targets are strictly positive and right-skewed (e.g. prices, durations).
- Use `Laplace` if you want robustness to outliers (heavier tails than Normal, L1-like loss).
- Use `StudentT` for heavy tails with an automatically learned tail weight (3 parameters).
- Use `Cauchy` only for extreme outlier scenarios — the mean is undefined.

**Positive targets:**

- `Exponential` for simple rate modeling (1 parameter).
- `Gamma` for more flexible positive-valued modeling (2 parameters, non-diagonal Fisher).
- `HalfNormal` for positive targets that are concentrated near zero.

**Count data:**

- `Poisson` for non-negative integer targets.

**Classification:**

- `Bernoulli` for binary (used automatically by `LightningBoostClassifier`).
- `k_categorical(K)` for multiclass — you must specify K explicitly.

**Survival:**

- `Weibull` is the primary choice for time-to-event data with
  [`LightningBoostSurvival`](survival.md).
- `LogNormal` and `Exponential` also support `logsf` for survival modeling.

## Using a Distribution

Pass the distribution class (not an instance) to the estimator:

```python
from ngboost_lightning import LightningBoostRegressor, Laplace

reg = LightningBoostRegressor(dist=Laplace, n_estimators=200)
reg.fit(X_train, y_train)

dist = reg.pred_dist(X_test)
dist.mean()   # location parameter
dist.scale    # scale parameter
```

## Fixed Degrees of Freedom (Student's T)

`StudentT` learns the degrees of freedom from data. If you want to fix it:

```python
from ngboost_lightning import LightningBoostRegressor, t_fixed_df

reg = LightningBoostRegressor(dist=t_fixed_df(5), n_estimators=200)
```

`StudentT3` is a convenience alias for `t_fixed_df(3)`.

## Distribution Interface

All distributions provide these methods after fitting:

| Method | Description |
|---|---|
| `mean()` | Conditional mean |
| `sample(n)` | Random samples |
| `cdf(y)` | Cumulative distribution function |
| `ppf(q)` | Quantile function (inverse CDF) |
| `logpdf(y)` | Log probability density / mass |
| `score(y)` | Per-sample negative log-likelihood |
| `logsf(y)` | Log survival function (if supported) |

See the [API Reference](../api/distributions/base.md) for the complete
`Distribution` abstract base class.
