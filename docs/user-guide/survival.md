# Survival Analysis

ngboost-lightning supports time-to-event modeling with right-censored data
via `LightningBoostSurvival`. This estimator uses a censored log-likelihood
scoring rule that properly handles observations where the event was not
observed (censored).

## Setup

Survival data has two target arrays:

- **T** — observed time (time-to-event or time-to-censoring)
- **E** — event indicator (1 = event observed, 0 = censored)

```python
from ngboost_lightning import LightningBoostSurvival, Weibull

surv = LightningBoostSurvival(
    dist=Weibull,
    n_estimators=200,
    learning_rate=0.05,
)
surv.fit(X_train, T_train, E_train)
```

## Predictions

```python
dist = surv.pred_dist(X_test)

# Predicted survival time
dist.mean()

# Survival function: P(T > t)
import numpy as np
t = np.array([1.0, 2.0, 5.0])
survival_probs = np.exp(dist.logsf(t))

# Hazard-related quantities
dist.logpdf(t)  # log-density
dist.cdf(t)     # CDF = 1 - survival function
```

## Compatible Distributions

Any distribution that implements `logsf` (log survival function) can be used
with the survival estimator:

| Distribution | Parameters | Notes |
|---|---|---|
| `Weibull` | log_scale, log_concentration | Primary choice for survival |
| `LogNormal` | mu, log_sigma | Log-normal survival times |
| `Exponential` | log_rate | Constant hazard rate |

`Weibull` is the most common choice because it can model both increasing and
decreasing hazard rates depending on the learned concentration parameter.

## How Censored Training Works

The `CensoredLogScore` scoring rule modifies the standard log-likelihood:

- **Uncensored observations** (E=1): uses `logpdf(t)` — the event was observed,
  so we maximize the density at the event time.
- **Censored observations** (E=0): uses `logsf(t)` — we only know the event
  didn't happen before time `t`, so we maximize the survival probability.

Gradients flow through both paths, so the model learns from censored
observations without imputing event times.

## Early Stopping

Early stopping works the same as for regression — pass validation data
with event indicators:

```python
surv.fit(
    X_train, T_train, E_train,
    X_val=X_val, T_val=T_val, E_val=E_val,
    early_stopping_rounds=10,
)
```

## Evaluation

The `evaluation` module provides survival-specific metrics:

```python
from ngboost_lightning.evaluation import concordance_index, calibration_survival

# Concordance index (discrimination)
c_index = concordance_index(dist, T_test, E_test)

# Survival calibration curve
obs, exp = calibration_survival(dist, T_test, E_test)
```

See [Evaluation](evaluation.md) for more details.

## Scoring

The `score()` method returns the negative mean censored log-likelihood
(higher is better, following the sklearn convention):

```python
score = surv.score(X_test, T_test, E_test)
```
