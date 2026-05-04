# Evaluation

ngboost-lightning provides diagnostic tools for assessing the quality of
probabilistic predictions. These are available in the `ngboost_lightning.evaluation`
module.

## PIT (Probability Integral Transform)

For a well-calibrated model, the CDF values `F(y)` evaluated at the true
observations should follow a Uniform(0, 1) distribution. Deviations from
uniformity indicate miscalibration.

```python
from ngboost_lightning.evaluation import pit_values, plot_pit_histogram

dist = reg.pred_dist(X_test)
pit = pit_values(dist, y_test)

# Visual check — histogram should be flat
plot_pit_histogram(pit)
```

**Interpreting the PIT histogram:**

- **Flat (uniform)** — well calibrated.
- **U-shaped** — underdispersed (prediction intervals too narrow).
- **Hump-shaped** — overdispersed (prediction intervals too wide).
- **Skewed** — systematic bias in location.

## Calibration Curve

The calibration curve plots observed coverage against expected quantile levels.
A perfectly calibrated model lies on the diagonal.

```python
from ngboost_lightning.evaluation import (
    calibration_regression,
    calibration_error,
    plot_calibration_curve,
)

obs, exp = calibration_regression(dist, y_test, bins=11)
plot_calibration_curve(obs, exp)

# Scalar summary: mean absolute deviation from diagonal
cal_err = calibration_error(dist, y_test)
```

## Survival Evaluation

### Concordance Index

The concordance index (C-index) measures discrimination — how well the model
ranks patients by predicted risk. A value of 1.0 means perfect ranking;
0.5 means random.

```python
from ngboost_lightning.evaluation import concordance_index

dist = surv.pred_dist(X_test)
c_index = concordance_index(dist, T_test, E_test)
```

### Survival Calibration

Analogous to regression calibration, but uses the survival function:

```python
from ngboost_lightning.evaluation import calibration_survival

obs, exp = calibration_survival(dist, T_test, E_test)
```

## Plotting

Plot functions require **matplotlib** (optional dependency). They return
the matplotlib `Axes` object for further customization:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_pit_histogram(pit, ax=axes[0])
plot_calibration_curve(obs, exp, ax=axes[1])
plt.tight_layout()
plt.show()
```

If no `ax` is provided, the functions create a new figure automatically.

## Function Reference

| Function | Description |
|---|---|
| `pit_values(dist, y)` | PIT values F(y), shape `[n_samples]` |
| `calibration_regression(dist, y)` | (expected, observed) quantile calibration |
| `calibration_error(dist, y)` | Scalar mean absolute calibration error |
| `calibration_survival(dist, T, E)` | Survival calibration curve |
| `concordance_index(dist, T, E)` | C-index for survival discrimination |
| `plot_pit_histogram(pit)` | Histogram of PIT values |
| `plot_calibration_curve(obs, exp)` | Calibration curve plot |

See the [API Reference](../api/evaluation.md) for full signatures and details.
