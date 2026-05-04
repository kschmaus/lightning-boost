# <img src="ngboost-lightning.svg" alt="ngboost-lightning logo" height="160" style="vertical-align: middle;"/> ngboost-lightning

[![CI](https://github.com/kschmaus/ngboost-lightning/actions/workflows/ci.yml/badge.svg)](https://github.com/kschmaus/ngboost-lightning/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ngboost-lightning)](https://pypi.org/project/ngboost-lightning/)
[![Python](https://img.shields.io/pypi/pyversions/ngboost-lightning)](https://pypi.org/project/ngboost-lightning/)
[![License](https://img.shields.io/github/license/kschmaus/ngboost-lightning)](https://github.com/kschmaus/ngboost-lightning/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://kschmaus.github.io/ngboost-lightning/)

Natural gradient boosting for probabilistic prediction, powered by LightGBM.

ngboost-lightning reimplements the [NGBoost](https://github.com/stanfordmlgroup/ngboost) algorithm using [LightGBM](https://github.com/lightgbm-org/LightGBM)'s histogram-based tree building instead of scikit-learn's exact splitter. The algorithm is structurally identical -- K independent boosters (one per distribution parameter), natural gradients, line search -- but training is **up to 13x faster** on larger datasets.

## 💿 Installation

Requires Python >= 3.11.

```bash
pip install ngboost-lightning
```

## 🚀 Quick Start

### Probabilistic Regression

```python
from ngboost_lightning import LightningBoostRegressor

reg = LightningBoostRegressor(n_estimators=200, learning_rate=0.05)
reg.fit(X_train, y_train)

# Point predictions
preds = reg.predict(X_test)

# Full predictive distribution
dist = reg.pred_dist(X_test)
dist.mean()          # conditional mean
dist.scale           # predicted uncertainty
dist.ppf(0.05)       # 5th percentile
dist.ppf(0.95)       # 95th percentile
dist.cdf(y_test)     # CDF evaluation
```

### Binary Classification

```python
from ngboost_lightning import LightningBoostClassifier

clf = LightningBoostClassifier(n_estimators=200, learning_rate=0.05)
clf.fit(X_train, y_train)

clf.predict(X_test)        # class labels
clf.predict_proba(X_test)  # probabilities [n_samples, 2]
```

### Multiclass Classification

```python
from ngboost_lightning import LightningBoostClassifier, k_categorical

clf = LightningBoostClassifier(
    dist=k_categorical(3),  # explicit K required
    n_estimators=200,
    learning_rate=0.05,
)
clf.fit(X_train, y_train)

clf.predict_proba(X_test)  # probabilities [n_samples, 3]
```

## 🎲 Distributions

| Distribution | Parameters | Use Case |
|---|---|---|
| `Normal` (default) | mean, log_scale | General regression |
| `LogNormal` | mu, log_sigma | Positive, right-skewed targets |
| `Exponential` | log_rate | Positive targets, waiting times |
| `Gamma` | log_alpha, log_beta | Positive targets (non-diagonal Fisher) |
| `Poisson` | log_rate | Count data |
| `Laplace` | loc, log_scale | Heavy-tailed, robust regression |
| `StudentT` | loc, log_scale, log_df | Heavy tails with learnable degrees of freedom |
| `Weibull` | log_scale, log_concentration | Survival / time-to-event data |
| `HalfNormal` | log_scale | Positive targets, folded normal |
| `Cauchy` | loc, log_scale | Extreme outliers |
| `Bernoulli` | logit | Binary classification (default) |
| `k_categorical(K)` | K-1 logits | Multiclass classification |

All distributions use log-link or logit parameterization internally to keep parameters unconstrained during boosting.

```python
from ngboost_lightning import LightningBoostRegressor, LogNormal

reg = LightningBoostRegressor(dist=LogNormal, n_estimators=200)
reg.fit(X_train, y_train)
```

## 🏥 Survival Analysis

ngboost-lightning supports right-censored data for time-to-event modeling via `LightningBoostSurvival`:

```python
from ngboost_lightning import LightningBoostSurvival, Weibull

surv = LightningBoostSurvival(
    dist=Weibull,
    n_estimators=200,
    learning_rate=0.05,
)
surv.fit(X_train, T_train, E_train)  # T = time, E = event indicator (1=observed)

dist = surv.pred_dist(X_test)
dist.mean()             # predicted survival time
dist.logsf(t)           # log survival function at time t
```

The survival estimator uses a censored log-likelihood scoring rule that properly handles right-censored observations.

## 📊 Scoring Rules

| Rule | Description |
|---|---|
| `LogScore` (default) | Negative log-likelihood |
| `CRPScore` | Continuous Ranked Probability Score (proper scoring rule) |
| `CensoredLogScore` | Censored negative log-likelihood (used automatically by survival estimator) |

```python
from ngboost_lightning import LightningBoostRegressor, CRPScore

reg = LightningBoostRegressor(scoring_rule=CRPScore, n_estimators=200)
reg.fit(X_train, y_train)
```

## 💡 How It Works

NGBoost fits one independent decision tree per distribution parameter per boosting iteration. Each tree gets its own gradient component as the regression target.

ngboost-lightning replicates this exactly with K independent LightGBM Booster instances. Speed gains come from LightGBM's histogram-based splitting, not from shared tree structure.

Each iteration:
1. Compute negative log-likelihood of the current distributional prediction
2. Differentiate w.r.t. all internal parameters
3. Transform to natural gradient via Fisher Information matrix inverse
4. Fit one LightGBM tree per parameter to its gradient component
5. Line search for optimal step size
6. Update parameters

## 🧩 sklearn Compatibility

All three estimators are fully compatible with scikit-learn:

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LightningBoostRegressor()),
])
scores = cross_val_score(pipe, X, y, cv=5)
```

Early stopping with validation data:

```python
reg.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    early_stopping_rounds=10,
)
```

Feature importances per distribution parameter:

```python
reg.feature_importances_  # shape [n_params, n_features], rows sum to 1
```

Sample weights:

```python
reg.fit(X_train, y_train, sample_weight=weights)
```

Staged predictions (one distribution per boosting iteration):

```python
for dist in reg.staged_pred_dist(X_test):
    print(dist.mean())
```

Automatic validation split:

```python
reg = LightningBoostRegressor(validation_fraction=0.1, early_stopping_rounds=10)
reg.fit(X_train, y_train)  # holds out 10% automatically
```

Column subsampling (per-iteration random feature subset):

```python
reg = LightningBoostRegressor(col_sample=0.8, n_estimators=200)
reg.fit(X_train, y_train)
```

Custom loss monitors:

```python
def mse_monitor(dist, y):
    return float(((dist.mean() - y) ** 2).mean())

reg.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    train_loss_monitor=mse_monitor,  # custom training loss recording
    val_loss_monitor=mse_monitor,    # custom validation loss + early stopping
)
```

## 📏 Evaluation Utilities

```python
from ngboost_lightning.evaluation import (
    calibration_error,
    calibration_regression,
    pit_values,
    plot_calibration_curve,
    plot_pit_histogram,
)

dist = reg.pred_dist(X_test)

# PIT (Probability Integral Transform) histogram
pit = pit_values(dist, y_test)
plot_pit_histogram(pit)

# Calibration curve and error
obs, exp = calibration_regression(dist, y_test)
cal_err = calibration_error(dist, y_test)
plot_calibration_curve(obs, exp)
```

For survival models, `concordance_index` and `calibration_survival` are also available.

## 📊 Benchmarks

Training speed on synthetic data (n_estimators=200, 3 trials):

| n_samples | NGBoost (s) | ngboost-lightning (s) | Speedup |
|---|---|---|---|
| 10,000 | 8.0 | 3.0 | 2.7x |
| 50,000 | 44.0 | 4.8 | 9.2x |
| 100,000 | 93.3 | 7.0 | **13.4x** |

Run benchmarks yourself:

```bash
uv run python -m benchmarks all
```

## 🧑‍💻 Development

```bash
git clone https://github.com/kschmaus/ngboost-lightning.git
cd ngboost-lightning
uv sync --group dev
```

Run tests (628 tests including NGBoost parity checks):

```bash
uv run pytest
```

Lint and type check:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy ngboost_lightning
```

## 📜 License

Apache 2.0
