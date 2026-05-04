# Quickstart

This guide walks through fitting a probabilistic model, inspecting the
predictive distribution, and evaluating calibration.

## Regression

```python
from ngboost_lightning import LightningBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0,
)

reg = LightningBoostRegressor(
    n_estimators=200,
    learning_rate=0.05,
    random_state=0,
)
reg.fit(X_train, y_train)
```

### Point Predictions

```python
preds = reg.predict(X_test)  # conditional mean
```

### Full Predictive Distribution

`pred_dist` returns a distribution object with one set of parameters per sample:

```python
dist = reg.pred_dist(X_test)

dist.mean()       # conditional mean (same as predict)
dist.scale        # predicted standard deviation
dist.ppf(0.05)    # 5th percentile
dist.ppf(0.95)    # 95th percentile
dist.cdf(y_test)  # CDF at observed values
dist.logpdf(y_test)  # log-density at observed values
```

### Prediction Intervals

```python
import numpy as np

lower = dist.ppf(0.05)
upper = dist.ppf(0.95)
coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
print(f"90% prediction interval coverage: {coverage:.1%}")
```

## Binary Classification

```python
from ngboost_lightning import LightningBoostClassifier

clf = LightningBoostClassifier(n_estimators=200, learning_rate=0.05)
clf.fit(X_train, y_train)

clf.predict(X_test)        # class labels
clf.predict_proba(X_test)  # probabilities, shape [n_samples, 2]
```

## Multiclass Classification

For multiclass, pass a `k_categorical(K)` distribution explicitly:

```python
from ngboost_lightning import LightningBoostClassifier, k_categorical

clf = LightningBoostClassifier(
    dist=k_categorical(3),
    n_estimators=200,
    learning_rate=0.05,
)
clf.fit(X_train, y_train)

clf.predict_proba(X_test)  # probabilities, shape [n_samples, 3]
```

## Early Stopping

Pass validation data to `fit()` to enable early stopping:

```python
reg.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    early_stopping_rounds=10,
)
```

Or let ngboost-lightning split the training data automatically:

```python
reg = LightningBoostRegressor(
    validation_fraction=0.1,
    early_stopping_rounds=10,
)
reg.fit(X_train, y_train)
```

## Feature Importances

Importances are available per distribution parameter:

```python
importances = reg.feature_importances_  # shape [n_params, n_features]
# Row 0 = mean parameter, Row 1 = log_scale parameter (for Normal)
# Each row sums to 1.0
```

## Next Steps

- [Distributions](../user-guide/distributions.md) — choose the right distribution for your data
- [Scoring Rules](../user-guide/scoring-rules.md) — LogScore vs CRPS
- [Advanced Features](../user-guide/advanced.md) — col_sample, loss monitors, staged predictions
