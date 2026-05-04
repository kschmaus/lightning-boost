# Advanced Features

This page covers features beyond the basics: regularization via subsampling,
custom loss tracking, staged predictions, and LightGBM parameter passthrough.

## Sample Weights

All estimators accept `sample_weight` in `fit()`:

```python
reg.fit(X_train, y_train, sample_weight=weights)
```

Weights affect both gradient computation and the scoring rule's `total_score`.

## Early Stopping

### With Explicit Validation Data

```python
reg.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    early_stopping_rounds=10,
)
```

Training stops when the validation loss does not improve for
`early_stopping_rounds` consecutive iterations. The final model uses the
best iteration.

### With Automatic Validation Split

Set `validation_fraction` to hold out a portion of training data automatically:

```python
reg = LightningBoostRegressor(
    validation_fraction=0.1,
    early_stopping_rounds=10,
)
reg.fit(X_train, y_train)  # 10% held out for early stopping
```

If both `validation_fraction` and explicit `X_val`/`y_val` are provided,
the explicit data takes priority.

## Column Subsampling

`col_sample` controls per-iteration feature subsampling at the boosting level.
Each iteration, a random subset of columns is selected and all K parameter
boosters see the same subset.

```python
reg = LightningBoostRegressor(col_sample=0.8, n_estimators=200)
reg.fit(X_train, y_train)
```

!!! note
    This is distinct from LightGBM's `colsample_bytree`, which subsamples
    per tree. `col_sample` operates at the outer boosting loop level — all
    K parameter boosters share the same column mask each iteration.

During prediction, the per-iteration column masks are replayed so that each
tree only sees the features it was trained on.

## Custom Loss Monitors

By default, training and validation loss are recorded using the scoring rule's
`total_score`. You can override this with custom callables:

```python
def mse_monitor(dist, y):
    """Track MSE instead of NLL."""
    return float(((dist.mean() - y) ** 2).mean())

reg.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    train_loss_monitor=mse_monitor,
    val_loss_monitor=mse_monitor,
    early_stopping_rounds=10,
)
```

The monitor signature is `(Distribution, NDArray) -> float`.

**Key behavior:**

- Monitors only affect loss recording and early stopping decisions.
- Gradients always come from the scoring rule, regardless of the monitor.
- The validation monitor drives early stopping — lower values are better.

After fitting, recorded losses are available as:

```python
reg.engine_.train_loss_  # list of per-iteration training losses
reg.engine_.val_loss_    # list of per-iteration validation losses
```

## Staged Predictions

Iterate over predictive distributions at each boosting iteration:

```python
for i, dist in enumerate(reg.staged_pred_dist(X_test)):
    mse = float(((dist.mean() - y_test) ** 2).mean())
    print(f"Iteration {i}: MSE = {mse:.4f}")
```

This is useful for analyzing how predictions evolve during boosting and
for custom early stopping analysis.

## Minibatch Training

`minibatch_frac` subsamples training rows each iteration for gradient
computation (NGBoost-style). This is distinct from LightGBM's `subsample`:

```python
reg = LightningBoostRegressor(minibatch_frac=0.5, n_estimators=200)
```

## LightGBM Parameter Passthrough

Common LightGBM parameters are surfaced as constructor kwargs:

- `num_leaves`, `max_depth`, `min_child_samples`
- `subsample`, `colsample_bytree`
- `reg_alpha`, `reg_lambda`

Less common parameters go through `lgbm_params`:

```python
reg = LightningBoostRegressor(
    num_leaves=63,
    lgbm_params={"max_bin": 512, "min_gain_to_split": 0.01},
)
```

If a key appears in both constructor kwargs and `lgbm_params`, a `ValueError`
is raised to prevent silent conflicts.

## sklearn Integration

All estimators work with scikit-learn utilities:

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

`get_params()` and `set_params()` work correctly for grid search
and randomized search.
