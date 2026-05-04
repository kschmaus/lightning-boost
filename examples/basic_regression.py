"""Basic probabilistic regression with LightningBoostRegressor.

Demonstrates fitting, point prediction, and uncertainty estimation
on the California Housing dataset.
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from ngboost_lightning import LightningBoostRegressor

# ── Load data ────────────────────────────────────────────────────────
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ── Fit model ────────────────────────────────────────────────────────
reg = LightningBoostRegressor(
    n_estimators=200,
    learning_rate=0.05,
    random_state=0,
    verbose=True,
    verbose_eval=50,
)
reg.fit(X_train, y_train)

# ── Point predictions ────────────────────────────────────────────────
preds = reg.predict(X_test)
mse = float(np.mean((preds - y_test) ** 2))
print(f"\nTest MSE: {mse:.4f}")

# ── Probabilistic predictions ────────────────────────────────────────
dist = reg.pred_dist(X_test)
print(f"Predicted mean  (first 5): {dist.mean()[:5].round(3)}")
print(f"Predicted scale (first 5): {dist.scale[:5].round(3)}")

# 90% prediction interval
lower = dist.ppf(0.05)
upper = dist.ppf(0.95)
coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
print(f"90% prediction interval coverage: {coverage:.1%}")

# ── NLL score (sklearn convention: higher is better) ─────────────────
score = reg.score(X_test, y_test)
print(f"Negative mean NLL: {score:.4f}")

# ── Feature importances ─────────────────────────────────────────────
importances = reg.feature_importances_
param_names = ["loc (mean)", "log_scale (uncertainty)"]
feature_names = fetch_california_housing().feature_names
for k, name in enumerate(param_names):
    top_idx = np.argsort(importances[k])[::-1][:3]
    top = [(feature_names[i], f"{importances[k, i]:.3f}") for i in top_idx]
    print(f"Top 3 features for {name}: {top}")
