"""Side-by-side comparison of lightning-boost and NGBoost.

Trains both models on the California Housing dataset and compares
NLL, MSE, and calibration.
"""

import numpy as np
from ngboost import NGBRegressor
from ngboost.distns import Normal as NGBNormal
from scipy.stats import ks_2samp
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from ngboost_lightning import LightningBoostRegressor

# ── Load data ────────────────────────────────────────────────────────
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=24601
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=24601
)

# ── Common hyperparameters ───────────────────────────────────────────
PARAMS = dict(n_estimators=200, learning_rate=0.05, random_state=24601)
EARLY_STOPPING_ROUNDS = 20

# ── Fit NGBoost ──────────────────────────────────────────────────────
print("Training NGBoost...")
ngb = NGBRegressor(Dist=NGBNormal, verbose=False, **PARAMS)
ngb.fit(
    X_train,
    y_train,
    X_val=X_val,
    Y_val=y_val,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
)

# ── Fit lightning-boost ──────────────────────────────────────────────
print("Training lightning-boost...")
lb = LightningBoostRegressor(
    verbose=False, num_leaves=8, max_depth=3, min_child_samples=1, **PARAMS
)
lb.fit(
    X_train,
    y_train,
    X_val=X_val,
    y_val=y_val,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
)

# ── Compare NLL ──────────────────────────────────────────────────────
ngb_dist = ngb.pred_dist(X_test)
lb_dist = lb.pred_dist(X_test)

ngb_nll = float(-ngb_dist.logpdf(y_test).mean())
lb_nll = float(lb_dist.total_score(y_test))

print(f"\n{'Metric':<25} {'NGBoost':>10} {'LB':>10} {'Diff':>10}")
print("-" * 57)
nll_diff = (lb_nll - ngb_nll) / ngb_nll
print(f"{'NLL (lower=better)':<25} {ngb_nll:>10.4f} {lb_nll:>10.4f} {nll_diff:>+10.2%}")

# ── Compare MSE ──────────────────────────────────────────────────────
ngb_preds = ngb.predict(X_test)
lb_preds = lb.predict(X_test)

ngb_mse = float(np.mean((ngb_preds - y_test) ** 2))
lb_mse = float(np.mean((lb_preds - y_test) ** 2))

mse_diff = (lb_mse - ngb_mse) / ngb_mse
print(f"{'MSE (lower=better)':<25} {ngb_mse:>10.4f} {lb_mse:>10.4f} {mse_diff:>+10.2%}")

# ── Compare calibration via PIT ──────────────────────────────────────
ngb_pit = ngb_dist.cdf(y_test)
lb_pit = lb_dist.cdf(y_test)

ks_stat, ks_pval = ks_2samp(ngb_pit, lb_pit)
print(f"{'KS statistic (PIT diff)':<25} {ks_stat:>10.4f}")
print(f"{'KS p-value':<25} {ks_pval:>10.4f}")

# ── 90% coverage ─────────────────────────────────────────────────────
for name, d in [("NGBoost", ngb_dist), ("LB", lb_dist)]:
    if hasattr(d, "ppf"):
        lo, hi = d.ppf(0.05), d.ppf(0.95)
    else:
        # NGBoost Normal uses scipy dist interface
        lo = d.dist.ppf(0.05)
        hi = d.dist.ppf(0.95)
    cov = float(np.mean((y_test >= lo) & (y_test <= hi)))
    print(f"90% coverage ({name}): {cov:.1%}")

print("\nDone. Both models implement the same natural gradient boosting algorithm.")
print("Differences arise from LightGBM (histogram) vs sklearn (exact) tree splitting.")
