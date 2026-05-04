"""Basic probabilistic classification with LightningBoostClassifier.

Demonstrates binary classification on Breast Cancer, multiclass on Iris,
and probabilistic predictions (predict_proba, pred_dist).
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ngboost_lightning import LightningBoostClassifier
from ngboost_lightning import k_categorical

# ── Binary classification (Breast Cancer) ────────────────────────────
print("=" * 60)
print("Binary Classification — Breast Cancer")
print("=" * 60)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = LightningBoostClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=0,
)
clf.fit(X_train, y_train)

# Point predictions
preds = clf.predict(X_test)
accuracy = float(np.mean(preds == y_test))
print(f"\nTest accuracy: {accuracy:.1%}")

# Probabilistic predictions
proba = clf.predict_proba(X_test)
print(f"Predicted probabilities (first 5):\n{proba[:5].round(3)}")

# Predictive distribution object
dist = clf.pred_dist(X_test)
print(f"Distribution type: {type(dist).__name__}")
print(f"Class probabilities shape: {dist.probs.shape}")

# NLL score (higher = better fit)
score = clf.score(X_test, y_test)
print(f"Negative mean NLL: {score:.4f}")

# Feature importances
importances = clf.feature_importances_
feature_names = load_breast_cancer().feature_names
top_idx = np.argsort(importances[0])[::-1][:5]
top = [(feature_names[i], f"{importances[0, i]:.3f}") for i in top_idx]
print(f"Top 5 features: {top}")

# ── Multiclass classification (Iris) ─────────────────────────────────
print(f"\n{'=' * 60}")
print("Multiclass Classification — Iris (K=3)")
print("=" * 60)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf_multi = LightningBoostClassifier(
    dist=k_categorical(3),
    n_estimators=200,
    learning_rate=0.05,
    random_state=0,
)
clf_multi.fit(X_train, y_train)

preds = clf_multi.predict(X_test)
accuracy = float(np.mean(preds == y_test))
print(f"\nTest accuracy: {accuracy:.1%}")

proba = clf_multi.predict_proba(X_test)
print(f"Predicted probabilities (first 5):\n{proba[:5].round(3)}")

score = clf_multi.score(X_test, y_test)
print(f"Negative mean NLL: {score:.4f}")
