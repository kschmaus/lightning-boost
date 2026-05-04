"""Tests for validation_fraction auto-split."""

import numpy as np
import pytest

from ngboost_lightning import LightningBoostClassifier
from ngboost_lightning import LightningBoostRegressor
from tests._constants import SEED

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Simple regression dataset."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(300, 5))
    y = 3 * X[:, 0] + rng.normal(scale=0.5, size=300)
    return X, y


@pytest.fixture()
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Simple binary classification dataset."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] + X[:, 1] + rng.normal(scale=0.3, size=300) > 0).astype(float)
    return X, y


# ===================================================================
# Regressor
# ===================================================================


class TestRegressorValidationFraction:
    """Auto validation split on LightningBoostRegressor."""

    def test_auto_split_produces_early_stopping(self, regression_data: tuple) -> None:
        """Setting validation_fraction triggers early stopping."""
        X, y = regression_data
        reg = LightningBoostRegressor(
            n_estimators=200,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
            validation_fraction=0.2,
        )
        reg.fit(X, y)
        assert hasattr(reg, "val_loss_")
        assert reg.n_estimators_ < 200

    def test_auto_split_default_early_stopping_rounds(
        self, regression_data: tuple
    ) -> None:
        """When validation_fraction is set, early_stopping_rounds defaults to 20."""
        X, y = regression_data
        reg = LightningBoostRegressor(
            n_estimators=500,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
            validation_fraction=0.2,
        )
        reg.fit(X, y)
        # Should have early stopped (val_loss_ exists and stopped before 500)
        assert hasattr(reg, "val_loss_")
        assert reg.n_estimators_ < 500

    def test_auto_split_custom_early_stopping_rounds(
        self, regression_data: tuple
    ) -> None:
        """Explicit early_stopping_rounds overrides the default 20."""
        X, y = regression_data
        reg = LightningBoostRegressor(
            n_estimators=200,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
            validation_fraction=0.2,
        )
        reg.fit(X, y, early_stopping_rounds=3)
        assert hasattr(reg, "val_loss_")

    def test_explicit_val_still_works(self, regression_data: tuple) -> None:
        """Explicit X_val/y_val works without validation_fraction."""
        X, y = regression_data
        reg = LightningBoostRegressor(
            n_estimators=200,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
        )
        reg.fit(
            X[:240],
            y[:240],
            X_val=X[240:],
            y_val=y[240:],
            early_stopping_rounds=5,
        )
        assert hasattr(reg, "val_loss_")

    def test_conflict_raises(self, regression_data: tuple) -> None:
        """Both validation_fraction and X_val/y_val raises ValueError."""
        X, y = regression_data
        reg = LightningBoostRegressor(
            n_estimators=50,
            verbose=False,
            validation_fraction=0.2,
        )
        with pytest.raises(ValueError, match="validation_fraction"):
            reg.fit(X[:200], y[:200], X_val=X[200:], y_val=y[200:])

    def test_auto_split_with_sample_weight(self, regression_data: tuple) -> None:
        """Sample weights are split consistently with data."""
        X, y = regression_data
        w = np.ones(len(y))
        w[:50] = 5.0
        reg = LightningBoostRegressor(
            n_estimators=200,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
            validation_fraction=0.2,
        )
        # Should not raise weight mismatch error — weights are auto-split
        reg.fit(X, y, sample_weight=w)
        assert hasattr(reg, "val_loss_")

    def test_deterministic_with_random_state(self, regression_data: tuple) -> None:
        """Same random_state produces identical splits and results."""
        X, y = regression_data
        losses = []
        for _ in range(2):
            reg = LightningBoostRegressor(
                n_estimators=30,
                learning_rate=0.05,
                verbose=False,
                random_state=SEED,
                validation_fraction=0.2,
            )
            reg.fit(X, y)
            losses.append(reg.val_loss_)
        np.testing.assert_allclose(losses[0], losses[1])


# ===================================================================
# Classifier
# ===================================================================


class TestClassifierValidationFraction:
    """Auto validation split on LightningBoostClassifier."""

    def test_auto_split_produces_early_stopping(
        self, classification_data: tuple
    ) -> None:
        """Setting validation_fraction triggers early stopping."""
        X, y = classification_data
        clf = LightningBoostClassifier(
            n_estimators=200,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
            validation_fraction=0.2,
        )
        clf.fit(X, y)
        assert hasattr(clf, "val_loss_")
        assert clf.n_estimators_ < 200

    def test_stratified_split(self, classification_data: tuple) -> None:
        """Auto-split uses stratification to preserve class balance."""
        X, y = classification_data
        clf = LightningBoostClassifier(
            n_estimators=30,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
            validation_fraction=0.2,
        )
        # Should not fail due to missing classes in val set
        clf.fit(X, y)
        assert hasattr(clf, "val_loss_")

    def test_conflict_raises(self, classification_data: tuple) -> None:
        """Both validation_fraction and X_val/y_val raises ValueError."""
        X, y = classification_data
        clf = LightningBoostClassifier(
            n_estimators=50,
            verbose=False,
            validation_fraction=0.2,
        )
        with pytest.raises(ValueError, match="validation_fraction"):
            clf.fit(X[:200], y[:200], X_val=X[200:], y_val=y[200:])

    def test_auto_split_with_sample_weight(self, classification_data: tuple) -> None:
        """Sample weights are split consistently with data."""
        X, y = classification_data
        w = np.ones(len(y))
        w[y == 1] = 5.0
        clf = LightningBoostClassifier(
            n_estimators=200,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
            validation_fraction=0.2,
        )
        clf.fit(X, y, sample_weight=w)
        assert hasattr(clf, "val_loss_")
