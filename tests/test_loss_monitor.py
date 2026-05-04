"""Tests for custom loss monitors (train_loss_monitor, val_loss_monitor)."""

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from ngboost_lightning.distributions.base import Distribution
from ngboost_lightning.engine import NGBEngine
from ngboost_lightning.regressor import LightningBoostRegressor
from tests._constants import SEED


@pytest.fixture()
def cal_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """California housing train/test split (small subset for speed)."""
    cal = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        cal.data, cal.target, test_size=0.2, random_state=SEED
    )
    return X_train[:500], X_test[:200], y_train[:500], y_test[:200]


def _mse_monitor(pred_dist: Distribution, y: np.ndarray) -> float:
    """MSE-based loss monitor."""
    return float(np.mean((pred_dist.mean() - y) ** 2))


def _constant_monitor(_pred_dist: Distribution, _y: np.ndarray) -> float:
    """Returns a constant value for testing."""
    return 42.0


class TestTrainLossMonitor:
    """Custom train_loss_monitor replaces recorded training loss."""

    def test_train_loss_uses_monitor(self, cal_split: tuple) -> None:
        """When train_loss_monitor is set, train_loss_ should reflect it."""
        X_train, _, y_train, _ = cal_split
        eng = NGBEngine(
            n_estimators=5, learning_rate=0.05, random_state=SEED, verbose=False
        )
        eng.fit(X_train, y_train, train_loss_monitor=_constant_monitor)
        # All recorded losses should be 42.0
        assert all(loss == 42.0 for loss in eng.train_loss_)

    def test_train_loss_default_unchanged(self, cal_split: tuple) -> None:
        """Without monitor, train_loss_ uses the default scoring rule."""
        X_train, _, y_train, _ = cal_split
        eng = NGBEngine(
            n_estimators=5, learning_rate=0.05, random_state=SEED, verbose=False
        )
        eng.fit(X_train, y_train)
        # Default NLL values should not be 42.0
        assert not any(loss == 42.0 for loss in eng.train_loss_)

    def test_train_monitor_does_not_affect_gradients(self, cal_split: tuple) -> None:
        """Custom train monitor should not change the model's predictions."""
        X_train, X_test, y_train, _ = cal_split
        kwargs = {
            "n_estimators": 10,
            "learning_rate": 0.05,
            "random_state": SEED,
            "verbose": False,
        }
        # Fit without monitor
        eng1 = NGBEngine(**kwargs)
        eng1.fit(X_train, y_train)

        # Fit with monitor — gradients should be identical
        eng2 = NGBEngine(**kwargs)
        eng2.fit(X_train, y_train, train_loss_monitor=_mse_monitor)

        # Predictions should be identical (monitor only affects recorded loss)
        np.testing.assert_array_equal(eng1.predict(X_test), eng2.predict(X_test))
        # But recorded losses differ
        assert eng1.train_loss_ != eng2.train_loss_

    def test_train_monitor_mse_values(self, cal_split: tuple) -> None:
        """MSE monitor should record MSE values."""
        X_train, _, y_train, _ = cal_split
        eng = NGBEngine(
            n_estimators=5, learning_rate=0.05, random_state=SEED, verbose=False
        )
        eng.fit(X_train, y_train, train_loss_monitor=_mse_monitor)
        # MSE values should be positive and finite
        assert all(np.isfinite(loss) and loss > 0 for loss in eng.train_loss_)


class TestValLossMonitor:
    """Custom val_loss_monitor replaces validation loss and early stopping."""

    def test_val_loss_uses_monitor(self, cal_split: tuple) -> None:
        """When val_loss_monitor is set, val_loss_ should reflect it."""
        X_train, X_test, y_train, y_test = cal_split
        eng = NGBEngine(
            n_estimators=10, learning_rate=0.05, random_state=SEED, verbose=False
        )
        eng.fit(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            val_loss_monitor=_constant_monitor,
        )
        assert all(loss == 42.0 for loss in eng.val_loss_)

    def test_val_monitor_triggers_early_stopping(self, cal_split: tuple) -> None:
        """A val monitor that increases should trigger early stopping."""
        call_count = 0

        def _increasing_monitor(_pred_dist: Distribution, _y: np.ndarray) -> float:
            nonlocal call_count
            call_count += 1
            # Return increasing values — early stopping should fire
            return float(call_count)

        X_train, X_test, y_train, y_test = cal_split
        eng = NGBEngine(
            n_estimators=100, learning_rate=0.05, random_state=SEED, verbose=False
        )
        eng.fit(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            early_stopping_rounds=5,
            val_loss_monitor=_increasing_monitor,
        )
        # Should have stopped early since loss only increases
        assert eng.n_estimators_ < 100
        # Best iteration should be the first one
        assert eng.best_val_loss_itr_ == 0

    def test_val_monitor_default_unchanged(self, cal_split: tuple) -> None:
        """Without monitor, val_loss_ uses the scoring rule as before."""
        X_train, X_test, y_train, y_test = cal_split
        eng = NGBEngine(
            n_estimators=10, learning_rate=0.05, random_state=SEED, verbose=False
        )
        eng.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        assert not any(loss == 42.0 for loss in eng.val_loss_)


class TestBothMonitors:
    """Using both monitors together."""

    def test_both_monitors(self, cal_split: tuple) -> None:
        """Both monitors can be used simultaneously."""
        X_train, X_test, y_train, y_test = cal_split
        eng = NGBEngine(
            n_estimators=5, learning_rate=0.05, random_state=SEED, verbose=False
        )
        eng.fit(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            train_loss_monitor=_mse_monitor,
            val_loss_monitor=_mse_monitor,
        )
        # Both should record MSE values (positive, finite)
        assert all(np.isfinite(loss) and loss > 0 for loss in eng.train_loss_)
        assert all(np.isfinite(loss) and loss > 0 for loss in eng.val_loss_)


class TestRegressorMonitor:
    """Monitors work through the regressor interface."""

    def test_regressor_train_monitor(self, cal_split: tuple) -> None:
        """Regressor passes train_loss_monitor to engine."""
        X_train, _, y_train, _ = cal_split
        reg = LightningBoostRegressor(
            n_estimators=5,
            learning_rate=0.05,
            random_state=SEED,
            verbose=False,
        )
        reg.fit(X_train, y_train, train_loss_monitor=_constant_monitor)
        assert all(loss == 42.0 for loss in reg.train_loss_)

    def test_regressor_val_monitor(self, cal_split: tuple) -> None:
        """Regressor passes val_loss_monitor to engine."""
        X_train, X_test, y_train, y_test = cal_split
        reg = LightningBoostRegressor(
            n_estimators=5,
            learning_rate=0.05,
            random_state=SEED,
            verbose=False,
        )
        reg.fit(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            val_loss_monitor=_constant_monitor,
        )
        assert all(loss == 42.0 for loss in reg.val_loss_)
