"""Tests for the sklearn-compatible LightningBoostRegressor."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ngboost_lightning import LightningBoostRegressor
from ngboost_lightning.distributions import Normal
from tests._constants import SEED


@pytest.fixture()
def cal_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """California housing train/test split (small subset for speed)."""
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    cal = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        cal.data, cal.target, test_size=0.2, random_state=SEED
    )
    return X_train[:500], X_test[:200], y_train[:500], y_test[:200]


# ---------- Constructor ----------


class TestConstructor:
    """Tests for constructor behavior and param merging."""

    def test_default_params(self) -> None:
        """Default constructor creates a valid object with expected defaults."""
        reg = LightningBoostRegressor()
        assert reg.n_estimators == 500
        assert reg.learning_rate == 0.01
        assert reg.num_leaves == 31
        assert reg.lgbm_params is None

    def test_custom_params(self) -> None:
        """All constructor args are stored correctly."""
        reg = LightningBoostRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=64,
            reg_alpha=0.1,
            lgbm_params={"max_bin": 127},
        )
        assert reg.n_estimators == 100
        assert reg.learning_rate == 0.05
        assert reg.num_leaves == 64
        assert reg.reg_alpha == 0.1
        assert reg.lgbm_params == {"max_bin": 127}

    def test_lgbm_params_conflict(self) -> None:
        """Raises ValueError when same key in surfaced kwargs and lgbm_params."""
        reg = LightningBoostRegressor(
            num_leaves=64,
            lgbm_params={"num_leaves": 32},
        )
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 5))
        y = rng.normal(size=50)
        with pytest.raises(ValueError, match="num_leaves"):
            reg.fit(X, y)

    def test_lgbm_params_no_conflict(self) -> None:
        """Non-overlapping lgbm_params keys merge cleanly."""
        reg = LightningBoostRegressor(
            n_estimators=5,
            learning_rate=0.05,
            verbose=False,
            lgbm_params={"max_bin": 127},
        )
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 5))
        y = rng.normal(size=50)
        reg.fit(X, y)
        assert reg.n_estimators_ == 5


# ---------- Fit ----------


class TestFit:
    """Tests for fit behavior."""

    def test_fit_returns_self(self, cal_split: tuple) -> None:
        """fit() returns the estimator instance."""
        X_train, _, y_train, _ = cal_split
        reg = LightningBoostRegressor(n_estimators=5, learning_rate=0.05, verbose=False)
        result = reg.fit(X_train, y_train)
        assert result is reg

    def test_fitted_attributes(self, cal_split: tuple) -> None:
        """After fit, all expected attributes exist."""
        X_train, _, y_train, _ = cal_split
        reg = LightningBoostRegressor(
            n_estimators=10, learning_rate=0.05, verbose=False
        )
        reg.fit(X_train, y_train)
        assert hasattr(reg, "engine_")
        assert hasattr(reg, "n_features_in_")
        assert hasattr(reg, "n_estimators_")
        assert hasattr(reg, "init_params_")
        assert hasattr(reg, "boosters_")
        assert hasattr(reg, "scalings_")
        assert hasattr(reg, "train_loss_")
        assert reg.n_features_in_ == X_train.shape[1]
        assert reg.n_estimators_ == 10

    def test_fit_with_validation(self, cal_split: tuple) -> None:
        """Fit with X_val/y_val sets val_loss_ and best_val_loss_itr_."""
        X_train, X_test, y_train, y_test = cal_split
        reg = LightningBoostRegressor(
            n_estimators=50, learning_rate=0.05, verbose=False
        )
        reg.fit(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            early_stopping_rounds=10,
        )
        assert hasattr(reg, "val_loss_")
        assert hasattr(reg, "best_val_loss_itr_")
        assert reg.best_val_loss_itr_ is not None


# ---------- Predict ----------


class TestPredict:
    """Tests for prediction methods."""

    def test_predict_shape(self, cal_split: tuple) -> None:
        """Predict returns [n_samples]."""
        X_train, X_test, y_train, _ = cal_split
        reg = LightningBoostRegressor(
            n_estimators=10, learning_rate=0.05, verbose=False
        )
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_predict_finite(self, cal_split: tuple) -> None:
        """All predictions should be finite."""
        X_train, X_test, y_train, _ = cal_split
        reg = LightningBoostRegressor(
            n_estimators=10, learning_rate=0.05, verbose=False
        )
        reg.fit(X_train, y_train)
        assert np.all(np.isfinite(reg.predict(X_test)))

    def test_pred_dist_type(self, cal_split: tuple) -> None:
        """pred_dist returns a Normal instance."""
        X_train, X_test, y_train, _ = cal_split
        reg = LightningBoostRegressor(
            n_estimators=10, learning_rate=0.05, verbose=False
        )
        reg.fit(X_train, y_train)
        dist = reg.pred_dist(X_test)
        assert isinstance(dist, Normal)
        assert len(dist) == len(X_test)

    def test_pred_dist_scale_positive(self, cal_split: tuple) -> None:
        """Predicted scale should always be positive."""
        X_train, X_test, y_train, _ = cal_split
        reg = LightningBoostRegressor(
            n_estimators=10, learning_rate=0.05, verbose=False
        )
        reg.fit(X_train, y_train)
        assert np.all(reg.pred_dist(X_test).scale > 0)

    def test_predict_before_fit_raises(self) -> None:
        """Predict before fit raises NotFittedError."""
        from sklearn.exceptions import NotFittedError

        reg = LightningBoostRegressor()
        with pytest.raises(NotFittedError):
            reg.predict(np.zeros((5, 3)))


# ---------- Score ----------


class TestScore:
    """Tests for the score method."""

    def test_score_is_finite(self, cal_split: tuple) -> None:
        """score() returns a finite float."""
        X_train, X_test, y_train, y_test = cal_split
        reg = LightningBoostRegressor(
            n_estimators=20, learning_rate=0.05, verbose=False
        )
        reg.fit(X_train, y_train)
        s = reg.score(X_test, y_test)
        assert np.isfinite(s)

    def test_score_sign_convention(self, cal_split: tuple) -> None:
        """Better-fit model should have higher score (less negative NLL)."""
        X_train, _, y_train, _ = cal_split
        reg_few = LightningBoostRegressor(
            n_estimators=5, learning_rate=0.05, verbose=False
        )
        reg_few.fit(X_train, y_train)

        reg_many = LightningBoostRegressor(
            n_estimators=100, learning_rate=0.05, verbose=False
        )
        reg_many.fit(X_train, y_train)

        # More iterations should fit better -> higher score on training data
        assert reg_many.score(X_train, y_train) > reg_few.score(X_train, y_train)


# ---------- Feature importances ----------


class TestFeatureImportances:
    """Tests for feature_importances_ property."""

    def test_shape(self, cal_split: tuple) -> None:
        """Shape should be [n_params, n_features]."""
        X_train, _, y_train, _ = cal_split
        reg = LightningBoostRegressor(
            n_estimators=20, learning_rate=0.05, verbose=False
        )
        reg.fit(X_train, y_train)
        fi = reg.feature_importances_
        assert fi.shape == (2, X_train.shape[1])

    def test_nonnegative(self, cal_split: tuple) -> None:
        """All importance values should be >= 0."""
        X_train, _, y_train, _ = cal_split
        reg = LightningBoostRegressor(
            n_estimators=20, learning_rate=0.05, verbose=False
        )
        reg.fit(X_train, y_train)
        assert np.all(reg.feature_importances_ >= 0)

    def test_normalized(self, cal_split: tuple) -> None:
        """Each row should sum to ~1.0."""
        X_train, _, y_train, _ = cal_split
        reg = LightningBoostRegressor(
            n_estimators=20, learning_rate=0.05, verbose=False
        )
        reg.fit(X_train, y_train)
        row_sums = reg.feature_importances_.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


# ---------- sklearn compatibility ----------


class TestSklearnCompat:
    """Tests for sklearn integration (clone, Pipeline, cross_val_score)."""

    def test_clone_preserves_params(self) -> None:
        """clone() produces an equivalent unfitted estimator."""
        reg = LightningBoostRegressor(
            n_estimators=100, learning_rate=0.05, num_leaves=64
        )
        cloned = clone(reg)
        assert cloned.n_estimators == 100
        assert cloned.learning_rate == 0.05
        assert cloned.num_leaves == 64

    def test_clone_preserves_lgbm_params(self) -> None:
        """Nested lgbm_params dict survives clone."""
        reg = LightningBoostRegressor(lgbm_params={"max_bin": 127})
        cloned = clone(reg)
        assert cloned.lgbm_params == {"max_bin": 127}

    def test_pipeline(self, cal_split: tuple) -> None:
        """Works inside a sklearn Pipeline with StandardScaler."""
        X_train, X_test, y_train, _ = cal_split
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "reg",
                    LightningBoostRegressor(
                        n_estimators=10, learning_rate=0.05, verbose=False
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert np.all(np.isfinite(preds))

    def test_cross_val_score(self, cal_split: tuple) -> None:
        """cross_val_score runs without error and returns finite scores."""
        X_train, _, y_train, _ = cal_split
        reg = LightningBoostRegressor(
            n_estimators=10, learning_rate=0.05, verbose=False
        )
        scores = cross_val_score(reg, X_train, y_train, cv=3)
        assert len(scores) == 3
        assert np.all(np.isfinite(scores))

    def test_get_set_params_roundtrip(self) -> None:
        """set_params(**get_params()) is idempotent."""
        reg = LightningBoostRegressor(
            n_estimators=100,
            num_leaves=64,
            lgbm_params={"max_bin": 127},
        )
        params = reg.get_params()
        reg2 = LightningBoostRegressor()
        reg2.set_params(**params)
        assert reg2.get_params() == params
