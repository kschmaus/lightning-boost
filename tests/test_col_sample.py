"""Tests for column subsampling (col_sample)."""

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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


class TestColSampleEngine:
    """Column subsampling at the engine level."""

    def test_col_sample_1_no_masking(self, cal_split: tuple) -> None:
        """col_sample=1.0 stores all column indices each iteration."""
        X_train, _, y_train, _ = cal_split
        eng = NGBEngine(
            n_estimators=5, learning_rate=0.05, col_sample=1.0, verbose=False
        )
        eng.fit(X_train, y_train)
        n_features = X_train.shape[1]
        for idx in eng.col_indices_:
            np.testing.assert_array_equal(idx, np.arange(n_features))

    def test_col_sample_stores_indices(self, cal_split: tuple) -> None:
        """col_sample<1.0 stores a subset of columns per iteration."""
        X_train, _, y_train, _ = cal_split
        n_features = X_train.shape[1]
        eng = NGBEngine(
            n_estimators=10,
            learning_rate=0.05,
            col_sample=0.5,
            random_state=SEED,
            verbose=False,
        )
        eng.fit(X_train, y_train)
        expected_n = max(1, int(np.ceil(0.5 * n_features)))
        assert len(eng.col_indices_) == 10
        for idx in eng.col_indices_:
            assert len(idx) == expected_n
            # Indices should be sorted
            np.testing.assert_array_equal(idx, np.sort(idx))
            # All indices valid
            assert np.all(idx >= 0)
            assert np.all(idx < n_features)

    def test_col_sample_different_subsets(self, cal_split: tuple) -> None:
        """Different iterations should generally have different subsets."""
        X_train, _, y_train, _ = cal_split
        eng = NGBEngine(
            n_estimators=20,
            learning_rate=0.05,
            col_sample=0.5,
            random_state=SEED,
            verbose=False,
        )
        eng.fit(X_train, y_train)
        # Not all iterations should have identical subsets
        unique_subsets = {tuple(idx) for idx in eng.col_indices_}
        assert len(unique_subsets) > 1

    def test_col_sample_predictions_differ(self, cal_split: tuple) -> None:
        """col_sample should change predictions vs no subsampling."""
        X_train, X_test, y_train, _ = cal_split
        eng_full = NGBEngine(
            n_estimators=20,
            learning_rate=0.05,
            col_sample=1.0,
            random_state=SEED,
            verbose=False,
        )
        eng_full.fit(X_train, y_train)

        eng_sub = NGBEngine(
            n_estimators=20,
            learning_rate=0.05,
            col_sample=0.5,
            random_state=SEED,
            verbose=False,
        )
        eng_sub.fit(X_train, y_train)

        preds_full = eng_full.predict(X_test)
        preds_sub = eng_sub.predict(X_test)
        # Predictions should differ (not identical)
        assert not np.allclose(preds_full, preds_sub)

    def test_col_sample_reproducible(self, cal_split: tuple) -> None:
        """Same random_state should give same results."""
        X_train, X_test, y_train, _ = cal_split
        kwargs = {
            "n_estimators": 10,
            "learning_rate": 0.05,
            "col_sample": 0.5,
            "random_state": SEED,
            "verbose": False,
        }
        eng1 = NGBEngine(**kwargs)
        eng1.fit(X_train, y_train)
        eng2 = NGBEngine(**kwargs)
        eng2.fit(X_train, y_train)

        np.testing.assert_array_equal(eng1.predict(X_test), eng2.predict(X_test))
        for i1, i2 in zip(eng1.col_indices_, eng2.col_indices_, strict=True):
            np.testing.assert_array_equal(i1, i2)

    def test_col_sample_staged_predict(self, cal_split: tuple) -> None:
        """staged_predict_params should work with col_sample."""
        X_train, X_test, y_train, _ = cal_split
        eng = NGBEngine(
            n_estimators=5,
            learning_rate=0.05,
            col_sample=0.5,
            random_state=SEED,
            verbose=False,
        )
        eng.fit(X_train, y_train)

        staged = list(eng.staged_predict_params(X_test))
        assert len(staged) == 5
        # Last staged params should match predict_params
        np.testing.assert_allclose(staged[-1], eng.predict_params(X_test), rtol=1e-10)

    def test_col_sample_does_not_mutate_input(self, cal_split: tuple) -> None:
        """Fitting and predicting should not mutate input arrays."""
        X_train, X_test, y_train, _ = cal_split
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        eng = NGBEngine(
            n_estimators=5,
            learning_rate=0.05,
            col_sample=0.5,
            random_state=SEED,
            verbose=False,
        )
        eng.fit(X_train, y_train)
        eng.predict(X_test)

        np.testing.assert_array_equal(X_train, X_train_copy)
        np.testing.assert_array_equal(X_test, X_test_copy)

    def test_col_sample_with_early_stopping(self, cal_split: tuple) -> None:
        """col_sample should work with validation / early stopping."""
        X_train, X_test, y_train, y_test = cal_split
        eng = NGBEngine(
            n_estimators=100,
            learning_rate=0.05,
            col_sample=0.5,
            random_state=SEED,
            verbose=False,
        )
        eng.fit(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            early_stopping_rounds=10,
        )
        # Should have stopped early or completed
        assert eng.n_estimators_ <= 100
        assert len(eng.col_indices_) == eng.n_estimators_


class TestColSampleValidation:
    """Input validation for col_sample."""

    def test_col_sample_zero_raises(self, cal_split: tuple) -> None:
        """col_sample=0 should raise ValueError."""
        X_train, _, y_train, _ = cal_split
        eng = NGBEngine(col_sample=0.0, verbose=False)
        with pytest.raises(ValueError, match="col_sample must be in"):
            eng.fit(X_train, y_train)

    def test_col_sample_negative_raises(self, cal_split: tuple) -> None:
        """Negative col_sample should raise ValueError."""
        X_train, _, y_train, _ = cal_split
        eng = NGBEngine(col_sample=-0.5, verbose=False)
        with pytest.raises(ValueError, match="col_sample must be in"):
            eng.fit(X_train, y_train)

    def test_col_sample_over_1_raises(self, cal_split: tuple) -> None:
        """col_sample > 1.0 should raise ValueError."""
        X_train, _, y_train, _ = cal_split
        eng = NGBEngine(col_sample=1.5, verbose=False)
        with pytest.raises(ValueError, match="col_sample must be in"):
            eng.fit(X_train, y_train)


class TestColSampleRegressor:
    """Column subsampling via the regressor interface."""

    def test_regressor_col_sample(self, cal_split: tuple) -> None:
        """Regressor should accept and pass col_sample to engine."""
        X_train, X_test, y_train, _ = cal_split
        reg = LightningBoostRegressor(
            n_estimators=10,
            learning_rate=0.05,
            col_sample=0.5,
            random_state=SEED,
            verbose=False,
        )
        reg.fit(X_train, y_train)
        assert reg.engine_.col_sample == 0.5
        preds = reg.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_regressor_col_sample_default(self) -> None:
        """Default col_sample should be 1.0."""
        reg = LightningBoostRegressor()
        assert reg.col_sample == 1.0
