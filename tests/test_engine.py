"""Integration tests for the core boosting engine."""

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from ngboost_lightning.distributions import Normal
from ngboost_lightning.engine import NGBEngine
from tests._constants import SEED


@pytest.fixture()
def cal_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """California housing train/test split (small subset for speed)."""
    cal = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        cal.data, cal.target, test_size=0.2, random_state=SEED
    )
    # Use a small subset for fast tests
    return X_train[:500], X_test[:200], y_train[:500], y_test[:200]


# ---------- Smoke tests ----------


class TestSmoke:
    """Basic smoke tests: fit runs, outputs have correct shapes."""

    def test_fit_runs(self, cal_split: tuple) -> None:
        """Fit should complete without error."""
        X_train, _, y_train, _ = cal_split
        eng = NGBEngine(n_estimators=10, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        assert eng.n_estimators_ == 10

    def test_predict_shape(self, cal_split: tuple) -> None:
        """Predict should return [n_samples]."""
        X_train, X_test, y_train, _ = cal_split
        eng = NGBEngine(n_estimators=10, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        preds = eng.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_predict_params_shape(self, cal_split: tuple) -> None:
        """predict_params should return [n_samples, 2]."""
        X_train, X_test, y_train, _ = cal_split
        eng = NGBEngine(n_estimators=10, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        params = eng.predict_params(X_test)
        assert params.shape == (len(X_test), 2)

    def test_pred_dist_type(self, cal_split: tuple) -> None:
        """pred_dist should return a Normal instance."""
        X_train, X_test, y_train, _ = cal_split
        eng = NGBEngine(n_estimators=10, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        dist = eng.pred_dist(X_test)
        assert isinstance(dist, Normal)
        assert len(dist) == len(X_test)


# ---------- Training behavior ----------


class TestTrainingBehavior:
    """Tests for correct training dynamics."""

    def test_loss_decreases(self, cal_split: tuple) -> None:
        """Training loss should generally decrease over many iterations."""
        X_train, _, y_train, _ = cal_split
        eng = NGBEngine(n_estimators=100, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        # Loss in the last quarter should be lower than the first quarter
        n = len(eng.train_loss_)
        q1_mean = np.mean(eng.train_loss_[: n // 4])
        q4_mean = np.mean(eng.train_loss_[3 * n // 4 :])
        assert q4_mean < q1_mean

    def test_scalings_stored(self, cal_split: tuple) -> None:
        """Scalings should be stored for each iteration."""
        X_train, _, y_train, _ = cal_split
        eng = NGBEngine(n_estimators=10, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        assert len(eng.scalings_) == eng.n_estimators_
        assert all(s > 0 for s in eng.scalings_)

    def test_different_boosters(self, cal_split: tuple) -> None:
        """The two boosters should learn different tree structures."""
        X_train, X_test, y_train, _ = cal_split
        eng = NGBEngine(n_estimators=20, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        # Predictions from the two boosters should differ
        pred_0 = eng.boosters_[0].predict(X_test, raw_score=True)
        pred_1 = eng.boosters_[1].predict(X_test, raw_score=True)
        assert not np.allclose(pred_0, pred_1)


# ---------- Early stopping ----------


class TestEarlyStopping:
    """Tests for early stopping behavior."""

    def test_early_stopping(self, cal_split: tuple) -> None:
        """With early stopping, n_estimators_ should be < n_estimators."""
        X_train, X_test, y_train, y_test = cal_split
        eng = NGBEngine(n_estimators=500, learning_rate=0.05, verbose=False)
        eng.fit(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            early_stopping_rounds=10,
        )
        assert eng.n_estimators_ < 500

    def test_early_stopping_best_iteration(self, cal_split: tuple) -> None:
        """best_val_loss_itr_ should be set when early stopping is used."""
        X_train, X_test, y_train, y_test = cal_split
        eng = NGBEngine(n_estimators=500, learning_rate=0.05, verbose=False)
        eng.fit(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            early_stopping_rounds=10,
        )
        assert eng.best_val_loss_itr_ is not None
        assert eng.best_val_loss_itr_ >= 0


# ---------- Minibatch ----------


class TestMinibatch:
    """Tests for minibatch subsampling."""

    def test_minibatch_runs(self, cal_split: tuple) -> None:
        """Fit with minibatch_frac=0.5 should complete."""
        X_train, _, y_train, _ = cal_split
        eng = NGBEngine(
            n_estimators=10,
            learning_rate=0.05,
            minibatch_frac=0.5,
            verbose=False,
        )
        eng.fit(X_train, y_train)
        assert eng.n_estimators_ == 10

    def test_minibatch_deterministic(self, cal_split: tuple) -> None:
        """Same random_state should give identical results."""
        X_train, X_test, y_train, _ = cal_split
        eng1 = NGBEngine(
            n_estimators=10,
            learning_rate=0.05,
            minibatch_frac=0.5,
            random_state=123,
            verbose=False,
        )
        eng1.fit(X_train, y_train)

        eng2 = NGBEngine(
            n_estimators=10,
            learning_rate=0.05,
            minibatch_frac=0.5,
            random_state=123,
            verbose=False,
        )
        eng2.fit(X_train, y_train)

        np.testing.assert_allclose(
            eng1.predict(X_test), eng2.predict(X_test), rtol=1e-10
        )


# ---------- Natural gradient toggle ----------


class TestNaturalGradient:
    """Tests for natural_gradient on/off."""

    def test_natural_gradient_off_runs(self, cal_split: tuple) -> None:
        """natural_gradient=False should run without error."""
        X_train, _, y_train, _ = cal_split
        eng = NGBEngine(
            n_estimators=10,
            learning_rate=0.05,
            natural_gradient=False,
            verbose=False,
        )
        eng.fit(X_train, y_train)
        assert eng.n_estimators_ == 10

    def test_natural_gradient_on_off_same_for_normal(self, cal_split: tuple) -> None:
        """For Normal, natural and ordinary gradient should give same results.

        The Fisher Information for Normal(mean, log_scale) is diagonal, so
        natural gradient = FI^{-1} @ grad is just element-wise rescaling.
        However, the actual tree fits differ because the gradient TARGETS
        differ (nat_grad vs d_score have different magnitudes). So we check
        that both converge to similar final losses, not exact equality.
        """
        X_train, _, y_train, _ = cal_split
        eng_nat = NGBEngine(
            n_estimators=50,
            learning_rate=0.05,
            natural_gradient=True,
            random_state=SEED,
            verbose=False,
        )
        eng_nat.fit(X_train, y_train)

        eng_ord = NGBEngine(
            n_estimators=50,
            learning_rate=0.05,
            natural_gradient=False,
            random_state=SEED,
            verbose=False,
        )
        eng_ord.fit(X_train, y_train)

        # Both should achieve similar final training loss
        assert eng_nat.train_loss_[-1] == pytest.approx(
            eng_ord.train_loss_[-1], rel=0.2
        )


# ---------- Edge cases ----------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_iteration(self, cal_split: tuple) -> None:
        """n_estimators=1 should work."""
        X_train, X_test, y_train, _ = cal_split
        eng = NGBEngine(n_estimators=1, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        assert eng.n_estimators_ == 1
        preds = eng.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_small_dataset(self) -> None:
        """Should work on a very small dataset."""
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(20, 3))
        y = rng.normal(size=20)
        eng = NGBEngine(
            n_estimators=5,
            learning_rate=0.05,
            verbose=False,
            lgbm_params={"num_leaves": 4, "min_child_samples": 1},
        )
        eng.fit(X, y)
        preds = eng.predict(X)
        assert preds.shape == (20,)


# ---------- Prediction correctness ----------


class TestPredictionCorrectness:
    """Tests for inference/prediction consistency."""

    def test_predict_finite(self, cal_split: tuple) -> None:
        """All predictions should be finite."""
        X_train, X_test, y_train, _ = cal_split
        eng = NGBEngine(n_estimators=20, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        preds = eng.predict(X_test)
        assert np.all(np.isfinite(preds))

    def test_pred_dist_scale_positive(self, cal_split: tuple) -> None:
        """Predicted scale should always be positive."""
        X_train, X_test, y_train, _ = cal_split
        eng = NGBEngine(n_estimators=20, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        dist = eng.pred_dist(X_test)
        assert np.all(dist.scale > 0)

    def test_n_features_stored(self, cal_split: tuple) -> None:
        """n_features_in_ should be set after fit."""
        X_train, _, y_train, _ = cal_split
        eng = NGBEngine(n_estimators=5, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        assert eng.n_features_in_ == X_train.shape[1]
