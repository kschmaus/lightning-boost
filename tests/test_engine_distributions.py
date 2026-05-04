"""Engine integration tests for non-Normal distributions.

Verifies that the NGBEngine training loop works end-to-end with each
new distribution, including 1-parameter distributions (Exponential,
Poisson) and the non-diagonal Fisher (Gamma).
"""

import numpy as np
import pytest

from ngboost_lightning.distributions import Exponential
from ngboost_lightning.distributions import Gamma
from ngboost_lightning.distributions import LogNormal
from ngboost_lightning.distributions import Poisson
from ngboost_lightning.distributions.categorical import Bernoulli
from ngboost_lightning.distributions.categorical import Categorical
from ngboost_lightning.distributions.categorical import k_categorical
from ngboost_lightning.engine import NGBEngine
from tests._constants import SEED

# ---------------------------------------------------------------------------
# Fixtures — synthetic data that suits each distribution family
# ---------------------------------------------------------------------------


@pytest.fixture()
def positive_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic positive-valued regression data for LogNormal/Exponential/Gamma."""
    rng = np.random.default_rng(SEED)
    n_train, n_test, n_features = 400, 100, 5
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    # y = exp(linear combination + noise) → strictly positive
    coef = rng.normal(size=n_features) * 0.3
    y_train = np.exp(X_train @ coef + rng.normal(scale=0.3, size=n_train))
    y_test = np.exp(X_test @ coef + rng.normal(scale=0.3, size=n_test))
    return X_train, X_test, y_train, y_test


@pytest.fixture()
def count_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic count data for Poisson."""
    rng = np.random.default_rng(SEED)
    n_train, n_test, n_features = 400, 100, 5
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    coef = rng.normal(size=n_features) * 0.3
    rate_train = np.exp(X_train @ coef + 1.0)  # rates around e^1 ≈ 2.7
    rate_test = np.exp(X_test @ coef + 1.0)
    y_train = rng.poisson(rate_train).astype(np.float64)
    y_test = rng.poisson(rate_test).astype(np.float64)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Two-parameter distributions: LogNormal, Gamma
# ---------------------------------------------------------------------------


class TestLogNormalEngine:
    """Engine integration tests for LogNormal (n_params=2)."""

    def test_fit_runs(self, positive_data: tuple) -> None:
        """Fit should complete without error."""
        X_train, _, y_train, _ = positive_data
        eng = NGBEngine(
            dist=LogNormal, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        assert eng.n_estimators_ == 20

    def test_predict_shape(self, positive_data: tuple) -> None:
        """Predict should return [n_samples]."""
        X_train, X_test, y_train, _ = positive_data
        eng = NGBEngine(
            dist=LogNormal, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        assert eng.predict(X_test).shape == (len(X_test),)

    def test_predict_params_shape(self, positive_data: tuple) -> None:
        """predict_params should return [n_samples, 2]."""
        X_train, X_test, y_train, _ = positive_data
        eng = NGBEngine(
            dist=LogNormal, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        assert eng.predict_params(X_test).shape == (len(X_test), 2)

    def test_pred_dist_type(self, positive_data: tuple) -> None:
        """pred_dist should return a LogNormal instance."""
        X_train, X_test, y_train, _ = positive_data
        eng = NGBEngine(
            dist=LogNormal, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        dist = eng.pred_dist(X_test)
        assert isinstance(dist, LogNormal)
        assert len(dist) == len(X_test)

    def test_loss_decreases(self, positive_data: tuple) -> None:
        """Training loss should decrease over iterations."""
        X_train, _, y_train, _ = positive_data
        eng = NGBEngine(
            dist=LogNormal, n_estimators=100, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        n = len(eng.train_loss_)
        q1 = np.mean(eng.train_loss_[: n // 4])
        q4 = np.mean(eng.train_loss_[3 * n // 4 :])
        assert q4 < q1

    def test_predictions_positive(self, positive_data: tuple) -> None:
        """All predictions should be positive and finite."""
        X_train, X_test, y_train, _ = positive_data
        eng = NGBEngine(
            dist=LogNormal, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        preds = eng.predict(X_test)
        assert np.all(preds > 0)
        assert np.all(np.isfinite(preds))


class TestGammaEngine:
    """Engine integration tests for Gamma (n_params=2, non-diagonal Fisher)."""

    def test_fit_runs(self, positive_data: tuple) -> None:
        """Fit should complete without error."""
        X_train, _, y_train, _ = positive_data
        eng = NGBEngine(dist=Gamma, n_estimators=20, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        assert eng.n_estimators_ == 20

    def test_predict_shape(self, positive_data: tuple) -> None:
        """Predict should return [n_samples]."""
        X_train, X_test, y_train, _ = positive_data
        eng = NGBEngine(dist=Gamma, n_estimators=20, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        assert eng.predict(X_test).shape == (len(X_test),)

    def test_predict_params_shape(self, positive_data: tuple) -> None:
        """predict_params should return [n_samples, 2]."""
        X_train, X_test, y_train, _ = positive_data
        eng = NGBEngine(dist=Gamma, n_estimators=20, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        assert eng.predict_params(X_test).shape == (len(X_test), 2)

    def test_pred_dist_type(self, positive_data: tuple) -> None:
        """pred_dist should return a Gamma instance."""
        X_train, X_test, y_train, _ = positive_data
        eng = NGBEngine(dist=Gamma, n_estimators=20, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        dist = eng.pred_dist(X_test)
        assert isinstance(dist, Gamma)
        assert len(dist) == len(X_test)

    def test_loss_decreases(self, positive_data: tuple) -> None:
        """Training loss should decrease over iterations."""
        X_train, _, y_train, _ = positive_data
        eng = NGBEngine(dist=Gamma, n_estimators=100, learning_rate=0.05, verbose=False)
        eng.fit(X_train, y_train)
        n = len(eng.train_loss_)
        q1 = np.mean(eng.train_loss_[: n // 4])
        q4 = np.mean(eng.train_loss_[3 * n // 4 :])
        assert q4 < q1

    def test_natural_gradient_uses_non_diagonal_fi(self, positive_data: tuple) -> None:
        """Gamma's non-diagonal Fisher should still allow convergence."""
        X_train, X_test, y_train, _ = positive_data
        eng = NGBEngine(
            dist=Gamma,
            n_estimators=50,
            learning_rate=0.05,
            natural_gradient=True,
            random_state=SEED,
            verbose=False,
        )
        eng.fit(X_train, y_train)
        preds = eng.predict(X_test)
        assert np.all(np.isfinite(preds))
        assert np.all(preds > 0)


# ---------------------------------------------------------------------------
# One-parameter distributions: Exponential, Poisson
# ---------------------------------------------------------------------------


class TestExponentialEngine:
    """Engine integration tests for Exponential (n_params=1)."""

    def test_fit_runs(self, positive_data: tuple) -> None:
        """Fit should complete without error."""
        X_train, _, y_train, _ = positive_data
        eng = NGBEngine(
            dist=Exponential, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        assert eng.n_estimators_ == 20

    def test_predict_shape(self, positive_data: tuple) -> None:
        """Predict should return [n_samples]."""
        X_train, X_test, y_train, _ = positive_data
        eng = NGBEngine(
            dist=Exponential, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        assert eng.predict(X_test).shape == (len(X_test),)

    def test_predict_params_shape(self, positive_data: tuple) -> None:
        """n_params=1: predict_params returns [n_samples, 1]."""
        X_train, X_test, y_train, _ = positive_data
        eng = NGBEngine(
            dist=Exponential, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        assert eng.predict_params(X_test).shape == (len(X_test), 1)

    def test_pred_dist_type(self, positive_data: tuple) -> None:
        """pred_dist should return an Exponential instance."""
        X_train, X_test, y_train, _ = positive_data
        eng = NGBEngine(
            dist=Exponential, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        dist = eng.pred_dist(X_test)
        assert isinstance(dist, Exponential)
        assert len(dist) == len(X_test)

    def test_loss_decreases(self, positive_data: tuple) -> None:
        """Training loss should decrease over iterations."""
        X_train, _, y_train, _ = positive_data
        eng = NGBEngine(
            dist=Exponential, n_estimators=100, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        n = len(eng.train_loss_)
        q1 = np.mean(eng.train_loss_[: n // 4])
        q4 = np.mean(eng.train_loss_[3 * n // 4 :])
        assert q4 < q1

    def test_predictions_positive(self, positive_data: tuple) -> None:
        """All predictions should be positive and finite."""
        X_train, X_test, y_train, _ = positive_data
        eng = NGBEngine(
            dist=Exponential, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        preds = eng.predict(X_test)
        assert np.all(preds > 0)
        assert np.all(np.isfinite(preds))

    def test_early_stopping(self, positive_data: tuple) -> None:
        """Early stopping should trigger before max iterations."""
        X_train, X_test, y_train, y_test = positive_data
        eng = NGBEngine(
            dist=Exponential, n_estimators=500, learning_rate=0.05, verbose=False
        )
        eng.fit(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            early_stopping_rounds=10,
        )
        assert eng.n_estimators_ < 500


class TestPoissonEngine:
    """Engine integration tests for Poisson (n_params=1, discrete)."""

    def test_fit_runs(self, count_data: tuple) -> None:
        """Fit should complete without error."""
        X_train, _, y_train, _ = count_data
        eng = NGBEngine(
            dist=Poisson, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        assert eng.n_estimators_ == 20

    def test_predict_shape(self, count_data: tuple) -> None:
        """Predict should return [n_samples]."""
        X_train, X_test, y_train, _ = count_data
        eng = NGBEngine(
            dist=Poisson, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        assert eng.predict(X_test).shape == (len(X_test),)

    def test_predict_params_shape(self, count_data: tuple) -> None:
        """n_params=1: predict_params returns [n_samples, 1]."""
        X_train, X_test, y_train, _ = count_data
        eng = NGBEngine(
            dist=Poisson, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        assert eng.predict_params(X_test).shape == (len(X_test), 1)

    def test_pred_dist_type(self, count_data: tuple) -> None:
        """pred_dist should return a Poisson instance."""
        X_train, X_test, y_train, _ = count_data
        eng = NGBEngine(
            dist=Poisson, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        dist = eng.pred_dist(X_test)
        assert isinstance(dist, Poisson)
        assert len(dist) == len(X_test)

    def test_loss_decreases(self, count_data: tuple) -> None:
        """Training loss should decrease over iterations."""
        X_train, _, y_train, _ = count_data
        eng = NGBEngine(
            dist=Poisson, n_estimators=100, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        n = len(eng.train_loss_)
        q1 = np.mean(eng.train_loss_[: n // 4])
        q4 = np.mean(eng.train_loss_[3 * n // 4 :])
        assert q4 < q1

    def test_predictions_positive(self, count_data: tuple) -> None:
        """All predictions should be positive and finite."""
        X_train, X_test, y_train, _ = count_data
        eng = NGBEngine(
            dist=Poisson, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        preds = eng.predict(X_test)
        assert np.all(preds > 0)
        assert np.all(np.isfinite(preds))

    def test_early_stopping(self, count_data: tuple) -> None:
        """Early stopping should trigger before max iterations."""
        X_train, X_test, y_train, y_test = count_data
        eng = NGBEngine(
            dist=Poisson, n_estimators=500, learning_rate=0.05, verbose=False
        )
        eng.fit(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            early_stopping_rounds=10,
        )
        assert eng.n_estimators_ < 500


# ---------------------------------------------------------------------------
# Cross-distribution test: minibatch with 1-param dist
# ---------------------------------------------------------------------------


class TestMinibatchOneParam:
    """Minibatch subsampling with 1-parameter distributions."""

    def test_exponential_minibatch(self, positive_data: tuple) -> None:
        """Minibatch fit with Exponential should complete."""
        X_train, _, y_train, _ = positive_data
        eng = NGBEngine(
            dist=Exponential,
            n_estimators=20,
            learning_rate=0.05,
            minibatch_frac=0.5,
            random_state=SEED,
            verbose=False,
        )
        eng.fit(X_train, y_train)
        assert eng.n_estimators_ == 20

    def test_poisson_minibatch(self, count_data: tuple) -> None:
        """Minibatch fit with Poisson should complete."""
        X_train, _, y_train, _ = count_data
        eng = NGBEngine(
            dist=Poisson,
            n_estimators=20,
            learning_rate=0.05,
            minibatch_frac=0.5,
            random_state=SEED,
            verbose=False,
        )
        eng.fit(X_train, y_train)
        assert eng.n_estimators_ == 20


# ---------------------------------------------------------------------------
# Classification distributions: Bernoulli, Categorical
# ---------------------------------------------------------------------------


@pytest.fixture()
def binary_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic binary classification data."""
    rng = np.random.default_rng(SEED)
    n_train, n_test, n_features = 400, 100, 5
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    coef = rng.normal(size=n_features) * 0.5
    p_train = 1 / (1 + np.exp(-(X_train @ coef)))
    y_train = rng.binomial(1, p_train).astype(np.float64)
    p_test = 1 / (1 + np.exp(-(X_test @ coef)))
    y_test = rng.binomial(1, p_test).astype(np.float64)
    return X_train, X_test, y_train, y_test


@pytest.fixture()
def multiclass_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic 3-class classification data."""
    rng = np.random.default_rng(SEED)
    n_train, n_test, n_features = 400, 100, 5
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    coef = rng.normal(size=(n_features, 3))
    y_train = np.argmax(X_train @ coef, axis=1).astype(np.float64)
    y_test = np.argmax(X_test @ coef, axis=1).astype(np.float64)
    return X_train, X_test, y_train, y_test


class TestBernoulliEngine:
    """Engine integration tests for Bernoulli (binary, n_params=1)."""

    def test_fit_runs(self, binary_data: tuple) -> None:
        """Fit with Bernoulli should complete without error."""
        X_train, _, y_train, _ = binary_data
        eng = NGBEngine(
            dist=Bernoulli, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        assert eng.n_estimators_ == 20

    def test_predict_shape(self, binary_data: tuple) -> None:
        """Predict returns [n_samples, K] (class probabilities via mean)."""
        X_train, X_test, y_train, _ = binary_data
        eng = NGBEngine(
            dist=Bernoulli, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        assert eng.predict(X_test).shape == (len(X_test), 2)

    def test_predict_params_shape(self, binary_data: tuple) -> None:
        """n_params=1: predict_params returns [n_samples, 1]."""
        X_train, X_test, y_train, _ = binary_data
        eng = NGBEngine(
            dist=Bernoulli, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        assert eng.predict_params(X_test).shape == (len(X_test), 1)

    def test_pred_dist_type(self, binary_data: tuple) -> None:
        """pred_dist should return a Categorical instance with correct length."""
        X_train, X_test, y_train, _ = binary_data
        eng = NGBEngine(
            dist=Bernoulli, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        dist = eng.pred_dist(X_test)
        assert isinstance(dist, Categorical)
        assert len(dist) == len(X_test)

    def test_loss_decreases(self, binary_data: tuple) -> None:
        """Training loss should decrease over iterations."""
        X_train, _, y_train, _ = binary_data
        eng = NGBEngine(
            dist=Bernoulli, n_estimators=100, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        n = len(eng.train_loss_)
        q1 = np.mean(eng.train_loss_[: n // 4])
        q4 = np.mean(eng.train_loss_[3 * n // 4 :])
        assert q4 < q1

    def test_predictions_finite(self, binary_data: tuple) -> None:
        """All predictions should be finite."""
        X_train, X_test, y_train, _ = binary_data
        eng = NGBEngine(
            dist=Bernoulli, n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        preds = eng.predict(X_test)
        assert np.all(np.isfinite(preds))


class TestCategoricalEngine:
    """Engine integration tests for Categorical (K=3, n_params=2)."""

    def test_fit_runs(self, multiclass_data: tuple) -> None:
        """Fit with k_categorical(3) should complete without error."""
        X_train, _, y_train, _ = multiclass_data
        eng = NGBEngine(
            dist=k_categorical(3), n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        assert eng.n_estimators_ == 20

    def test_predict_params_shape(self, multiclass_data: tuple) -> None:
        """K=3: predict_params returns [n_samples, 2]."""
        X_train, X_test, y_train, _ = multiclass_data
        eng = NGBEngine(
            dist=k_categorical(3), n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        assert eng.predict_params(X_test).shape == (len(X_test), 2)

    def test_pred_dist_type(self, multiclass_data: tuple) -> None:
        """pred_dist should return a Categorical instance."""
        X_train, X_test, y_train, _ = multiclass_data
        eng = NGBEngine(
            dist=k_categorical(3), n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        dist = eng.pred_dist(X_test)
        assert isinstance(dist, Categorical)
        assert len(dist) == len(X_test)

    def test_loss_decreases(self, multiclass_data: tuple) -> None:
        """Training loss should decrease over iterations."""
        X_train, _, y_train, _ = multiclass_data
        eng = NGBEngine(
            dist=k_categorical(3), n_estimators=100, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        n = len(eng.train_loss_)
        q1 = np.mean(eng.train_loss_[: n // 4])
        q4 = np.mean(eng.train_loss_[3 * n // 4 :])
        assert q4 < q1

    def test_predictions_finite(self, multiclass_data: tuple) -> None:
        """All predictions should be finite."""
        X_train, X_test, y_train, _ = multiclass_data
        eng = NGBEngine(
            dist=k_categorical(3), n_estimators=20, learning_rate=0.05, verbose=False
        )
        eng.fit(X_train, y_train)
        preds = eng.predict(X_test)
        assert np.all(np.isfinite(preds))
