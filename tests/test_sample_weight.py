"""Tests for sample_weight support across the stack."""

import numpy as np
import pytest

from ngboost_lightning import LightningBoostClassifier
from ngboost_lightning import LightningBoostRegressor
from ngboost_lightning.distributions import Exponential
from ngboost_lightning.distributions import Gamma
from ngboost_lightning.distributions import LogNormal
from ngboost_lightning.distributions import Normal
from ngboost_lightning.distributions import Poisson
from ngboost_lightning.distributions.categorical import Bernoulli
from ngboost_lightning.engine import NGBEngine
from ngboost_lightning.scoring import CRPScore
from ngboost_lightning.scoring import LogScore
from tests._constants import SEED

# ===================================================================
# Fixtures
# ===================================================================

RNG = np.random.default_rng(SEED)


@pytest.fixture()
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Simple regression dataset."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(200, 5))
    y = 3 * X[:, 0] + rng.normal(scale=0.5, size=200)
    return X, y


@pytest.fixture()
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Simple binary classification dataset."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + X[:, 1] + rng.normal(scale=0.3, size=200) > 0).astype(float)
    return X, y


# ===================================================================
# Distribution.fit weighted tests
# ===================================================================


class TestWeightedDistFit:
    """Weighted Distribution.fit produces correct initial parameters."""

    def test_normal_weighted_fit(self) -> None:
        """Normal.fit with weights shifts mean toward high-weight samples."""
        y = np.array([0.0, 0.0, 0.0, 10.0])
        # Uniform: mean ~2.5
        unweighted = Normal.fit(y)
        # Heavy weight on last sample: mean should be closer to 10
        w = np.array([1.0, 1.0, 1.0, 100.0])
        weighted = Normal.fit(y, sample_weight=w)
        assert weighted[0] > unweighted[0]
        assert weighted[0] > 8.0  # should be close to 10

    def test_exponential_weighted_fit(self) -> None:
        """Exponential.fit with weights adjusts rate."""
        y = np.array([1.0, 1.0, 1.0, 100.0])
        w = np.array([1.0, 1.0, 1.0, 100.0])
        # Heavy weight on large value → lower rate
        unweighted = Exponential.fit(y)
        weighted = Exponential.fit(y, sample_weight=w)
        # log(rate) should be smaller (lower rate = larger mean)
        assert weighted[0] < unweighted[0]

    def test_poisson_weighted_fit(self) -> None:
        """Poisson.fit with weights shifts rate."""
        y = np.array([1.0, 1.0, 1.0, 50.0])
        w = np.array([1.0, 1.0, 1.0, 100.0])
        unweighted = Poisson.fit(y)
        weighted = Poisson.fit(y, sample_weight=w)
        assert weighted[0] > unweighted[0]  # higher log_rate

    def test_lognormal_weighted_fit(self) -> None:
        """LogNormal.fit with weights shifts mu."""
        y = np.array([1.0, 1.0, 1.0, 100.0])
        w = np.array([1.0, 1.0, 1.0, 100.0])
        unweighted = LogNormal.fit(y)
        weighted = LogNormal.fit(y, sample_weight=w)
        assert weighted[0] > unweighted[0]

    def test_gamma_weighted_fit(self) -> None:
        """Gamma.fit with weights adjusts shape/rate."""
        y = np.array([1.0, 1.0, 1.0, 100.0])
        w = np.array([1.0, 1.0, 1.0, 100.0])
        unweighted = Gamma.fit(y)
        weighted = Gamma.fit(y, sample_weight=w)
        # Weighted mean is much larger → different parameters
        assert not np.allclose(unweighted, weighted)

    def test_categorical_weighted_fit(self) -> None:
        """Categorical.fit with weights shifts class probabilities."""
        y = np.array([0.0, 0.0, 0.0, 1.0])
        # Unweighted: p(1) = 0.25
        unweighted = Bernoulli.fit(y)
        # Heavy weight on class 1: p(1) should be much higher
        w = np.array([1.0, 1.0, 1.0, 100.0])
        weighted = Bernoulli.fit(y, sample_weight=w)
        # logit = log(p1/p0), so higher logit = higher p(1)
        assert weighted[0] > unweighted[0]

    def test_uniform_weights_match_unweighted(self) -> None:
        """Uniform weights produce same result as no weights."""
        rng = np.random.default_rng(42)
        y = rng.normal(3.0, 1.0, size=100)
        w = np.ones(100)
        np.testing.assert_allclose(
            Normal.fit(y), Normal.fit(y, sample_weight=w), rtol=1e-10
        )


# ===================================================================
# ScoringRule.total_score weighted tests
# ===================================================================


class TestWeightedTotalScore:
    """Weighted total_score produces correct weighted averages."""

    def test_logscore_weighted(self) -> None:
        """LogScore.total_score with weights differs from unweighted."""
        rng = np.random.default_rng(42)
        params = np.column_stack([rng.normal(size=50), np.zeros(50)])
        dist = Normal(params)
        y = rng.normal(size=50)
        w = rng.uniform(0.1, 10.0, size=50)

        ls = LogScore()
        unweighted = ls.total_score(dist, y)
        weighted = ls.total_score(dist, y, sample_weight=w)
        assert unweighted != weighted

    def test_crpscore_weighted(self) -> None:
        """CRPScore.total_score with weights differs from unweighted."""
        rng = np.random.default_rng(42)
        params = np.column_stack([rng.normal(size=50), np.zeros(50)])
        dist = Normal(params)
        y = rng.normal(size=50)
        w = rng.uniform(0.1, 10.0, size=50)

        cs = CRPScore()
        unweighted = cs.total_score(dist, y)
        weighted = cs.total_score(dist, y, sample_weight=w)
        assert unweighted != weighted

    def test_uniform_weights_match_unweighted(self) -> None:
        """Uniform weights give same total_score as no weights."""
        rng = np.random.default_rng(42)
        params = np.column_stack([rng.normal(size=50), np.zeros(50)])
        dist = Normal(params)
        y = rng.normal(size=50)
        w = np.ones(50)

        ls = LogScore()
        np.testing.assert_allclose(
            ls.total_score(dist, y),
            ls.total_score(dist, y, sample_weight=w),
            rtol=1e-10,
        )


# ===================================================================
# Engine-level tests
# ===================================================================


class TestEngineWeights:
    """Tests for sample weights in NGBEngine."""

    def test_weighted_fit_runs(self, regression_data: tuple) -> None:
        """Engine fit with sample_weight completes without error."""
        X, y = regression_data
        w = np.ones(len(y))
        eng = NGBEngine(n_estimators=10, learning_rate=0.05, verbose=False)
        eng.fit(X, y, sample_weight=w)
        assert eng.n_estimators_ == 10

    def test_weighted_changes_result(self, regression_data: tuple) -> None:
        """Weights that emphasize different samples produce different models."""
        X, y = regression_data
        eng1 = NGBEngine(
            n_estimators=20, learning_rate=0.05, verbose=False, random_state=SEED
        )
        eng1.fit(X, y)

        # Heavy weight on first half
        w = np.ones(len(y))
        w[: len(y) // 2] = 10.0
        eng2 = NGBEngine(
            n_estimators=20, learning_rate=0.05, verbose=False, random_state=SEED
        )
        eng2.fit(X, y, sample_weight=w)

        # Predictions should differ
        preds1 = eng1.predict(X[:10])
        preds2 = eng2.predict(X[:10])
        assert not np.allclose(preds1, preds2, atol=1e-4)

    def test_uniform_weights_match_unweighted(self, regression_data: tuple) -> None:
        """Uniform weights produce identical results to no weights."""
        X, y = regression_data
        eng1 = NGBEngine(
            n_estimators=10, learning_rate=0.05, verbose=False, random_state=SEED
        )
        eng1.fit(X, y)

        w = np.ones(len(y))
        eng2 = NGBEngine(
            n_estimators=10, learning_rate=0.05, verbose=False, random_state=SEED
        )
        eng2.fit(X, y, sample_weight=w)

        preds1 = eng1.predict(X[:10])
        preds2 = eng2.predict(X[:10])
        np.testing.assert_allclose(preds1, preds2, rtol=1e-6)

    def test_zero_weight_samples_ignored(self) -> None:
        """Zero-weight samples have no effect, equivalent to removal."""
        rng = np.random.default_rng(SEED)
        X_real = rng.normal(size=(100, 3))
        y_real = 2 * X_real[:, 0] + rng.normal(scale=0.3, size=100)

        # Add junk rows with zero weight
        X_junk = rng.normal(loc=100, size=(50, 3))
        y_junk = rng.normal(loc=999, size=50)

        X_combined = np.vstack([X_real, X_junk])
        y_combined = np.concatenate([y_real, y_junk])
        w = np.concatenate([np.ones(100), np.zeros(50)])

        # Fit on real data only
        eng1 = NGBEngine(
            n_estimators=15, learning_rate=0.05, verbose=False, random_state=SEED
        )
        eng1.fit(X_real, y_real)

        # Fit on combined data with zero weights on junk
        eng2 = NGBEngine(
            n_estimators=15, learning_rate=0.05, verbose=False, random_state=SEED
        )
        eng2.fit(X_combined, y_combined, sample_weight=w)

        # Init params should match (zero-weight samples don't affect
        # weighted mean/variance in fit)
        np.testing.assert_allclose(eng1.init_params_, eng2.init_params_, rtol=1e-6)

    def test_val_weight_mismatch_raises(self, regression_data: tuple) -> None:
        """Providing sample_weight without val_sample_weight raises."""
        X, y = regression_data
        n = len(y)
        eng = NGBEngine(n_estimators=5, verbose=False)
        with pytest.raises(ValueError, match="val_sample_weight"):
            eng.fit(
                X,
                y,
                X_val=X[:20],
                y_val=y[:20],
                sample_weight=np.ones(n),
            )

    def test_val_weight_reverse_mismatch_raises(self, regression_data: tuple) -> None:
        """Providing val_sample_weight without sample_weight raises."""
        X, y = regression_data
        eng = NGBEngine(n_estimators=5, verbose=False)
        with pytest.raises(ValueError, match="sample_weight"):
            eng.fit(
                X,
                y,
                X_val=X[:20],
                y_val=y[:20],
                val_sample_weight=np.ones(20),
            )

    def test_weighted_validation_loss(self, regression_data: tuple) -> None:
        """Validation loss uses val_sample_weight."""
        X, y = regression_data
        n = len(y)
        w_train = np.ones(n)
        w_val = np.ones(20)
        w_val[:10] = 10.0  # heavy weight on first 10 val samples

        eng = NGBEngine(n_estimators=10, learning_rate=0.05, verbose=False)
        eng.fit(
            X,
            y,
            X_val=X[:20],
            y_val=y[:20],
            sample_weight=w_train,
            val_sample_weight=w_val,
        )
        assert len(eng.val_loss_) == 10


# ===================================================================
# Regressor-level tests
# ===================================================================


class TestRegressorWeights:
    """Tests for sample weights on LightningBoostRegressor."""

    def test_regressor_weighted_fit(self, regression_data: tuple) -> None:
        """Regressor accepts and passes through sample_weight."""
        X, y = regression_data
        reg = LightningBoostRegressor(
            n_estimators=10, learning_rate=0.05, verbose=False
        )
        w = np.ones(len(y))
        w[:50] = 5.0
        reg.fit(X, y, sample_weight=w)
        assert reg.n_estimators_ == 10

    def test_regressor_weighted_early_stopping(self, regression_data: tuple) -> None:
        """Regressor passes weights through to validation correctly."""
        X, y = regression_data
        n = len(y)
        reg = LightningBoostRegressor(
            n_estimators=100, learning_rate=0.05, verbose=False
        )
        w_train = np.ones(n - 40)
        w_val = np.ones(40)
        reg.fit(
            X[: n - 40],
            y[: n - 40],
            X_val=X[n - 40 :],
            y_val=y[n - 40 :],
            early_stopping_rounds=5,
            sample_weight=w_train,
            val_sample_weight=w_val,
        )
        assert hasattr(reg, "val_loss_")


# ===================================================================
# Classifier-level tests
# ===================================================================


class TestClassifierWeights:
    """Tests for sample weights on LightningBoostClassifier."""

    def test_classifier_weighted_fit(self, classification_data: tuple) -> None:
        """Classifier accepts and passes through sample_weight."""
        X, y = classification_data
        clf = LightningBoostClassifier(
            n_estimators=10, learning_rate=0.05, verbose=False
        )
        w = np.ones(len(y))
        w[y == 1] = 5.0  # upweight positive class
        clf.fit(X, y, sample_weight=w)
        assert clf.n_estimators_ == 10

    def test_classifier_weights_shift_probabilities(
        self, classification_data: tuple
    ) -> None:
        """Heavy weights on one class shift predicted probabilities."""
        X, y = classification_data
        clf1 = LightningBoostClassifier(
            n_estimators=20, learning_rate=0.05, verbose=False, random_state=SEED
        )
        clf1.fit(X, y)
        probs1 = clf1.predict_proba(X)[:, 1].mean()

        # Upweight class 1 heavily
        w = np.ones(len(y))
        w[y == 1] = 20.0
        clf2 = LightningBoostClassifier(
            n_estimators=20, learning_rate=0.05, verbose=False, random_state=SEED
        )
        clf2.fit(X, y, sample_weight=w)
        probs2 = clf2.predict_proba(X)[:, 1].mean()

        # Upweighting class 1 should increase average predicted P(1)
        assert probs2 > probs1


# ===================================================================
# Minibatch + weights
# ===================================================================


class TestMinibatchWeights:
    """Test that minibatch correctly slices weights."""

    def test_minibatch_with_weights_runs(self, regression_data: tuple) -> None:
        """Minibatch with sample_weight completes without error."""
        X, y = regression_data
        w = np.ones(len(y))
        w[:50] = 5.0
        eng = NGBEngine(
            n_estimators=10,
            learning_rate=0.05,
            minibatch_frac=0.5,
            verbose=False,
            random_state=SEED,
        )
        eng.fit(X, y, sample_weight=w)
        assert eng.n_estimators_ == 10
