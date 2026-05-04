"""Unit tests for survival analysis (right-censored data)."""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import weibull_min as sp_weibull

from ngboost_lightning.distributions.exponential import Exponential
from ngboost_lightning.distributions.lognormal import LogNormal
from ngboost_lightning.distributions.weibull import Weibull
from ngboost_lightning.survival import SURVIVAL_DTYPE
from ngboost_lightning.survival import CensoredLogScore
from ngboost_lightning.survival import Y_from_censored
from ngboost_lightning.survival import _is_censored_y
from ngboost_lightning.survival_estimator import LightningBoostSurvival

SEED = 42


class TestYFromCensored:
    """Tests for the Y_from_censored helper."""

    def test_creates_structured_array(self) -> None:
        """Should create a structured array with Event and Time fields."""
        T = np.array([1.0, 2.0, 3.0])
        E = np.array([1, 0, 1])
        Y = Y_from_censored(T, E)
        assert Y.dtype == SURVIVAL_DTYPE
        assert len(Y) == 3

    def test_event_field(self) -> None:
        """Event field should be boolean."""
        T = np.array([1.0, 2.0])
        E = np.array([1, 0])
        Y = Y_from_censored(T, E)
        assert Y["Event"].dtype == bool
        assert Y["Event"][0] is np.True_
        assert Y["Event"][1] is np.False_

    def test_time_field(self) -> None:
        """Time field should be float64."""
        T = np.array([1.5, 2.5])
        E = np.array([1, 1])
        Y = Y_from_censored(T, E)
        np.testing.assert_allclose(Y["Time"], [1.5, 2.5])

    def test_indexing(self) -> None:
        """Structured array should support fancy indexing."""
        T = np.array([1.0, 2.0, 3.0, 4.0])
        E = np.array([1, 0, 1, 0])
        Y = Y_from_censored(T, E)
        sub = Y[np.array([0, 2])]
        assert len(sub) == 2
        np.testing.assert_allclose(sub["Time"], [1.0, 3.0])
        assert sub["Event"][0] is np.True_
        assert sub["Event"][1] is np.True_


class TestIsCensoredY:
    """Tests for the _is_censored_y helper."""

    def test_structured_array(self) -> None:
        """Should return True for censored Y."""
        Y = Y_from_censored(np.array([1.0]), np.array([1]))
        assert _is_censored_y(Y) is True

    def test_plain_array(self) -> None:
        """Should return False for plain float arrays."""
        y = np.array([1.0, 2.0, 3.0])
        assert _is_censored_y(y) is False


class TestLogsf:
    """Tests for logsf on survival-eligible distributions."""

    def test_exponential_logsf(self) -> None:
        """Exponential.logsf should match log(1-cdf)."""
        params = np.array([[np.log(2.0)]])
        dist = Exponential(params)
        y = np.array([0.5])
        expected = np.log(1.0 - dist.cdf(y))
        np.testing.assert_allclose(dist.logsf(y), expected, rtol=1e-10)

    def test_lognormal_logsf(self) -> None:
        """LogNormal.logsf should match log(1-cdf)."""
        params = np.array([[0.0, 0.0]])
        dist = LogNormal(params)
        y = np.array([1.0])
        expected = np.log(1.0 - dist.cdf(y))
        np.testing.assert_allclose(dist.logsf(y), expected, rtol=1e-10)

    def test_weibull_logsf(self) -> None:
        """Weibull.logsf should match log(1-cdf)."""
        params = np.array([[np.log(2.0), np.log(1.0)]])
        dist = Weibull(params)
        y = np.array([0.5])
        expected = np.log(1.0 - dist.cdf(y))
        np.testing.assert_allclose(dist.logsf(y), expected, rtol=1e-10)


class TestCensoredLogScore:
    """Tests for the CensoredLogScore scoring rule."""

    def test_uncensored_matches_logscore(self) -> None:
        """All-observed data should match standard LogScore."""
        from ngboost_lightning.scoring import LogScore

        params = np.array([[0.0, 0.0], [0.5, -0.5]])
        dist = LogNormal(params)
        T = np.array([1.0, 2.0])
        E = np.array([1, 1])
        Y = Y_from_censored(T, E)

        censored_score = CensoredLogScore().score(dist, Y)
        uncensored_score = LogScore().score(dist, T)
        np.testing.assert_allclose(censored_score, uncensored_score, rtol=1e-10)

    def test_censored_uses_logsf(self) -> None:
        """All-censored data should use -logsf."""
        params = np.array([[0.0, 0.0], [0.5, -0.5]])
        dist = LogNormal(params)
        T = np.array([1.0, 2.0])
        E = np.array([0, 0])
        Y = Y_from_censored(T, E)

        censored_score = CensoredLogScore().score(dist, Y)
        expected = -dist.logsf(T)
        np.testing.assert_allclose(censored_score, expected, rtol=1e-10)

    def test_mixed_censoring(self) -> None:
        """Mixed censoring should blend logpdf and logsf correctly."""
        params = np.array([[0.0, 0.0], [0.5, -0.5]])
        dist = LogNormal(params)
        T = np.array([1.0, 2.0])
        E = np.array([1, 0])
        Y = Y_from_censored(T, E)

        scores = CensoredLogScore().score(dist, Y)
        expected_0 = -dist.logpdf(T)[0]
        expected_1 = -dist.logsf(T)[1]
        assert scores[0] == pytest.approx(expected_0, rel=1e-10)
        assert scores[1] == pytest.approx(expected_1, rel=1e-10)

    def test_gradient_shape(self) -> None:
        """Gradient should have shape [n_samples, n_params]."""
        params = np.array([[0.0, 0.0], [0.5, -0.5]])
        dist = LogNormal(params)
        Y = Y_from_censored(np.array([1.0, 2.0]), np.array([1, 0]))
        grad = CensoredLogScore().d_score(dist, Y)
        assert grad.shape == (2, 2)

    def test_gradient_finite_difference(self) -> None:
        """Gradient should match numerical finite differences."""
        params = np.array([[0.0, 0.0], [0.5, -0.5]])
        Y = Y_from_censored(np.array([1.0, 2.0]), np.array([1, 0]))
        scorer = CensoredLogScore()
        grad = scorer.d_score(LogNormal(params), Y)

        eps = 1e-5
        for k in range(2):
            pp = params.copy()
            pp[:, k] += eps
            pm = params.copy()
            pm[:, k] -= eps
            score_plus = scorer.score(LogNormal(pp), Y)
            score_minus = scorer.score(LogNormal(pm), Y)
            numerical = (score_plus - score_minus) / (2 * eps)
            np.testing.assert_allclose(grad[:, k], numerical, atol=1e-3)

    def test_total_score(self) -> None:
        """Total_score should return weighted average."""
        params = np.array([[0.0, 0.0], [0.5, -0.5]])
        dist = LogNormal(params)
        Y = Y_from_censored(np.array([1.0, 2.0]), np.array([1, 0]))
        scorer = CensoredLogScore()
        expected = float(np.mean(scorer.score(dist, Y)))
        assert scorer.total_score(dist, Y) == pytest.approx(expected)

    def test_censored_gradient_smaller(self) -> None:
        """Censored observations should generally contribute different grad."""
        params = np.array([[0.0, 0.0]])
        T = np.array([1.5])
        scorer = CensoredLogScore()

        Y_obs = Y_from_censored(T, np.array([1]))
        Y_cens = Y_from_censored(T, np.array([0]))
        grad_obs = scorer.d_score(LogNormal(params), Y_obs)
        grad_cens = scorer.d_score(LogNormal(params), Y_cens)
        # Gradients should differ
        assert not np.allclose(grad_obs, grad_cens)


class TestLightningBoostSurvival:
    """Tests for the LightningBoostSurvival estimator."""

    @pytest.fixture()
    def synthetic_survival(
        self,
    ) -> tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.bool_],
    ]:
        """Generate synthetic survival data with random censoring."""
        rng = np.random.default_rng(SEED)
        n = 200
        X = rng.normal(size=(n, 3))
        # True survival times from Weibull with covariates
        shape = 2.0
        scale = np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1])
        T_true = sp_weibull.rvs(c=shape, scale=scale, random_state=rng)
        # Random censoring times
        C = rng.exponential(scale=3.0, size=n)
        T = np.minimum(T_true, C)
        E = (T_true <= C).astype(bool)
        return X, T, E

    def test_fit_runs(
        self,
        synthetic_survival: tuple[
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.bool_],
        ],
    ) -> None:
        """Fit should complete without errors."""
        X, T, E = synthetic_survival
        surv = LightningBoostSurvival(
            n_estimators=10,
            learning_rate=0.05,
            random_state=SEED,
            verbose=False,
        )
        surv.fit(X, T, E)
        assert surv.n_estimators_ == 10

    def test_predict_returns_positive(
        self,
        synthetic_survival: tuple[
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.bool_],
        ],
    ) -> None:
        """Predicted median times should be positive."""
        X, T, E = synthetic_survival
        surv = LightningBoostSurvival(
            n_estimators=20,
            learning_rate=0.05,
            random_state=SEED,
            verbose=False,
        )
        surv.fit(X, T, E)
        preds = surv.predict(X[:10])
        assert preds.shape == (10,)
        assert np.all(preds > 0)
        assert np.all(np.isfinite(preds))

    def test_pred_dist(
        self,
        synthetic_survival: tuple[
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.bool_],
        ],
    ) -> None:
        """Pred_dist should return a distribution with expected methods."""
        X, T, E = synthetic_survival
        surv = LightningBoostSurvival(
            n_estimators=10,
            learning_rate=0.05,
            random_state=SEED,
            verbose=False,
        )
        surv.fit(X, T, E)
        dist = surv.pred_dist(X[:5])
        assert len(dist) == 5
        assert hasattr(dist, "cdf")
        assert hasattr(dist, "ppf")

    def test_early_stopping(
        self,
        synthetic_survival: tuple[
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.bool_],
        ],
    ) -> None:
        """Early stopping should work with validation data."""
        X, T, E = synthetic_survival
        n_train = 150
        surv = LightningBoostSurvival(
            n_estimators=200,
            learning_rate=0.05,
            random_state=SEED,
            verbose=False,
        )
        surv.fit(
            X[:n_train],
            T[:n_train],
            E[:n_train],
            X_val=X[n_train:],
            T_val=T[n_train:],
            E_val=E[n_train:],
            early_stopping_rounds=10,
        )
        assert surv.n_estimators_ < 200

    def test_validation_fraction(
        self,
        synthetic_survival: tuple[
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.bool_],
        ],
    ) -> None:
        """Validation_fraction should auto-split and enable early stopping."""
        X, T, E = synthetic_survival
        surv = LightningBoostSurvival(
            n_estimators=200,
            learning_rate=0.05,
            random_state=SEED,
            verbose=False,
            validation_fraction=0.2,
        )
        surv.fit(X, T, E)
        assert surv.n_estimators_ < 200
        assert hasattr(surv, "val_loss_")

    def test_validation_fraction_conflict(self) -> None:
        """Should raise if both validation_fraction and explicit val data."""
        surv = LightningBoostSurvival(
            n_estimators=10,
            validation_fraction=0.2,
            verbose=False,
        )
        X = np.zeros((20, 2))
        T = np.ones(20)
        E = np.ones(20, dtype=bool)
        with pytest.raises(ValueError, match="cannot both be provided"):
            surv.fit(X, T, E, X_val=X[:5], T_val=T[:5], E_val=E[:5])

    def test_score_method(
        self,
        synthetic_survival: tuple[
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.bool_],
        ],
    ) -> None:
        """Score method should return a finite float."""
        X, T, E = synthetic_survival
        surv = LightningBoostSurvival(
            n_estimators=20,
            learning_rate=0.05,
            random_state=SEED,
            verbose=False,
        )
        surv.fit(X, T, E)
        s = surv.score(X[:50], T[:50], E[:50])
        assert np.isfinite(s)

    def test_staged_predict(
        self,
        synthetic_survival: tuple[
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.bool_],
        ],
    ) -> None:
        """Staged predictions should yield one per iteration."""
        X, T, E = synthetic_survival
        n_est = 10
        surv = LightningBoostSurvival(
            n_estimators=n_est,
            learning_rate=0.05,
            random_state=SEED,
            verbose=False,
        )
        surv.fit(X, T, E)
        stages = list(surv.staged_predict(X[:5]))
        assert len(stages) == n_est
        # Final staged prediction should equal predict
        np.testing.assert_allclose(stages[-1], surv.predict(X[:5]))

    def test_with_weibull_dist(
        self,
        synthetic_survival: tuple[
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.bool_],
        ],
    ) -> None:
        """Should work with Weibull distribution."""
        X, T, E = synthetic_survival
        surv = LightningBoostSurvival(
            dist=Weibull,
            n_estimators=10,
            learning_rate=0.05,
            random_state=SEED,
            verbose=False,
        )
        surv.fit(X, T, E)
        preds = surv.predict(X[:5])
        assert np.all(preds > 0)

    def test_with_exponential_dist(
        self,
        synthetic_survival: tuple[
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.bool_],
        ],
    ) -> None:
        """Should work with Exponential distribution."""
        X, T, E = synthetic_survival
        surv = LightningBoostSurvival(
            dist=Exponential,
            n_estimators=10,
            learning_rate=0.05,
            random_state=SEED,
            verbose=False,
        )
        surv.fit(X, T, E)
        preds = surv.predict(X[:5])
        assert np.all(preds > 0)

    def test_concordance_reasonable(
        self,
        synthetic_survival: tuple[
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.bool_],
        ],
    ) -> None:
        """Concordance index should be better than random (> 0.5)."""
        from lifelines.utils import concordance_index

        X, T, E = synthetic_survival
        surv = LightningBoostSurvival(
            n_estimators=50,
            learning_rate=0.05,
            random_state=SEED,
            verbose=False,
        )
        surv.fit(X, T, E)
        preds = surv.predict(X)
        # concordance_index expects: event_times, predicted_scores, event_observed
        # Higher predicted median -> longer survival time
        ci = concordance_index(T, preds, E)
        assert ci > 0.5

    def test_feature_importances(
        self,
        synthetic_survival: tuple[
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.bool_],
        ],
    ) -> None:
        """Feature importances should have correct shape."""
        X, T, E = synthetic_survival
        surv = LightningBoostSurvival(
            n_estimators=10,
            learning_rate=0.05,
            random_state=SEED,
            verbose=False,
        )
        surv.fit(X, T, E)
        imp = surv.feature_importances_
        assert imp.shape == (2, 3)  # LogNormal has 2 params, 3 features
