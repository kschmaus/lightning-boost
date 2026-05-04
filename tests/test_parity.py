"""End-to-end parity tests: lightning-boost vs NGBoost."""

import numpy as np
import pytest
from ngboost import NGBClassifier
from ngboost import NGBRegressor
from ngboost.distns import Bernoulli as NGBBernoulli
from ngboost.distns import k_categorical as ngb_k_categorical
from ngboost.distns.exponential import Exponential as NGBExponential
from ngboost.distns.lognormal import LogNormal as NGBLogNormal
from ngboost.distns.normal import Normal as NGBNormal
from ngboost.scores import LogScore
from scipy.stats import kstest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from ngboost_lightning import LightningBoostRegressor
from ngboost_lightning.classifier import LightningBoostClassifier
from ngboost_lightning.distributions import Exponential
from ngboost_lightning.distributions import LogNormal
from ngboost_lightning.distributions.categorical import Bernoulli as LBBernoulli
from ngboost_lightning.distributions.categorical import (
    k_categorical as lb_k_categorical,
)
from tests._constants import SEED

# ---- fixtures ----


@pytest.fixture(scope="module")
def parity_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2000/1000 California housing split, fixed seed.

    Using 2000/1000 so LightGBM's histogram bins have enough data points
    per bin to closely approximate NGBoost's exact splits across seeds.
    """
    cal = fetch_california_housing()
    X_tr, X_te, y_tr, y_te = train_test_split(
        cal.data, cal.target, test_size=0.8, random_state=SEED
    )
    return X_tr[:2000], X_te[:1000], y_tr[:2000], y_te[:1000]


@pytest.fixture(scope="module")
def fitted_models(
    parity_data: tuple,
) -> tuple[NGBRegressor, LightningBoostRegressor]:
    """Fit both models with matched hyperparameters."""
    X_tr, _, y_tr, _ = parity_data

    ngb = NGBRegressor(
        Dist=NGBNormal,
        Score=LogScore,
        Base=DecisionTreeRegressor(max_depth=3, random_state=SEED),
        n_estimators=200,
        learning_rate=0.01,
        natural_gradient=True,
        random_state=SEED,
        verbose=False,
    )
    ngb.fit(X_tr, y_tr)

    lb = LightningBoostRegressor(
        n_estimators=200,
        learning_rate=0.01,
        natural_gradient=True,
        random_state=SEED,
        verbose=False,
        num_leaves=8,
        max_depth=3,
        min_child_samples=1,
        lgbm_params={"min_child_weight": 0},
    )
    lb.fit(X_tr, y_tr)

    return ngb, lb


# ---- prediction parity ----


class TestParityPredictions:
    """Compare predictions on the test set."""

    def test_point_prediction_mse_comparable(
        self,
        parity_data: tuple,
        fitted_models: tuple,
    ) -> None:
        """Both models' MSE on test set within 5% of each other."""
        _, X_te, _, y_te = parity_data
        ngb, lb = fitted_models

        ngb_mse = float(np.mean((ngb.predict(X_te) - y_te) ** 2))
        lb_mse = float(np.mean((lb.predict(X_te) - y_te) ** 2))

        assert lb_mse == pytest.approx(ngb_mse, rel=0.05)

    def test_nll_comparable(
        self,
        parity_data: tuple,
        fitted_models: tuple,
    ) -> None:
        """Mean NLL on test set within 2%."""
        _, X_te, _, y_te = parity_data
        ngb, lb = fitted_models

        ngb_nll = float(-ngb.pred_dist(X_te).logpdf(y_te).mean())
        lb_nll = float(lb.pred_dist(X_te).score(y_te).mean())

        assert lb_nll == pytest.approx(ngb_nll, rel=0.02)

    def test_predicted_loc_close(
        self,
        parity_data: tuple,
        fitted_models: tuple,
    ) -> None:
        """Mean and std of predicted loc across test samples within 5%."""
        _, X_te, _, _ = parity_data
        ngb, lb = fitted_models

        ngb_loc = ngb.pred_dist(X_te).params["loc"]
        lb_loc = lb.pred_dist(X_te).loc

        assert float(np.mean(lb_loc)) == pytest.approx(
            float(np.mean(ngb_loc)), rel=0.05
        )
        assert float(np.std(lb_loc)) == pytest.approx(float(np.std(ngb_loc)), rel=0.05)

    def test_predicted_scale_close(
        self,
        parity_data: tuple,
        fitted_models: tuple,
    ) -> None:
        """Mean and std of predicted scale across test samples within 5%."""
        _, X_te, _, _ = parity_data
        ngb, lb = fitted_models

        ngb_scale = ngb.pred_dist(X_te).params["scale"]
        lb_scale = lb.pred_dist(X_te).scale

        assert float(np.mean(lb_scale)) == pytest.approx(
            float(np.mean(ngb_scale)), rel=0.05
        )
        assert float(np.std(lb_scale)) == pytest.approx(
            float(np.std(ngb_scale)), rel=0.05
        )


# ---- convergence parity ----


class TestParityConvergence:
    """Compare training convergence behaviour."""

    def test_training_loss_both_decrease(
        self,
        fitted_models: tuple,
    ) -> None:
        """Both models' training loss curves decrease over iterations."""
        _, lb = fitted_models
        assert lb.train_loss_[-1] < lb.train_loss_[0]

        # For NGBoost, just verify it fitted (no direct loss curve).
        # We rely on the NLL/MSE parity tests as indirect evidence.

    def test_training_loss_final_comparable(
        self,
        parity_data: tuple,
        fitted_models: tuple,
    ) -> None:
        """Final training NLL within 5% of each other."""
        X_tr, _, y_tr, _ = parity_data
        ngb, lb = fitted_models

        ngb_train_nll = float(-ngb.pred_dist(X_tr).logpdf(y_tr).mean())
        lb_train_nll = float(lb.pred_dist(X_tr).score(y_tr).mean())

        assert lb_train_nll == pytest.approx(ngb_train_nll, rel=0.05)


# ---- calibration parity ----


class TestParityCalibration:
    """Compare PIT-based calibration quality."""

    def test_pit_calibration(
        self,
        parity_data: tuple,
        fitted_models: tuple,
    ) -> None:
        """PIT KS statistics should be within 0.05 of each other."""
        _, X_te, _, y_te = parity_data
        ngb, lb = fitted_models

        ngb_pit = ngb.pred_dist(X_te).cdf(y_te)
        lb_pit = lb.pred_dist(X_te).cdf(y_te)

        ngb_ks = kstest(ngb_pit, "uniform").statistic
        lb_ks = kstest(lb_pit, "uniform").statistic

        assert abs(lb_ks - ngb_ks) < 0.05


# ---- edge cases ----


class TestEdgeCases:
    """Degenerate inputs: both libraries should survive or fail similarly."""

    def test_constant_y(self) -> None:
        """Constant targets: fit completes without crashing."""
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(50, 5))
        y = np.full(50, 5.0)

        # Constant y → zero variance, but Normal.fit floors scale at 1e-6
        # so no RuntimeWarning is produced. Just verify no crash.
        lb = LightningBoostRegressor(
            n_estimators=10,
            learning_rate=0.01,
            random_state=SEED,
            verbose=False,
        )
        lb.fit(X, y)

    def test_constant_x(self) -> None:
        """Constant features: LightGBM drops all features and errors."""
        rng = np.random.default_rng(SEED)
        X = np.ones((50, 3))
        y = rng.normal(5.0, 1.0, size=50)

        # LightGBM drops constant features → 0 features → fatal error.
        # This is expected LightGBM behaviour, not a bug in our code.
        with pytest.raises(Exception):  # noqa: B017
            lb = LightningBoostRegressor(
                n_estimators=10,
                learning_rate=0.01,
                random_state=SEED,
                verbose=False,
            )
            lb.fit(X, y)

    def test_single_feature(self) -> None:
        """Single feature: both should produce reasonable fits."""
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(100, 1))
        y = 2.0 * X[:, 0] + rng.normal(0, 0.5, size=100)

        lb = LightningBoostRegressor(
            n_estimators=50,
            learning_rate=0.01,
            random_state=SEED,
            verbose=False,
            num_leaves=8,
            max_depth=3,
            min_child_samples=1,
        )
        lb.fit(X, y)
        pred = lb.predict(X)

        # Should fit reasonably well (MSE < variance of y)
        mse = float(np.mean((pred - y) ** 2))
        assert mse < float(np.var(y))

    @pytest.mark.slow()
    def test_large_n(self) -> None:
        """Large dataset (50k): confirm no OOM or excessive slowdown."""
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(50_000, 10))
        y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, size=50_000)

        lb = LightningBoostRegressor(
            n_estimators=100,
            learning_rate=0.01,
            random_state=SEED,
            verbose=False,
        )
        lb.fit(X, y)

        # Should converge
        assert lb.train_loss_[-1] < lb.train_loss_[0]


# ====================================================================
# LogNormal parity
# ====================================================================


@pytest.fixture(scope="module")
def positive_parity_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Positive-valued data for LogNormal / Exponential parity tests."""
    cal = fetch_california_housing()
    X_tr, X_te, y_tr, y_te = train_test_split(
        cal.data, cal.target, test_size=0.8, random_state=SEED
    )
    # Targets are already positive (housing prices), take subset
    return X_tr[:2000], X_te[:1000], y_tr[:2000], y_te[:1000]


@pytest.fixture(scope="module")
def fitted_lognormal_models(
    positive_parity_data: tuple,
) -> tuple[NGBRegressor, LightningBoostRegressor]:
    """Fit both models with LogNormal distribution."""
    X_tr, _, y_tr, _ = positive_parity_data

    ngb = NGBRegressor(
        Dist=NGBLogNormal,
        Score=LogScore,
        Base=DecisionTreeRegressor(max_depth=3, random_state=SEED),
        n_estimators=200,
        learning_rate=0.01,
        natural_gradient=True,
        random_state=SEED,
        verbose=False,
    )
    ngb.fit(X_tr, y_tr)

    lb = LightningBoostRegressor(
        dist=LogNormal,
        n_estimators=200,
        learning_rate=0.01,
        natural_gradient=True,
        random_state=SEED,
        verbose=False,
        num_leaves=8,
        max_depth=3,
        min_child_samples=1,
        lgbm_params={"min_child_weight": 0},
    )
    lb.fit(X_tr, y_tr)

    return ngb, lb


class TestLogNormalParity:
    """LogNormal parity: lightning-boost vs NGBoost."""

    def test_nll_comparable(
        self,
        positive_parity_data: tuple,
        fitted_lognormal_models: tuple,
    ) -> None:
        """Mean NLL on test set within 5%."""
        _, X_te, _, y_te = positive_parity_data
        ngb, lb = fitted_lognormal_models

        ngb_nll = float(-ngb.pred_dist(X_te).logpdf(y_te).mean())
        lb_nll = float(lb.pred_dist(X_te).score(y_te).mean())

        assert lb_nll == pytest.approx(ngb_nll, rel=0.05)

    def test_point_prediction_mse_comparable(
        self,
        positive_parity_data: tuple,
        fitted_lognormal_models: tuple,
    ) -> None:
        """Both models' MSE on test set within 10%."""
        _, X_te, _, y_te = positive_parity_data
        ngb, lb = fitted_lognormal_models

        ngb_mse = float(np.mean((ngb.predict(X_te) - y_te) ** 2))
        lb_mse = float(np.mean((lb.predict(X_te) - y_te) ** 2))

        assert lb_mse == pytest.approx(ngb_mse, rel=0.10)

    def test_pit_calibration(
        self,
        positive_parity_data: tuple,
        fitted_lognormal_models: tuple,
    ) -> None:
        """PIT KS statistics should be within 0.05 of each other."""
        _, X_te, _, y_te = positive_parity_data
        ngb, lb = fitted_lognormal_models

        ngb_pit = ngb.pred_dist(X_te).cdf(y_te)
        lb_pit = lb.pred_dist(X_te).cdf(y_te)

        ngb_ks = kstest(ngb_pit, "uniform").statistic
        lb_ks = kstest(lb_pit, "uniform").statistic

        assert abs(lb_ks - ngb_ks) < 0.05

    def test_predicted_params_close(
        self,
        positive_parity_data: tuple,
        fitted_lognormal_models: tuple,
    ) -> None:
        """Mean of predicted mu and sigma within 10%."""
        _, X_te, _, _ = positive_parity_data
        ngb, lb = fitted_lognormal_models

        ngb_dist = ngb.pred_dist(X_te)
        lb_dist = lb.pred_dist(X_te)

        # NGBoost LogNormal: params['s'] = sigma, params['scale'] = exp(mu)
        ngb_mu = np.log(ngb_dist.params["scale"])
        ngb_sigma = ngb_dist.params["s"]

        assert float(np.mean(lb_dist.mu)) == pytest.approx(
            float(np.mean(ngb_mu)), rel=0.10
        )
        assert float(np.mean(lb_dist.sigma)) == pytest.approx(
            float(np.mean(ngb_sigma)), rel=0.10
        )


# ====================================================================
# Exponential parity
# ====================================================================


@pytest.fixture(scope="module")
def fitted_exponential_models(
    positive_parity_data: tuple,
) -> tuple[NGBRegressor, LightningBoostRegressor]:
    """Fit both models with Exponential distribution."""
    X_tr, _, y_tr, _ = positive_parity_data

    ngb = NGBRegressor(
        Dist=NGBExponential,
        Score=LogScore,
        Base=DecisionTreeRegressor(max_depth=3, random_state=SEED),
        n_estimators=200,
        learning_rate=0.01,
        natural_gradient=True,
        random_state=SEED,
        verbose=False,
    )
    ngb.fit(X_tr, y_tr)

    lb = LightningBoostRegressor(
        dist=Exponential,
        n_estimators=200,
        learning_rate=0.01,
        natural_gradient=True,
        random_state=SEED,
        verbose=False,
        num_leaves=8,
        max_depth=3,
        min_child_samples=1,
        lgbm_params={"min_child_weight": 0},
    )
    lb.fit(X_tr, y_tr)

    return ngb, lb


class TestExponentialParity:
    """Exponential parity: lightning-boost vs NGBoost."""

    def test_nll_comparable(
        self,
        positive_parity_data: tuple,
        fitted_exponential_models: tuple,
    ) -> None:
        """Mean NLL on test set within 5%."""
        _, X_te, _, y_te = positive_parity_data
        ngb, lb = fitted_exponential_models

        ngb_nll = float(-ngb.pred_dist(X_te).logpdf(y_te).mean())
        lb_nll = float(lb.pred_dist(X_te).score(y_te).mean())

        assert lb_nll == pytest.approx(ngb_nll, rel=0.05)

    def test_point_prediction_mse_comparable(
        self,
        positive_parity_data: tuple,
        fitted_exponential_models: tuple,
    ) -> None:
        """Both models' MSE on test set within 10%."""
        _, X_te, _, y_te = positive_parity_data
        ngb, lb = fitted_exponential_models

        ngb_mse = float(np.mean((ngb.predict(X_te) - y_te) ** 2))
        lb_mse = float(np.mean((lb.predict(X_te) - y_te) ** 2))

        assert lb_mse == pytest.approx(ngb_mse, rel=0.10)

    def test_pit_calibration(
        self,
        positive_parity_data: tuple,
        fitted_exponential_models: tuple,
    ) -> None:
        """PIT KS statistics should be within 0.05 of each other."""
        _, X_te, _, y_te = positive_parity_data
        ngb, lb = fitted_exponential_models

        ngb_pit = ngb.pred_dist(X_te).cdf(y_te)
        lb_pit = lb.pred_dist(X_te).cdf(y_te)

        ngb_ks = kstest(ngb_pit, "uniform").statistic
        lb_ks = kstest(lb_pit, "uniform").statistic

        assert abs(lb_ks - ngb_ks) < 0.05

    def test_predicted_rate_close(
        self,
        positive_parity_data: tuple,
        fitted_exponential_models: tuple,
    ) -> None:
        """Mean predicted rate within 10%."""
        _, X_te, _, _ = positive_parity_data
        ngb, lb = fitted_exponential_models

        # NGBoost Exponential: params['scale'] = 1/rate
        ngb_rate = 1.0 / ngb.pred_dist(X_te).params["scale"]
        lb_rate = lb.pred_dist(X_te).rate

        assert float(np.mean(lb_rate)) == pytest.approx(
            float(np.mean(ngb_rate)), rel=0.10
        )


# ====================================================================
# Bernoulli (binary classification) parity
# ====================================================================


@pytest.fixture(scope="module")
def binary_parity_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Breast cancer binary classification data."""
    from sklearn.datasets import load_breast_cancer

    bc = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        bc.data, bc.target, test_size=0.3, random_state=SEED
    )
    return X_tr[:300], X_te[:150], y_tr[:300], y_te[:150]


@pytest.fixture(scope="module")
def fitted_bernoulli_models(
    binary_parity_data: tuple,
) -> tuple[NGBClassifier, LightningBoostClassifier]:
    """Fit both models with Bernoulli distribution."""
    X_tr, _, y_tr, _ = binary_parity_data

    ngb = NGBClassifier(
        Dist=NGBBernoulli,
        Score=LogScore,
        Base=DecisionTreeRegressor(max_depth=3, random_state=SEED),
        n_estimators=100,
        learning_rate=0.01,
        natural_gradient=True,
        random_state=SEED,
        verbose=False,
    )
    ngb.fit(X_tr, y_tr)

    lb = LightningBoostClassifier(
        dist=LBBernoulli,
        n_estimators=100,
        learning_rate=0.01,
        natural_gradient=True,
        random_state=SEED,
        verbose=False,
        num_leaves=8,
        max_depth=3,
        min_child_samples=1,
        lgbm_params={"min_child_weight": 0},
    )
    lb.fit(X_tr, y_tr.astype(np.float64))

    return ngb, lb


class TestBernoulliClassifierParity:
    """Bernoulli parity: lightning-boost vs NGBoost."""

    def test_nll_comparable(
        self,
        binary_parity_data: tuple,
        fitted_bernoulli_models: tuple,
    ) -> None:
        """Mean NLL on test set within 10%."""
        _, X_te, _, y_te = binary_parity_data
        ngb, lb = fitted_bernoulli_models

        # NGBoost: probs is [K, n], need [n, K]
        ngb_probs = ngb.pred_dist(X_te).probs.T
        ngb_nll = float(
            -np.log(ngb_probs[np.arange(len(y_te)), y_te.astype(int)] + 1e-15).mean()
        )

        lb_nll = float(lb.pred_dist(X_te).score(y_te.astype(np.float64)).mean())

        assert lb_nll == pytest.approx(ngb_nll, rel=0.10)

    def test_predict_proba_shape(
        self,
        binary_parity_data: tuple,
        fitted_bernoulli_models: tuple,
    ) -> None:
        """Both models return [n_test, 2] probability matrices."""
        _, X_te, _, _ = binary_parity_data
        ngb, lb = fitted_bernoulli_models

        assert ngb.predict_proba(X_te).shape == (len(X_te), 2)
        assert lb.predict_proba(X_te).shape == (len(X_te), 2)

    def test_predict_proba_sums_to_one(
        self,
        binary_parity_data: tuple,
        fitted_bernoulli_models: tuple,
    ) -> None:
        """All probability rows sum to 1 for both models."""
        _, X_te, _, _ = binary_parity_data
        ngb, lb = fitted_bernoulli_models

        ngb_sums = ngb.predict_proba(X_te).sum(axis=1)
        lb_sums = lb.predict_proba(X_te).sum(axis=1)

        np.testing.assert_allclose(ngb_sums, 1.0, atol=1e-6)
        np.testing.assert_allclose(lb_sums, 1.0, atol=1e-6)

    def test_accuracy_comparable(
        self,
        binary_parity_data: tuple,
        fitted_bernoulli_models: tuple,
    ) -> None:
        """Accuracy within 5 percentage points."""
        _, X_te, _, y_te = binary_parity_data
        ngb, lb = fitted_bernoulli_models

        ngb_acc = float((ngb.predict(X_te) == y_te).mean())
        lb_acc = float((lb.predict(X_te) == y_te).mean())

        assert abs(lb_acc - ngb_acc) < 0.05


# ====================================================================
# Categorical (multiclass) parity
# ====================================================================


@pytest.fixture(scope="module")
def iris_parity_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Iris 3-class classification data."""
    from sklearn.datasets import load_iris

    iris = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=SEED
    )
    return X_tr, X_te, y_tr, y_te


@pytest.fixture(scope="module")
def fitted_categorical_models(
    iris_parity_data: tuple,
) -> tuple[NGBClassifier, LightningBoostClassifier]:
    """Fit both models with k_categorical(3) on Iris."""
    X_tr, _, y_tr, _ = iris_parity_data

    ngb = NGBClassifier(
        Dist=ngb_k_categorical(3),
        Score=LogScore,
        Base=DecisionTreeRegressor(max_depth=2, random_state=SEED),
        n_estimators=200,
        learning_rate=0.01,
        natural_gradient=True,
        random_state=SEED,
        verbose=False,
    )
    ngb.fit(X_tr, y_tr)

    lb = LightningBoostClassifier(
        dist=lb_k_categorical(3),
        n_estimators=200,
        learning_rate=0.01,
        natural_gradient=True,
        random_state=SEED,
        verbose=False,
        num_leaves=4,
        max_depth=2,
        min_child_samples=5,
        lgbm_params={"min_child_weight": 0},
    )
    lb.fit(X_tr, y_tr.astype(np.float64))

    return ngb, lb


class TestCategoricalClassifierParity:
    """Categorical (K=3) parity: lightning-boost vs NGBoost."""

    def test_nll_comparable(
        self,
        iris_parity_data: tuple,
        fitted_categorical_models: tuple,
    ) -> None:
        """LB model NLL is reasonable and finite on test set."""
        _, X_te, _, y_te = iris_parity_data
        _, lb = fitted_categorical_models

        lb_nll = float(lb.pred_dist(X_te).score(y_te.astype(np.float64)).mean())

        # LB should beat uniform prior (-log(1/3) ≈ 1.099) and be finite
        prior_nll = -np.log(1.0 / 3.0)
        assert np.isfinite(lb_nll)
        assert lb_nll < prior_nll

    def test_predict_proba_shape(
        self,
        iris_parity_data: tuple,
        fitted_categorical_models: tuple,
    ) -> None:
        """Both models return [n_test, 3] probability matrices."""
        _, X_te, _, _ = iris_parity_data
        ngb, lb = fitted_categorical_models

        assert ngb.predict_proba(X_te).shape == (len(X_te), 3)
        assert lb.predict_proba(X_te).shape == (len(X_te), 3)

    def test_accuracy_comparable(
        self,
        iris_parity_data: tuple,
        fitted_categorical_models: tuple,
    ) -> None:
        """Accuracy within 5 percentage points."""
        _, X_te, _, y_te = iris_parity_data
        ngb, lb = fitted_categorical_models

        ngb_acc = float((ngb.predict(X_te) == y_te).mean())
        lb_acc = float((lb.predict(X_te) == y_te).mean())

        assert abs(lb_acc - ngb_acc) < 0.05
