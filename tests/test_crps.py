"""Tests for CRPS scoring rule and per-distribution CRPS methods."""

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import expon as sp_expon
from scipy.stats import norm as sp_norm

from ngboost_lightning.distributions import Exponential
from ngboost_lightning.distributions import Gamma
from ngboost_lightning.distributions import Laplace
from ngboost_lightning.distributions import LogNormal
from ngboost_lightning.distributions import Normal
from ngboost_lightning.distributions import Poisson
from ngboost_lightning.scoring import CRPScore
from ngboost_lightning.scoring import LogScore
from ngboost_lightning.scoring import ScoringRule

# ---------------------------------------------------------------------------
# ScoringRule protocol conformance
# ---------------------------------------------------------------------------


class TestScoringRuleProtocol:
    """LogScore and CRPScore satisfy the ScoringRule protocol."""

    def test_logscore_is_scoring_rule(self) -> None:
        """LogScore should be an instance of ScoringRule."""
        assert isinstance(LogScore(), ScoringRule)

    def test_crpscore_is_scoring_rule(self) -> None:
        """CRPScore should be an instance of ScoringRule."""
        assert isinstance(CRPScore(), ScoringRule)


# ---------------------------------------------------------------------------
# LogScore delegation
# ---------------------------------------------------------------------------


class TestLogScoreDelegation:
    """LogScore delegates to the distribution's built-in methods."""

    def test_score_matches_dist(self, rng: np.random.Generator) -> None:
        """LogScore.score should equal dist.score."""
        params = np.tile([0.0, 0.0], (30, 1))
        dist = Normal(params)
        y = rng.normal(size=30)
        rule = LogScore()
        np.testing.assert_array_equal(rule.score(dist, y), dist.score(y))

    def test_total_score_matches_dist(self, rng: np.random.Generator) -> None:
        """LogScore.total_score should equal dist.total_score."""
        params = np.tile([0.0, 0.0], (30, 1))
        dist = Normal(params)
        y = rng.normal(size=30)
        rule = LogScore()
        assert rule.total_score(dist, y) == pytest.approx(dist.total_score(y))


# ---------------------------------------------------------------------------
# CRPScore delegation
# ---------------------------------------------------------------------------


class TestCRPScoreDelegation:
    """CRPScore dispatches to crps_* methods on distributions."""

    def test_score_calls_crps_score(self, rng: np.random.Generator) -> None:
        """CRPScore.score should equal dist.crps_score."""
        params = np.tile([0.0, 0.0], (30, 1))
        dist = Normal(params)
        y = rng.normal(size=30)
        rule = CRPScore()
        np.testing.assert_array_equal(rule.score(dist, y), dist.crps_score(y))

    def test_d_score_calls_crps_d_score(self, rng: np.random.Generator) -> None:
        """CRPScore.d_score should equal dist.crps_d_score."""
        params = np.tile([0.0, 0.0], (30, 1))
        dist = Normal(params)
        y = rng.normal(size=30)
        rule = CRPScore()
        np.testing.assert_array_equal(rule.d_score(dist, y), dist.crps_d_score(y))

    def test_metric_calls_crps_metric(self) -> None:
        """CRPScore.metric should equal dist.crps_metric."""
        params = np.tile([0.0, 0.0], (10, 1))
        dist = Normal(params)
        rule = CRPScore()
        np.testing.assert_array_equal(rule.metric(dist), dist.crps_metric())

    def test_total_score_is_mean_crps(self, rng: np.random.Generator) -> None:
        """CRPScore.total_score should be the mean of crps_score."""
        params = np.tile([0.0, 0.0], (30, 1))
        dist = Normal(params)
        y = rng.normal(size=30)
        rule = CRPScore()
        expected = float(np.mean(dist.crps_score(y)))
        assert rule.total_score(dist, y) == pytest.approx(expected)

    def test_natural_gradient_uses_fast_path(self, rng: np.random.Generator) -> None:
        """When crps_natural_gradient exists, CRPScore should use it."""
        params = np.tile([0.0, 0.0], (20, 1))
        dist = Normal(params)
        y = rng.normal(size=20)
        rule = CRPScore()
        np.testing.assert_array_equal(
            rule.natural_gradient(dist, y), dist.crps_natural_gradient(y)
        )

    def test_natural_gradient_fallback_solve(self, rng: np.random.Generator) -> None:
        """For distributions without crps_natural_gradient, solve metric."""
        params = np.column_stack(
            [rng.uniform(0, 1, size=10), rng.uniform(-0.5, 0.5, size=10)]
        )
        dist = Gamma(params)
        y = rng.gamma(np.exp(params[:, 0]), 1.0 / np.exp(params[:, 1]))
        rule = CRPScore()
        ng = rule.natural_gradient(dist, y)
        # Should equal linalg.solve(metric, d_score)
        grad = dist.crps_d_score(y)
        met = dist.crps_metric()
        expected = np.linalg.solve(met, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(ng, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Helper: numerical CRPS via quadrature
# ---------------------------------------------------------------------------


def _numerical_crps_normal(y: float, loc: float, scale: float) -> float:
    """CRPS via numerical integration of (F(x) - 1(x>=y))^2."""

    def integrand(x: float) -> float:
        cdf_val = sp_norm.cdf(x, loc=loc, scale=scale)
        indicator = 1.0 if x >= y else 0.0
        return (cdf_val - indicator) ** 2

    lo = loc - 10 * scale
    hi = loc + 10 * scale
    result, _ = integrate.quad(integrand, lo, hi, limit=200)
    return result


def _numerical_crps_exponential(y: float, rate: float) -> float:
    """CRPS via numerical integration for Exponential."""
    scale = 1.0 / rate

    def integrand(x: float) -> float:
        cdf_val = sp_expon.cdf(x, scale=scale)
        indicator = 1.0 if x >= y else 0.0
        return (cdf_val - indicator) ** 2

    hi = y + 20 * scale
    result, _ = integrate.quad(integrand, 0, hi, limit=200)
    return result


# ---------------------------------------------------------------------------
# Normal CRPS
# ---------------------------------------------------------------------------


class TestNormalCRPS:
    """CRPS methods for Normal distribution."""

    def test_crps_score_shape(self) -> None:
        """crps_score should return [n_samples]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Normal(params)
        y = np.array([0.1, 0.9])
        assert dist.crps_score(y).shape == (2,)

    def test_crps_score_nonnegative(self, rng: np.random.Generator) -> None:
        """CRPS should always be non-negative."""
        params = np.tile([0.0, 0.0], (50, 1))
        dist = Normal(params)
        y = rng.normal(size=50)
        assert np.all(dist.crps_score(y) >= 0)

    def test_crps_score_matches_numerical(self) -> None:
        """Closed-form CRPS should match numerical integration."""
        loc, scale = 2.0, 1.5
        y_vals = np.array([0.5, 2.0, 4.0])
        params = np.tile([loc, np.log(scale)], (len(y_vals), 1))
        dist = Normal(params)
        analytical = dist.crps_score(y_vals)
        for i, y in enumerate(y_vals):
            numerical = _numerical_crps_normal(y, loc, scale)
            assert analytical[i] == pytest.approx(numerical, rel=1e-6)

    def test_crps_d_score_shape(self) -> None:
        """crps_d_score should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Normal(params)
        y = np.array([0.1, 0.9])
        assert dist.crps_d_score(y).shape == (2, 2)

    def test_crps_d_score_finite_difference(self, rng: np.random.Generator) -> None:
        """CRPS analytical gradient should match finite differences."""
        n = 50
        loc, log_scale = 1.0, 0.3
        params = np.tile([loc, log_scale], (n, 1))
        y = rng.normal(loc=loc, scale=np.exp(log_scale), size=n)

        dist = Normal(params)
        analytical = dist.crps_d_score(y)

        eps = 1e-5
        for k in range(2):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps
            score_plus = Normal(params_plus).crps_score(y)
            score_minus = Normal(params_minus).crps_score(y)
            numerical = (score_plus - score_minus) / (2 * eps)
            np.testing.assert_allclose(
                analytical[:, k],
                numerical,
                atol=1e-6,
                rtol=1e-4,
                err_msg=f"CRPS gradient mismatch for Normal param {k}",
            )

    def test_crps_metric_shape(self) -> None:
        """crps_metric should return [n_samples, 2, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Normal(params)
        assert dist.crps_metric().shape == (2, 2, 2)

    def test_crps_metric_diagonal(self) -> None:
        """Normal CRPS metric should be diagonal."""
        params = np.array([[0.0, np.log(2.0)]])
        dist = Normal(params)
        met = dist.crps_metric()
        assert met[0, 0, 1] == pytest.approx(0.0)
        assert met[0, 1, 0] == pytest.approx(0.0)

    def test_crps_metric_positive_definite(self, rng: np.random.Generator) -> None:
        """CRPS metric should be positive definite for all samples."""
        n = 50
        params = np.column_stack([rng.normal(size=n), rng.uniform(-1, 2, size=n)])
        dist = Normal(params)
        met = dist.crps_metric()
        for i in range(n):
            eigvals = np.linalg.eigvalsh(met[i])
            assert np.all(eigvals > 0), (
                f"CRPS metric not positive definite at sample {i}"
            )

    def test_crps_natural_gradient_shape(self) -> None:
        """crps_natural_gradient should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Normal(params)
        y = np.array([0.1, 0.9])
        assert dist.crps_natural_gradient(y).shape == (2, 2)

    def test_crps_natural_gradient_equals_solve(self, rng: np.random.Generator) -> None:
        """Fast-path should match metric^{-1} @ d_score."""
        n = 50
        params = np.column_stack([rng.normal(size=n), rng.uniform(-1, 2, size=n)])
        dist = Normal(params)
        y = rng.normal(size=n)
        fast = dist.crps_natural_gradient(y)
        grad = dist.crps_d_score(y)
        met = dist.crps_metric()
        expected = np.linalg.solve(met, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(fast, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# LogNormal CRPS
# ---------------------------------------------------------------------------


class TestLogNormalCRPS:
    """CRPS methods for LogNormal distribution."""

    def test_crps_score_nonnegative(self, rng: np.random.Generator) -> None:
        """CRPS should always be non-negative."""
        params = np.tile([0.0, 0.0], (50, 1))
        dist = LogNormal(params)
        y = rng.lognormal(mean=0.0, sigma=1.0, size=50)
        assert np.all(dist.crps_score(y) >= 0)

    def test_crps_d_score_finite_difference(self, rng: np.random.Generator) -> None:
        """CRPS gradient should match finite differences."""
        n = 50
        mu, log_sigma = 0.5, -0.3
        params = np.tile([mu, log_sigma], (n, 1))
        y = rng.lognormal(mean=mu, sigma=np.exp(log_sigma), size=n)

        dist = LogNormal(params)
        analytical = dist.crps_d_score(y)

        eps = 1e-5
        for k in range(2):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps
            score_plus = LogNormal(params_plus).crps_score(y)
            score_minus = LogNormal(params_minus).crps_score(y)
            numerical = (score_plus - score_minus) / (2 * eps)
            np.testing.assert_allclose(
                analytical[:, k],
                numerical,
                atol=1e-6,
                rtol=1e-4,
                err_msg=f"CRPS gradient mismatch for LogNormal param {k}",
            )

    def test_crps_metric_positive_definite(self, rng: np.random.Generator) -> None:
        """CRPS metric should be positive definite."""
        n = 30
        params = np.column_stack([rng.normal(size=n), rng.uniform(-1, 1, size=n)])
        dist = LogNormal(params)
        met = dist.crps_metric()
        for i in range(n):
            eigvals = np.linalg.eigvalsh(met[i])
            assert np.all(eigvals > 0)

    def test_crps_natural_gradient_equals_solve(self, rng: np.random.Generator) -> None:
        """Fast-path should match metric^{-1} @ d_score."""
        n = 30
        params = np.column_stack([rng.normal(size=n), rng.uniform(-1, 1, size=n)])
        dist = LogNormal(params)
        y = rng.lognormal(mean=params[:, 0], sigma=np.exp(params[:, 1]))
        fast = dist.crps_natural_gradient(y)
        grad = dist.crps_d_score(y)
        met = dist.crps_metric()
        expected = np.linalg.solve(met, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(fast, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Exponential CRPS
# ---------------------------------------------------------------------------


class TestExponentialCRPS:
    """CRPS methods for Exponential distribution."""

    def test_crps_score_nonnegative(self, rng: np.random.Generator) -> None:
        """CRPS should always be non-negative."""
        params = np.tile([0.0], (50, 1))
        dist = Exponential(params)
        y = rng.exponential(size=50)
        assert np.all(dist.crps_score(y) >= 0)

    def test_crps_score_matches_numerical(self) -> None:
        """Closed-form CRPS should match numerical integration."""
        rate = 2.0
        y_vals = np.array([0.1, 0.5, 2.0])
        params = np.tile([np.log(rate)], (len(y_vals), 1))
        dist = Exponential(params)
        analytical = dist.crps_score(y_vals)
        for i, y in enumerate(y_vals):
            numerical = _numerical_crps_exponential(y, rate)
            assert analytical[i] == pytest.approx(numerical, rel=1e-6)

    def test_crps_d_score_finite_difference(self, rng: np.random.Generator) -> None:
        """CRPS gradient should match finite differences."""
        n = 50
        log_rate = 0.5
        params = np.tile([log_rate], (n, 1))
        y = rng.exponential(scale=np.exp(-log_rate), size=n)

        dist = Exponential(params)
        analytical = dist.crps_d_score(y)

        eps = 1e-5
        params_plus = params.copy()
        params_plus[:, 0] += eps
        params_minus = params.copy()
        params_minus[:, 0] -= eps
        score_plus = Exponential(params_plus).crps_score(y)
        score_minus = Exponential(params_minus).crps_score(y)
        numerical = (score_plus - score_minus) / (2 * eps)
        np.testing.assert_allclose(analytical[:, 0], numerical, atol=1e-6, rtol=1e-4)

    def test_crps_metric_positive(self, rng: np.random.Generator) -> None:
        """CRPS metric should be positive for all samples."""
        n = 30
        params = rng.uniform(-1, 2, size=(n, 1))
        dist = Exponential(params)
        met = dist.crps_metric()
        assert met.shape == (n, 1, 1)
        assert np.all(met > 0)

    def test_crps_natural_gradient_equals_solve(self, rng: np.random.Generator) -> None:
        """Fast-path should match metric^{-1} @ d_score."""
        n = 30
        params = rng.uniform(-1, 2, size=(n, 1))
        dist = Exponential(params)
        y = rng.exponential(scale=1.0 / np.exp(params[:, 0]))
        fast = dist.crps_natural_gradient(y)
        grad = dist.crps_d_score(y)
        met = dist.crps_metric()
        expected = np.linalg.solve(met, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(fast, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Gamma CRPS
# ---------------------------------------------------------------------------


class TestGammaCRPS:
    """CRPS methods for Gamma distribution."""

    def test_crps_score_nonnegative(self, rng: np.random.Generator) -> None:
        """CRPS should always be non-negative."""
        params = np.tile([0.5, 0.5], (30, 1))
        dist = Gamma(params)
        a, b = np.exp(0.5), np.exp(0.5)
        y = rng.gamma(a, 1.0 / b, size=30)
        assert np.all(dist.crps_score(y) >= 0)

    def test_crps_score_shape(self) -> None:
        """crps_score should return [n_samples]."""
        params = np.array([[0.5, 0.5], [1.0, 0.0]])
        dist = Gamma(params)
        y = np.array([1.0, 2.0])
        assert dist.crps_score(y).shape == (2,)

    def test_crps_d_score_shape(self) -> None:
        """crps_d_score should return [n_samples, 2]."""
        params = np.array([[0.5, 0.5], [1.0, 0.0]])
        dist = Gamma(params)
        y = np.array([1.0, 2.0])
        assert dist.crps_d_score(y).shape == (2, 2)

    def test_crps_d_score_finite(self, rng: np.random.Generator) -> None:
        """Finite-difference gradient should be finite."""
        params = np.tile([0.5, 0.5], (10, 1))
        dist = Gamma(params)
        a, b = np.exp(0.5), np.exp(0.5)
        y = rng.gamma(a, 1.0 / b, size=10)
        grad = dist.crps_d_score(y)
        assert np.all(np.isfinite(grad))

    def test_crps_metric_shape(self) -> None:
        """crps_metric should return [n_samples, 2, 2]."""
        params = np.array([[0.5, 0.5], [1.0, 0.0]])
        dist = Gamma(params)
        met = dist.crps_metric()
        assert met.shape == (2, 2, 2)

    def test_crps_metric_positive_definite(self) -> None:
        """MC metric should be positive semi-definite."""
        params = np.array([[0.5, 0.5]])
        dist = Gamma(params)
        met = dist.crps_metric()
        eigvals = np.linalg.eigvalsh(met[0])
        assert np.all(eigvals >= 0)

    def test_crps_natural_gradient_finite(self, rng: np.random.Generator) -> None:
        """Natural gradient via linalg.solve should be finite."""
        params = np.tile([0.5, 0.5], (10, 1))
        dist = Gamma(params)
        a, b = np.exp(0.5), np.exp(0.5)
        y = rng.gamma(a, 1.0 / b, size=10)
        rule = CRPScore()
        ng = rule.natural_gradient(dist, y)
        assert ng.shape == (10, 2)
        assert np.all(np.isfinite(ng))


# ---------------------------------------------------------------------------
# Poisson CRPS
# ---------------------------------------------------------------------------


class TestPoissonCRPS:
    """CRPS methods for Poisson distribution."""

    def test_crps_score_nonnegative(self, rng: np.random.Generator) -> None:
        """CRPS should always be non-negative."""
        params = np.tile([1.0], (30, 1))
        dist = Poisson(params)
        y = rng.poisson(lam=np.exp(1.0), size=30).astype(np.float64)
        assert np.all(dist.crps_score(y) >= 0)

    def test_crps_score_shape(self) -> None:
        """crps_score should return [n_samples]."""
        params = np.array([[0.5], [1.0]])
        dist = Poisson(params)
        y = np.array([1.0, 3.0])
        assert dist.crps_score(y).shape == (2,)

    def test_crps_d_score_shape(self) -> None:
        """crps_d_score should return [n_samples, 1]."""
        params = np.array([[0.5], [1.0]])
        dist = Poisson(params)
        y = np.array([1.0, 3.0])
        assert dist.crps_d_score(y).shape == (2, 1)

    def test_crps_d_score_finite(self, rng: np.random.Generator) -> None:
        """Finite-difference gradient should be finite."""
        params = np.tile([1.0], (10, 1))
        dist = Poisson(params)
        y = rng.poisson(lam=np.exp(1.0), size=10).astype(np.float64)
        grad = dist.crps_d_score(y)
        assert np.all(np.isfinite(grad))

    def test_crps_metric_shape(self) -> None:
        """crps_metric should return [n_samples, 1, 1]."""
        params = np.array([[0.5], [1.0]])
        dist = Poisson(params)
        met = dist.crps_metric()
        assert met.shape == (2, 1, 1)

    def test_crps_metric_positive(self) -> None:
        """CRPS metric should be positive."""
        params = np.array([[1.0]])
        dist = Poisson(params)
        met = dist.crps_metric()
        assert met[0, 0, 0] > 0

    def test_crps_natural_gradient_finite(self, rng: np.random.Generator) -> None:
        """Natural gradient via linalg.solve should be finite."""
        params = np.tile([1.0], (10, 1))
        dist = Poisson(params)
        y = rng.poisson(lam=np.exp(1.0), size=10).astype(np.float64)
        rule = CRPScore()
        ng = rule.natural_gradient(dist, y)
        assert ng.shape == (10, 1)
        assert np.all(np.isfinite(ng))


# ---------------------------------------------------------------------------
# Laplace CRPS
# ---------------------------------------------------------------------------


def _numerical_crps_laplace(y: float, loc: float, scale: float) -> float:
    """CRPS via numerical integration for Laplace."""
    from scipy.stats import laplace as sp_laplace

    def integrand(x: float) -> float:
        cdf_val = sp_laplace.cdf(x, loc=loc, scale=scale)
        indicator = 1.0 if x >= y else 0.0
        return (cdf_val - indicator) ** 2

    lo = loc - 20 * scale
    hi = loc + 20 * scale
    result, _ = integrate.quad(integrand, lo, hi, limit=200)
    return result


class TestLaplaceCRPS:
    """CRPS methods for Laplace distribution."""

    def test_crps_score_shape(self) -> None:
        """crps_score should return [n_samples]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Laplace(params)
        y = np.array([0.1, 0.9])
        assert dist.crps_score(y).shape == (2,)

    def test_crps_score_nonnegative(self, rng: np.random.Generator) -> None:
        """CRPS should always be non-negative."""
        params = np.tile([0.0, 0.0], (50, 1))
        dist = Laplace(params)
        y = rng.laplace(size=50)
        assert np.all(dist.crps_score(y) >= 0)

    def test_crps_score_matches_numerical(self) -> None:
        """Closed-form CRPS should match numerical integration."""
        loc, scale = 2.0, 1.5
        y_vals = np.array([0.5, 2.0, 4.0])
        params = np.tile([loc, np.log(scale)], (len(y_vals), 1))
        dist = Laplace(params)
        analytical = dist.crps_score(y_vals)
        for i, y in enumerate(y_vals):
            numerical = _numerical_crps_laplace(y, loc, scale)
            assert analytical[i] == pytest.approx(numerical, rel=1e-6)

    def test_crps_d_score_shape(self) -> None:
        """crps_d_score should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Laplace(params)
        y = np.array([0.1, 0.9])
        assert dist.crps_d_score(y).shape == (2, 2)

    def test_crps_d_score_finite_difference(self) -> None:
        """crps_d_score should match numerical finite differences."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        y = np.array([2.5, -0.5])
        dist = Laplace(params)
        grad = dist.crps_d_score(y)

        eps = 1e-5
        for k in range(2):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps
            score_plus = Laplace(params_plus).crps_score(y)
            score_minus = Laplace(params_minus).crps_score(y)
            numerical = (score_plus - score_minus) / (2 * eps)
            np.testing.assert_allclose(grad[:, k], numerical, atol=1e-4)

    def test_crps_metric_shape(self) -> None:
        """crps_metric should return [n_samples, 2, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Laplace(params)
        assert dist.crps_metric().shape == (2, 2, 2)

    def test_crps_metric_diagonal(self) -> None:
        """Laplace CRPS metric should be diagonal."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Laplace(params)
        met = dist.crps_metric()
        for i in range(2):
            assert met[i, 0, 1] == 0.0
            assert met[i, 1, 0] == 0.0

    def test_crps_metric_positive_definite(self, rng: np.random.Generator) -> None:
        """CRPS metric should be positive definite."""
        params = np.column_stack([rng.normal(size=20), rng.normal(size=20)])
        dist = Laplace(params)
        met = dist.crps_metric()
        for i in range(20):
            eigvals = np.linalg.eigvalsh(met[i])
            assert np.all(eigvals > 0)

    def test_crps_natural_gradient_shape(self) -> None:
        """crps_natural_gradient should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Laplace(params)
        y = np.array([0.1, 0.9])
        assert dist.crps_natural_gradient(y).shape == (2, 2)

    def test_crps_natural_gradient_equals_solve(self) -> None:
        """Fast path should match linalg.solve(metric, d_score)."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        dist = Laplace(params)
        y = np.array([2.5, -0.5])
        grad = dist.crps_d_score(y)
        met = dist.crps_metric()
        expected = np.linalg.solve(met, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(dist.crps_natural_gradient(y), expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Classifier guard
# ---------------------------------------------------------------------------


class TestClassifierCRPSGuard:
    """CRPScore should be rejected by the classifier."""

    def test_classifier_rejects_crps(self) -> None:
        """fit() should raise ValueError when given CRPScore."""
        from ngboost_lightning.classifier import LightningBoostClassifier

        clf = LightningBoostClassifier(scoring_rule=CRPScore())
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="CRPS"):
            clf.fit(X, y)
