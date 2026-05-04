"""Unit tests for the Poisson distribution."""

import numpy as np
import pytest
from scipy.stats import poisson as sp_poisson

from ngboost_lightning.distributions.base import Distribution
from ngboost_lightning.distributions.poisson import Poisson


class TestPoissonFit:
    """Tests for Poisson.fit (initial parameter estimation)."""

    def test_fit_recovers_rate(self, rng: np.random.Generator) -> None:
        """fit() should recover log(rate) from Poisson data."""
        true_rate = 5.0
        y = rng.poisson(lam=true_rate, size=10_000).astype(np.float64)
        params = Poisson.fit(y)
        assert params.shape == (1,)
        assert params[0] == pytest.approx(np.log(true_rate), abs=0.1)

    def test_fit_returns_finite(self, rng: np.random.Generator) -> None:
        """fit() should return finite values."""
        y = rng.poisson(lam=3.0, size=100).astype(np.float64)
        params = Poisson.fit(y)
        assert np.all(np.isfinite(params))


class TestPoissonConstruction:
    """Tests for Poisson construction from params."""

    def test_from_params_shape(self) -> None:
        """Poisson should accept [n_samples, 1] params."""
        params = np.array([[1.0], [0.5], [-0.5]])
        dist = Poisson(params)
        assert len(dist) == 3
        assert dist.rate.shape == (3,)

    def test_rate_link(self) -> None:
        """Rate should be exp(log_rate)."""
        log_rate = 1.5
        params = np.array([[log_rate]])
        dist = Poisson(params)
        assert dist.rate[0] == pytest.approx(np.exp(log_rate))

    def test_getitem_slice(self) -> None:
        """Slicing should return a Poisson with fewer samples."""
        params = np.array([[0.5], [1.0], [1.5]])
        dist = Poisson(params)
        sliced = dist[1:]
        assert len(sliced) == 2
        assert sliced.rate[0] == pytest.approx(np.exp(1.0))

    def test_getitem_int(self) -> None:
        """Integer indexing should return a single-sample Poisson."""
        params = np.array([[0.5], [1.0], [1.5]])
        dist = Poisson(params)
        single = dist[1]
        assert len(single) == 1
        assert single.rate[0] == pytest.approx(np.exp(1.0))


class TestPoissonScore:
    """Tests for score (negative log-likelihood)."""

    def test_score_matches_scipy(self, rng: np.random.Generator) -> None:
        """score() should match -scipy.stats.poisson.logpmf()."""
        rate = 4.0
        log_rate = np.log(rate)
        n = 100
        params = np.full((n, 1), log_rate)
        dist = Poisson(params)
        y = rng.poisson(lam=rate, size=n).astype(np.float64)

        expected = -sp_poisson.logpmf(y, mu=rate)
        np.testing.assert_allclose(dist.score(y), expected, rtol=1e-10)

    def test_score_is_finite(self, rng: np.random.Generator) -> None:
        """NLL should be finite for valid count inputs."""
        params = np.full((50, 1), np.log(3.0))
        dist = Poisson(params)
        y = rng.poisson(lam=3.0, size=50).astype(np.float64)
        assert np.all(np.isfinite(dist.score(y)))

    def test_total_score_is_mean(self, rng: np.random.Generator) -> None:
        """total_score should be the mean of per-sample scores."""
        params = np.full((100, 1), np.log(5.0))
        dist = Poisson(params)
        y = rng.poisson(lam=5.0, size=100).astype(np.float64)
        expected = np.mean(dist.score(y))
        assert dist.total_score(y) == pytest.approx(float(expected))


class TestPoissonGradient:
    """Tests for d_score (analytical gradient)."""

    def test_gradient_shape(self) -> None:
        """d_score should return [n_samples, 1]."""
        params = np.array([[1.0], [0.5]])
        dist = Poisson(params)
        y = np.array([2.0, 1.0])
        grad = dist.d_score(y)
        assert grad.shape == (2, 1)

    def test_gradient_finite_difference(self, rng: np.random.Generator) -> None:
        """Analytical gradient should match finite-difference approximation.

        For discrete distributions, we perturb the parameters (not y) and
        compare the numerical derivative to the analytical gradient.
        """
        n = 50
        rate = 5.0
        log_rate = np.log(rate)
        params = np.full((n, 1), log_rate)
        y = rng.poisson(lam=rate, size=n).astype(np.float64)

        dist = Poisson(params)
        analytical = dist.d_score(y)

        eps = 1e-5
        for k in range(1):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps

            score_plus = Poisson(params_plus).score(y)
            score_minus = Poisson(params_minus).score(y)
            numerical = (score_plus - score_minus) / (2 * eps)

            np.testing.assert_allclose(
                analytical[:, k],
                numerical,
                atol=1e-6,
                rtol=1e-4,
                err_msg=f"Gradient mismatch for parameter {k}",
            )

    def test_gradient_formula(self) -> None:
        """Verify gradient matches the formula: rate - y."""
        rate = 3.0
        params = np.array([[np.log(rate)]])
        dist = Poisson(params)
        y = np.array([5.0])

        grad = dist.d_score(y)
        expected = rate - y[0]
        assert grad[0, 0] == pytest.approx(expected)


class TestPoissonFisherInfo:
    """Tests for metric (Fisher Information)."""

    def test_metric_shape(self) -> None:
        """metric() should return [n_samples, 1, 1]."""
        params = np.array([[1.0], [0.5]])
        dist = Poisson(params)
        fi = dist.metric()
        assert fi.shape == (2, 1, 1)

    def test_metric_values_equal_rate(self) -> None:
        """FI should equal rate for log-rate parameterization."""
        rate = 4.0
        params = np.array([[np.log(rate)]])
        dist = Poisson(params)
        fi = dist.metric()
        assert fi[0, 0, 0] == pytest.approx(rate)

    def test_metric_positive(self, rng: np.random.Generator) -> None:
        """Fisher Information should be positive (rate > 0)."""
        n = 50
        params = rng.uniform(-1, 3, size=(n, 1))
        dist = Poisson(params)
        fi = dist.metric()
        assert np.all(fi[:, 0, 0] > 0)


class TestPoissonNaturalGradient:
    """Tests for natural_gradient (FI^{-1} @ d_score)."""

    def test_natural_gradient_shape(self) -> None:
        """natural_gradient should return [n_samples, 1]."""
        params = np.array([[1.0], [0.5]])
        dist = Poisson(params)
        y = np.array([2.0, 1.0])
        ng = dist.natural_gradient(y)
        assert ng.shape == (2, 1)

    def test_natural_gradient_equals_fi_inv_grad(
        self, rng: np.random.Generator
    ) -> None:
        """natural_gradient should equal FI^{-1} @ d_score."""
        n = 50
        params = rng.uniform(0.5, 2.0, size=(n, 1))
        dist = Poisson(params)
        y = rng.poisson(lam=dist.rate, size=n).astype(np.float64)

        ng = dist.natural_gradient(y)
        grad = dist.d_score(y)
        fi = dist.metric()

        expected = np.linalg.solve(fi, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(ng, expected, rtol=1e-10)

    def test_natural_gradient_equals_base_class_default(
        self, rng: np.random.Generator
    ) -> None:
        """Poisson's fast-path should match the base class's linalg.solve."""
        n = 30
        params = rng.uniform(0.5, 2.0, size=(n, 1))
        dist = Poisson(params)
        y = rng.poisson(lam=dist.rate, size=n).astype(np.float64)

        fast = dist.natural_gradient(y)
        base_result = Distribution.natural_gradient(dist, y)
        np.testing.assert_allclose(fast, base_result, rtol=1e-10)


class TestPoissonSampling:
    """Tests for sample() and statistical methods."""

    def test_sample_shape(self) -> None:
        """sample(n) should return shape [n, n_samples]."""
        params = np.array([[1.0], [0.5]])
        dist = Poisson(params)
        samples = dist.sample(100)
        assert samples.shape == (100, 2)

    def test_sample_mean_equals_rate(self, rng: np.random.Generator) -> None:
        """Sample mean should approximate rate."""
        rate = 7.0
        params = np.array([[np.log(rate)]])
        dist = Poisson(params)
        samples = dist.sample(10_000)
        assert np.mean(samples) == pytest.approx(rate, rel=0.05)

    def test_cdf(self) -> None:
        """cdf() should match scipy.stats.poisson.cdf."""
        rate = 3.0
        params = np.array([[np.log(rate)]])
        dist = Poisson(params)
        y = np.array([3.0])
        expected = sp_poisson.cdf(3.0, mu=rate)
        assert dist.cdf(y)[0] == pytest.approx(expected)

    def test_ppf_returns_integers(self) -> None:
        """ppf() should return integer-like values for discrete distribution."""
        rate = 5.0
        params = np.array([[np.log(rate)]])
        dist = Poisson(params)
        q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        result = dist.ppf(q)
        np.testing.assert_array_equal(result, np.floor(result))

    def test_logpdf_matches_logpmf(self) -> None:
        """logpdf() should return logpmf values for this discrete distribution."""
        rate = 4.0
        params = np.array([[np.log(rate)]])
        dist = Poisson(params)
        y = np.array([2.0])
        expected = sp_poisson.logpmf(2.0, mu=rate)
        assert dist.logpdf(y)[0] == pytest.approx(expected)

    def test_mean_returns_rate(self) -> None:
        """mean() should return rate."""
        params = np.array([[np.log(3.0)], [np.log(7.0)]])
        dist = Poisson(params)
        expected = np.exp(params[:, 0])
        np.testing.assert_allclose(dist.mean(), expected)
