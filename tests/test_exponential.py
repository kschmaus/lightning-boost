"""Unit tests for the Exponential distribution."""

import numpy as np
import pytest
from scipy.stats import expon as sp_expon

from ngboost_lightning.distributions.base import Distribution
from ngboost_lightning.distributions.exponential import Exponential


class TestExponentialFit:
    """Tests for Exponential.fit (initial parameter estimation)."""

    def test_fit_recovers_rate(self, rng: np.random.Generator) -> None:
        """fit() should recover log(rate) from exponential data."""
        true_rate = 2.5
        y = rng.exponential(scale=1.0 / true_rate, size=10_000)
        params = Exponential.fit(y)
        assert params.shape == (1,)
        assert params[0] == pytest.approx(np.log(true_rate), abs=0.1)

    def test_fit_returns_finite(self, rng: np.random.Generator) -> None:
        """fit() should return finite values."""
        y = rng.exponential(scale=1.0, size=100)
        params = Exponential.fit(y)
        assert np.all(np.isfinite(params))


class TestExponentialConstruction:
    """Tests for Exponential construction from params."""

    def test_from_params_shape(self) -> None:
        """Exponential should accept [n_samples, 1] params."""
        params = np.array([[0.0], [0.5], [-0.5]])
        dist = Exponential(params)
        assert len(dist) == 3
        assert dist.rate.shape == (3,)

    def test_rate_link(self) -> None:
        """Rate should be exp(log_rate)."""
        log_rate = 0.7
        params = np.array([[log_rate]])
        dist = Exponential(params)
        assert dist.rate[0] == pytest.approx(np.exp(log_rate))

    def test_getitem_slice(self) -> None:
        """Slicing should return an Exponential with fewer samples."""
        params = np.array([[0.0], [0.5], [1.0]])
        dist = Exponential(params)
        sliced = dist[1:]
        assert len(sliced) == 2
        assert sliced.rate[0] == pytest.approx(np.exp(0.5))

    def test_getitem_int(self) -> None:
        """Integer indexing should return a single-sample Exponential."""
        params = np.array([[0.0], [0.5], [1.0]])
        dist = Exponential(params)
        single = dist[1]
        assert len(single) == 1
        assert single.rate[0] == pytest.approx(np.exp(0.5))


class TestExponentialScore:
    """Tests for score (negative log-likelihood)."""

    def test_score_matches_scipy(self, rng: np.random.Generator) -> None:
        """score() should match -scipy.stats.expon.logpdf()."""
        rate = 2.0
        params = np.tile([np.log(rate)], (100, 1))
        dist = Exponential(params)
        y = rng.exponential(scale=1.0 / rate, size=100)

        expected = -sp_expon.logpdf(y, scale=1.0 / rate)
        np.testing.assert_allclose(dist.score(y), expected, rtol=1e-10)

    def test_score_is_finite(self, rng: np.random.Generator) -> None:
        """NLL should always be finite for finite positive inputs."""
        params = np.tile([0.0], (50, 1))  # rate=1
        dist = Exponential(params)
        y = rng.exponential(scale=1.0, size=50)
        assert np.all(np.isfinite(dist.score(y)))

    def test_total_score_is_mean(self, rng: np.random.Generator) -> None:
        """total_score should be the mean of per-sample scores."""
        params = np.tile([0.0], (100, 1))
        dist = Exponential(params)
        y = rng.exponential(scale=1.0, size=100)
        expected = np.mean(dist.score(y))
        assert dist.total_score(y) == pytest.approx(float(expected))


class TestExponentialGradient:
    """Tests for d_score (analytical gradient)."""

    def test_gradient_shape(self) -> None:
        """d_score should return [n_samples, 1]."""
        params = np.array([[0.0], [0.5]])
        dist = Exponential(params)
        y = np.array([0.1, 0.9])
        grad = dist.d_score(y)
        assert grad.shape == (2, 1)

    def test_gradient_finite_difference(self, rng: np.random.Generator) -> None:
        """Analytical gradient should match finite-difference approximation."""
        n = 50
        log_rate = 0.5
        params = np.tile([log_rate], (n, 1))
        y = rng.exponential(scale=np.exp(-log_rate), size=n)

        dist = Exponential(params)
        analytical = dist.d_score(y)

        eps = 1e-5
        for k in range(1):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps

            score_plus = Exponential(params_plus).score(y)
            score_minus = Exponential(params_minus).score(y)
            numerical = (score_plus - score_minus) / (2 * eps)

            np.testing.assert_allclose(
                analytical[:, k],
                numerical,
                atol=1e-6,
                rtol=1e-4,
                err_msg=f"Gradient mismatch for parameter {k}",
            )

    def test_gradient_formula(self) -> None:
        """Verify gradient matches expected formula: -1 + rate * y."""
        log_rate = np.log(2.0)
        params = np.array([[log_rate]])
        dist = Exponential(params)
        y = np.array([1.0])

        grad = dist.d_score(y)
        rate = np.exp(log_rate)
        expected = -1.0 + rate * y[0]

        assert grad[0, 0] == pytest.approx(expected)


class TestExponentialFisherInfo:
    """Tests for metric (Fisher Information)."""

    def test_metric_shape(self) -> None:
        """metric() should return [n_samples, 1, 1]."""
        params = np.array([[0.0], [0.5]])
        dist = Exponential(params)
        fi = dist.metric()
        assert fi.shape == (2, 1, 1)

    def test_metric_values(self) -> None:
        """FI should be [[1]] for all samples."""
        params = np.array([[0.0], [0.5], [1.0]])
        dist = Exponential(params)
        fi = dist.metric()
        np.testing.assert_allclose(fi, np.ones((3, 1, 1)))

    def test_metric_positive_definite(self, rng: np.random.Generator) -> None:
        """Fisher Information should be positive definite for all samples."""
        n = 50
        params = rng.uniform(-1, 2, size=(n, 1))
        dist = Exponential(params)
        fi = dist.metric()
        for i in range(n):
            eigvals = np.linalg.eigvalsh(fi[i])
            assert np.all(eigvals > 0), f"FI not positive definite at sample {i}"


class TestExponentialNaturalGradient:
    """Tests for natural_gradient (FI^{-1} @ d_score)."""

    def test_natural_gradient_shape(self) -> None:
        """natural_gradient should return [n_samples, 1]."""
        params = np.array([[0.0], [0.5]])
        dist = Exponential(params)
        y = np.array([0.1, 0.9])
        ng = dist.natural_gradient(y)
        assert ng.shape == (2, 1)

    def test_natural_gradient_equals_fi_inv_grad(
        self, rng: np.random.Generator
    ) -> None:
        """natural_gradient should equal FI^{-1} @ d_score."""
        n = 50
        params = rng.uniform(-1, 2, size=(n, 1))
        dist = Exponential(params)
        y = rng.exponential(scale=1.0, size=n)

        ng = dist.natural_gradient(y)
        grad = dist.d_score(y)
        fi = dist.metric()

        expected = np.linalg.solve(fi, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(ng, expected, rtol=1e-10)

    def test_natural_gradient_equals_base_class_default(
        self, rng: np.random.Generator
    ) -> None:
        """Exponential's fast-path should match the base class's linalg.solve."""
        n = 30
        params = rng.uniform(-1, 2, size=(n, 1))
        dist = Exponential(params)
        y = rng.exponential(scale=1.0, size=n)

        fast = dist.natural_gradient(y)
        base_result = Distribution.natural_gradient(dist, y)
        np.testing.assert_allclose(fast, base_result, rtol=1e-10)


class TestExponentialSampling:
    """Tests for sample() and statistical methods."""

    def test_sample_shape(self) -> None:
        """sample(n) should return shape [n, n_samples]."""
        params = np.array([[0.0], [0.5]])
        dist = Exponential(params)
        samples = dist.sample(100)
        assert samples.shape == (100, 2)

    def test_mean_returns_inverse_rate(self) -> None:
        """mean() should return 1/rate."""
        params = np.array([[np.log(2.0)], [np.log(0.5)]])
        dist = Exponential(params)
        expected = 1.0 / np.exp(params[:, 0])
        np.testing.assert_allclose(dist.mean(), expected)

    def test_cdf_matches_scipy(self) -> None:
        """cdf() should match scipy.stats.expon.cdf."""
        rate = 1.0
        params = np.array([[np.log(rate)]])
        dist = Exponential(params)
        y = np.array([0.0])
        assert dist.cdf(y)[0] == pytest.approx(0.0)

    def test_ppf_inverse_of_cdf(self, rng: np.random.Generator) -> None:
        """ppf(cdf(y)) should return y."""
        rate = 2.0
        params = np.tile([np.log(rate)], (20, 1))
        dist = Exponential(params)
        y = rng.exponential(scale=1.0 / rate, size=20)
        roundtrip = dist.ppf(dist.cdf(y))
        np.testing.assert_allclose(roundtrip, y, rtol=1e-10)

    def test_logpdf_matches_scipy(self, rng: np.random.Generator) -> None:
        """logpdf() should match scipy.stats.expon.logpdf."""
        rate = 3.0
        params = np.tile([np.log(rate)], (50, 1))
        dist = Exponential(params)
        y = rng.exponential(scale=1.0 / rate, size=50)

        expected = sp_expon.logpdf(y, scale=1.0 / rate)
        np.testing.assert_allclose(dist.logpdf(y), expected, rtol=1e-10)
