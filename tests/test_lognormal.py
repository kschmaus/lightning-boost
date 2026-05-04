"""Unit tests for the LogNormal distribution."""

import numpy as np
import pytest
from scipy.stats import lognorm as sp_lognorm

from ngboost_lightning.distributions.base import Distribution
from ngboost_lightning.distributions.lognormal import LogNormal


class TestLogNormalFit:
    """Tests for LogNormal.fit (initial parameter estimation)."""

    def test_fit_recovers_parameters(self, rng: np.random.Generator) -> None:
        """fit() should recover (mu, log_sigma) from generated data."""
        true_mu, true_sigma = 1.0, 0.5
        y = rng.lognormal(mean=true_mu, sigma=true_sigma, size=10_000)
        params = LogNormal.fit(y)
        assert params.shape == (2,)
        assert params[0] == pytest.approx(true_mu, abs=0.1)
        assert params[1] == pytest.approx(np.log(true_sigma), abs=0.1)

    def test_fit_returns_finite(self, rng: np.random.Generator) -> None:
        """fit() should return finite values."""
        y = rng.lognormal(size=100)
        params = LogNormal.fit(y)
        assert np.all(np.isfinite(params))


class TestLogNormalConstruction:
    """Tests for LogNormal construction from params."""

    def test_from_params_shape(self) -> None:
        """LogNormal should accept [n_samples, 2] params."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]])
        dist = LogNormal(params)
        assert len(dist) == 3
        assert dist.mu.shape == (3,)
        assert dist.sigma.shape == (3,)

    def test_log_sigma_link(self) -> None:
        """Sigma should be exp(log_sigma)."""
        log_sigma = 0.5
        params = np.array([[0.0, log_sigma]])
        dist = LogNormal(params)
        assert dist.sigma[0] == pytest.approx(np.exp(log_sigma))

    def test_getitem_slice(self) -> None:
        """Slicing should return a LogNormal with fewer samples."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        dist = LogNormal(params)
        sliced = dist[1:]
        assert len(sliced) == 2
        assert sliced.mu[0] == pytest.approx(1.0)

    def test_getitem_int(self) -> None:
        """Integer indexing should return a single-sample LogNormal."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        dist = LogNormal(params)
        single = dist[1]
        assert len(single) == 1
        assert single.mu[0] == pytest.approx(1.0)


class TestLogNormalScore:
    """Tests for score (negative log-likelihood)."""

    def test_score_matches_scipy(self, rng: np.random.Generator) -> None:
        """score() should match -scipy.stats.lognorm.logpdf()."""
        mu, sigma = 1.0, 0.5
        params = np.tile([mu, np.log(sigma)], (100, 1))
        dist = LogNormal(params)
        y = rng.lognormal(mean=mu, sigma=sigma, size=100)

        expected = -sp_lognorm.logpdf(y, s=sigma, scale=np.exp(mu))
        np.testing.assert_allclose(dist.score(y), expected, rtol=1e-10)

    def test_score_is_finite(self, rng: np.random.Generator) -> None:
        """NLL should always be finite for finite positive inputs."""
        params = np.tile([0.0, 0.0], (50, 1))  # sigma=1
        dist = LogNormal(params)
        y = rng.lognormal(size=50)
        assert np.all(np.isfinite(dist.score(y)))

    def test_total_score_is_mean(self, rng: np.random.Generator) -> None:
        """total_score should be the mean of per-sample scores."""
        params = np.tile([0.0, 0.0], (100, 1))
        dist = LogNormal(params)
        y = rng.lognormal(size=100)
        expected = np.mean(dist.score(y))
        assert dist.total_score(y) == pytest.approx(float(expected))


class TestLogNormalGradient:
    """Tests for d_score (analytical gradient)."""

    def test_gradient_shape(self) -> None:
        """d_score should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = LogNormal(params)
        y = np.array([1.1, 2.9])
        grad = dist.d_score(y)
        assert grad.shape == (2, 2)

    def test_gradient_finite_difference(self, rng: np.random.Generator) -> None:
        """Analytical gradient should match finite-difference approximation.

        For each parameter, perturb by eps, compute
        (score+ - score-) / (2*eps), and compare to the analytical gradient.
        """
        n = 50
        mu, log_sigma = 1.0, -0.5
        params = np.tile([mu, log_sigma], (n, 1))
        y = rng.lognormal(mean=mu, sigma=np.exp(log_sigma), size=n)

        dist = LogNormal(params)
        analytical = dist.d_score(y)

        eps = 1e-5
        for k in range(2):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps

            score_plus = LogNormal(params_plus).score(y)
            score_minus = LogNormal(params_minus).score(y)
            numerical = (score_plus - score_minus) / (2 * eps)

            np.testing.assert_allclose(
                analytical[:, k],
                numerical,
                atol=1e-6,
                rtol=1e-4,
                err_msg=f"Gradient mismatch for parameter {k}",
            )

    def test_gradient_formula(self) -> None:
        """Verify gradient matches the analytical formula.

        D[:, 0] = -(log(y) - mu) / sigma^2
        D[:, 1] = 1 - ((log(y) - mu) / sigma)^2
        """
        mu, sigma = 1.0, 0.5
        params = np.array([[mu, np.log(sigma)]])
        dist = LogNormal(params)
        y = np.array([3.0])

        grad = dist.d_score(y)
        log_y = np.log(y[0])
        expected_0 = -(log_y - mu) / sigma**2
        expected_1 = 1.0 - ((log_y - mu) / sigma) ** 2

        assert grad[0, 0] == pytest.approx(expected_0)
        assert grad[0, 1] == pytest.approx(expected_1)


class TestLogNormalFisherInfo:
    """Tests for metric (Fisher Information)."""

    def test_metric_shape(self) -> None:
        """metric() should return [n_samples, 2, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = LogNormal(params)
        fi = dist.metric()
        assert fi.shape == (2, 2, 2)

    def test_metric_diagonal(self) -> None:
        """For LogNormal, Fisher Information should be diagonal."""
        params = np.array([[0.0, np.log(2.0)]])
        dist = LogNormal(params)
        fi = dist.metric()
        assert fi[0, 0, 1] == pytest.approx(0.0)
        assert fi[0, 1, 0] == pytest.approx(0.0)

    def test_metric_values(self) -> None:
        """FI should be diag(1/sigma^2, 2)."""
        sigma = 2.0
        params = np.array([[0.0, np.log(sigma)]])
        dist = LogNormal(params)
        fi = dist.metric()
        assert fi[0, 0, 0] == pytest.approx(1.0 / sigma**2)
        assert fi[0, 1, 1] == pytest.approx(2.0)

    def test_metric_positive_definite(self, rng: np.random.Generator) -> None:
        """Fisher Information should be positive definite for all samples."""
        n = 50
        params = np.column_stack(
            [
                rng.normal(size=n),
                rng.uniform(-1, 2, size=n),  # log_sigma range
            ]
        )
        dist = LogNormal(params)
        fi = dist.metric()
        for i in range(n):
            eigvals = np.linalg.eigvalsh(fi[i])
            assert np.all(eigvals > 0), f"FI not positive definite at sample {i}"


class TestLogNormalNaturalGradient:
    """Tests for natural_gradient (FI^{-1} @ d_score)."""

    def test_natural_gradient_shape(self) -> None:
        """natural_gradient should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = LogNormal(params)
        y = np.array([1.1, 2.9])
        ng = dist.natural_gradient(y)
        assert ng.shape == (2, 2)

    def test_natural_gradient_equals_fi_inv_grad(
        self, rng: np.random.Generator
    ) -> None:
        """natural_gradient should equal FI^{-1} @ d_score."""
        n = 50
        params = np.column_stack(
            [
                rng.normal(size=n),
                rng.uniform(-1, 2, size=n),
            ]
        )
        dist = LogNormal(params)
        y = rng.lognormal(size=n)

        ng = dist.natural_gradient(y)
        grad = dist.d_score(y)
        fi = dist.metric()

        # Manual: solve FI @ x = grad for each sample
        expected = np.linalg.solve(fi, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(ng, expected, rtol=1e-10)

    def test_natural_gradient_equals_base_class_default(
        self, rng: np.random.Generator
    ) -> None:
        """LogNormal's fast-path should match the base class's linalg.solve."""
        n = 30
        params = np.column_stack(
            [
                rng.normal(size=n),
                rng.uniform(-1, 2, size=n),
            ]
        )
        dist = LogNormal(params)
        y = rng.lognormal(size=n)

        fast = dist.natural_gradient(y)
        # Call base class default directly
        base_result = Distribution.natural_gradient(dist, y)
        np.testing.assert_allclose(fast, base_result, rtol=1e-10)


class TestLogNormalSampling:
    """Tests for sample() and statistical methods."""

    def test_sample_shape(self) -> None:
        """sample(n) should return shape [n, n_samples]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = LogNormal(params)
        samples = dist.sample(100)
        assert samples.shape == (100, 2)

    def test_mean_formula(self) -> None:
        """mean() should return exp(mu + sigma^2/2)."""
        mu, sigma = 1.0, 0.5
        params = np.array([[mu, np.log(sigma)]])
        dist = LogNormal(params)
        expected = np.exp(mu + sigma**2 / 2.0)
        assert dist.mean()[0] == pytest.approx(expected)

    def test_cdf(self) -> None:
        """cdf() should match scipy.stats.lognorm.cdf."""
        mu, sigma = 0.0, 1.0
        params = np.array([[mu, np.log(sigma)]])
        dist = LogNormal(params)
        # Median of lognormal(mu=0, sigma=1) is exp(0) = 1.0
        y = np.array([1.0])
        assert dist.cdf(y)[0] == pytest.approx(0.5)

    def test_ppf_inverse_of_cdf(self, rng: np.random.Generator) -> None:
        """ppf(cdf(y)) should return y."""
        mu, sigma = 1.0, 0.5
        params = np.tile([mu, np.log(sigma)], (20, 1))
        dist = LogNormal(params)
        y = rng.lognormal(mean=mu, sigma=sigma, size=20)
        roundtrip = dist.ppf(dist.cdf(y))
        np.testing.assert_allclose(roundtrip, y, rtol=1e-10)

    def test_logpdf_matches_neg_score(self, rng: np.random.Generator) -> None:
        """logpdf() should equal -score()."""
        mu, sigma = 1.0, 0.5
        params = np.tile([mu, np.log(sigma)], (50, 1))
        dist = LogNormal(params)
        y = rng.lognormal(mean=mu, sigma=sigma, size=50)
        np.testing.assert_allclose(dist.logpdf(y), -dist.score(y), rtol=1e-10)
