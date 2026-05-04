"""Unit tests for the distribution abstraction layer."""

import numpy as np
import pytest
from scipy.stats import norm as sp_norm

from ngboost_lightning.distributions import Normal
from ngboost_lightning.distributions.base import Distribution


class TestNormalFit:
    """Tests for Normal.fit (initial parameter estimation)."""

    def test_fit_recovers_parameters(self, rng: np.random.Generator) -> None:
        """fit() should recover (mean, log_scale) from generated data."""
        true_loc, true_scale = 3.0, 2.0
        y = rng.normal(loc=true_loc, scale=true_scale, size=10_000)
        params = Normal.fit(y)
        assert params.shape == (2,)
        assert params[0] == pytest.approx(true_loc, abs=0.1)
        assert params[1] == pytest.approx(np.log(true_scale), abs=0.1)

    def test_fit_returns_finite(self, rng: np.random.Generator) -> None:
        """fit() should return finite values."""
        y = rng.normal(size=100)
        params = Normal.fit(y)
        assert np.all(np.isfinite(params))


class TestNormalConstruction:
    """Tests for Normal construction from params."""

    def test_from_params_shape(self) -> None:
        """Normal should accept [n_samples, 2] params."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]])
        dist = Normal(params)
        assert len(dist) == 3
        assert dist.loc.shape == (3,)
        assert dist.scale.shape == (3,)

    def test_log_scale_link(self) -> None:
        """Scale should be exp(log_scale)."""
        log_scale = 0.5
        params = np.array([[0.0, log_scale]])
        dist = Normal(params)
        assert dist.scale[0] == pytest.approx(np.exp(log_scale))

    def test_getitem_slice(self) -> None:
        """Slicing should return a Normal with fewer samples."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        dist = Normal(params)
        sliced = dist[1:]
        assert len(sliced) == 2
        assert sliced.loc[0] == pytest.approx(1.0)

    def test_getitem_int(self) -> None:
        """Integer indexing should return a single-sample Normal."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        dist = Normal(params)
        single = dist[1]
        assert len(single) == 1
        assert single.loc[0] == pytest.approx(1.0)


class TestNormalScore:
    """Tests for score (negative log-likelihood)."""

    def test_score_matches_scipy(self, rng: np.random.Generator) -> None:
        """score() should match -scipy.stats.norm.logpdf()."""
        loc, scale = 2.0, 1.5
        params = np.tile([loc, np.log(scale)], (100, 1))
        dist = Normal(params)
        y = rng.normal(loc=loc, scale=scale, size=100)

        expected = -sp_norm.logpdf(y, loc=loc, scale=scale)
        np.testing.assert_allclose(dist.score(y), expected, rtol=1e-10)

    def test_score_is_finite(self, rng: np.random.Generator) -> None:
        """NLL should always be finite for finite inputs."""
        params = np.tile([0.0, 0.0], (50, 1))  # scale=1
        dist = Normal(params)
        y = rng.normal(size=50)
        assert np.all(np.isfinite(dist.score(y)))

    def test_total_score_is_mean(self, rng: np.random.Generator) -> None:
        """total_score should be the mean of per-sample scores."""
        params = np.tile([0.0, 0.0], (100, 1))
        dist = Normal(params)
        y = rng.normal(size=100)
        expected = np.mean(dist.score(y))
        assert dist.total_score(y) == pytest.approx(float(expected))


class TestNormalGradient:
    """Tests for d_score (analytical gradient)."""

    def test_gradient_shape(self) -> None:
        """d_score should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Normal(params)
        y = np.array([0.1, 0.9])
        grad = dist.d_score(y)
        assert grad.shape == (2, 2)

    def test_gradient_finite_difference(self, rng: np.random.Generator) -> None:
        """Analytical gradient should match finite-difference approximation.

        For each parameter, perturb by eps, compute
        (score+ - score-) / (2*eps), and compare to the analytical gradient.
        """
        n = 50
        loc, log_scale = 1.0, 0.3
        params = np.tile([loc, log_scale], (n, 1))
        y = rng.normal(loc=loc, scale=np.exp(log_scale), size=n)

        dist = Normal(params)
        analytical = dist.d_score(y)

        eps = 1e-5
        for k in range(2):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps

            score_plus = Normal(params_plus).score(y)
            score_minus = Normal(params_minus).score(y)
            numerical = (score_plus - score_minus) / (2 * eps)

            np.testing.assert_allclose(
                analytical[:, k],
                numerical,
                atol=1e-6,
                rtol=1e-4,
                err_msg=f"Gradient mismatch for parameter {k}",
            )

    def test_gradient_matches_ngboost_formula(self) -> None:
        """Verify gradient matches NGBoost's NormalLogScore.d_score formula.

        NGBoost: D[:, 0] = (loc - Y) / var
                 D[:, 1] = 1 - ((loc - Y)^2) / var
        """
        params = np.array([[2.0, np.log(1.5)]])
        dist = Normal(params)
        y = np.array([1.0])

        grad = dist.d_score(y)
        loc, var = 2.0, 1.5**2
        expected_0 = (loc - y[0]) / var
        expected_1 = 1.0 - ((loc - y[0]) ** 2) / var

        assert grad[0, 0] == pytest.approx(expected_0)
        assert grad[0, 1] == pytest.approx(expected_1)


class TestNormalFisherInfo:
    """Tests for metric (Fisher Information)."""

    def test_metric_shape(self) -> None:
        """metric() should return [n_samples, 2, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Normal(params)
        fi = dist.metric()
        assert fi.shape == (2, 2, 2)

    def test_metric_diagonal(self) -> None:
        """For Normal, Fisher Information should be diagonal."""
        params = np.array([[0.0, np.log(2.0)]])
        dist = Normal(params)
        fi = dist.metric()
        assert fi[0, 0, 1] == pytest.approx(0.0)
        assert fi[0, 1, 0] == pytest.approx(0.0)

    def test_metric_values(self) -> None:
        """FI should be diag(1/scale^2, 2)."""
        scale = 2.0
        params = np.array([[0.0, np.log(scale)]])
        dist = Normal(params)
        fi = dist.metric()
        assert fi[0, 0, 0] == pytest.approx(1.0 / scale**2)
        assert fi[0, 1, 1] == pytest.approx(2.0)

    def test_metric_positive_definite(self, rng: np.random.Generator) -> None:
        """Fisher Information should be positive definite for all samples."""
        n = 50
        params = np.column_stack(
            [
                rng.normal(size=n),
                rng.uniform(-1, 2, size=n),  # log_scale range
            ]
        )
        dist = Normal(params)
        fi = dist.metric()
        for i in range(n):
            eigvals = np.linalg.eigvalsh(fi[i])
            assert np.all(eigvals > 0), f"FI not positive definite at sample {i}"


class TestNormalNaturalGradient:
    """Tests for natural_gradient (FI^{-1} @ d_score)."""

    def test_natural_gradient_shape(self) -> None:
        """natural_gradient should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Normal(params)
        y = np.array([0.1, 0.9])
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
        dist = Normal(params)
        y = rng.normal(size=n)

        ng = dist.natural_gradient(y)
        grad = dist.d_score(y)
        fi = dist.metric()

        # Manual: solve FI @ x = grad for each sample
        expected = np.linalg.solve(fi, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(ng, expected, rtol=1e-10)

    def test_natural_gradient_equals_base_class_default(
        self, rng: np.random.Generator
    ) -> None:
        """Normal's fast-path should match the base class's linalg.solve."""
        n = 30
        params = np.column_stack(
            [
                rng.normal(size=n),
                rng.uniform(-1, 2, size=n),
            ]
        )
        dist = Normal(params)
        y = rng.normal(size=n)

        fast = dist.natural_gradient(y)
        # Call base class default directly
        base_result = Distribution.natural_gradient(dist, y)
        np.testing.assert_allclose(fast, base_result, rtol=1e-10)


class TestNormalSampling:
    """Tests for sample() and statistical methods."""

    def test_sample_shape(self) -> None:
        """sample(n) should return shape [n, n_samples]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Normal(params)
        samples = dist.sample(100)
        assert samples.shape == (100, 2)

    def test_mean_returns_loc(self) -> None:
        """mean() should return loc."""
        params = np.array([[3.0, 1.0], [-1.0, 0.0]])
        dist = Normal(params)
        np.testing.assert_array_equal(dist.mean(), params[:, 0])

    def test_cdf_matches_scipy(self) -> None:
        """cdf() should match scipy.stats.norm.cdf."""
        params = np.array([[0.0, 0.0]])  # standard normal
        dist = Normal(params)
        y = np.array([0.0])
        assert dist.cdf(y)[0] == pytest.approx(0.5)

    def test_ppf_inverse_of_cdf(self, rng: np.random.Generator) -> None:
        """ppf(cdf(y)) should return y."""
        params = np.tile([1.0, 0.5], (20, 1))
        dist = Normal(params)
        y = rng.normal(loc=1.0, scale=np.exp(0.5), size=20)
        roundtrip = dist.ppf(dist.cdf(y))
        np.testing.assert_allclose(roundtrip, y, rtol=1e-10)
