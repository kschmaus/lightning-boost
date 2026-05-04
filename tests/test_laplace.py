"""Unit tests for the Laplace distribution."""

import numpy as np
import pytest
from scipy.stats import laplace as sp_laplace

from ngboost_lightning.distributions.base import Distribution
from ngboost_lightning.distributions.laplace import Laplace


class TestLaplaceFit:
    """Tests for Laplace.fit (initial parameter estimation)."""

    def test_fit_recovers_parameters(self, rng: np.random.Generator) -> None:
        """fit() should recover (loc, log_scale) from generated data."""
        true_loc, true_scale = 3.0, 2.0
        y = rng.laplace(loc=true_loc, scale=true_scale, size=10_000)
        params = Laplace.fit(y)
        assert params.shape == (2,)
        assert params[0] == pytest.approx(true_loc, abs=0.15)
        assert params[1] == pytest.approx(np.log(true_scale), abs=0.1)

    def test_fit_returns_finite(self, rng: np.random.Generator) -> None:
        """fit() should return finite values."""
        y = rng.laplace(size=100)
        params = Laplace.fit(y)
        assert np.all(np.isfinite(params))


class TestLaplaceConstruction:
    """Tests for Laplace construction from params."""

    def test_from_params_shape(self) -> None:
        """Laplace should accept [n_samples, 2] params."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]])
        dist = Laplace(params)
        assert len(dist) == 3
        assert dist.loc.shape == (3,)
        assert dist.scale.shape == (3,)

    def test_log_scale_link(self) -> None:
        """Scale should be exp(log_scale)."""
        log_scale = 0.5
        params = np.array([[0.0, log_scale]])
        dist = Laplace(params)
        assert dist.scale[0] == pytest.approx(np.exp(log_scale))

    def test_getitem_slice(self) -> None:
        """Slicing should return a Laplace with fewer samples."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        dist = Laplace(params)
        sub = dist[:2]
        assert isinstance(sub, Laplace)
        assert len(sub) == 2

    def test_getitem_int(self) -> None:
        """Integer indexing should return a single-sample Laplace."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Laplace(params)
        sub = dist[0]
        assert isinstance(sub, Laplace)
        assert len(sub) == 1


class TestLaplaceScore:
    """Tests for Laplace.score (negative log-likelihood)."""

    def test_score_matches_scipy(self) -> None:
        """score() should match -scipy.logpdf."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        dist = Laplace(params)
        y = np.array([2.5, -0.5])
        expected = -sp_laplace.logpdf(y, loc=dist.loc, scale=dist.scale)
        np.testing.assert_allclose(dist.score(y), expected, rtol=1e-10)

    def test_score_is_finite(self, rng: np.random.Generator) -> None:
        """score() should return finite values."""
        params = np.column_stack([rng.normal(size=50), rng.normal(size=50)])
        dist = Laplace(params)
        y = rng.normal(size=50)
        assert np.all(np.isfinite(dist.score(y)))

    def test_total_score_is_mean(self) -> None:
        """total_score() should return mean of score()."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]])
        dist = Laplace(params)
        y = np.array([0.1, 0.9, -1.1])
        expected = float(np.mean(dist.score(y)))
        assert dist.total_score(y) == pytest.approx(expected)


class TestLaplaceGradient:
    """Tests for Laplace.d_score (gradient of NLL)."""

    def test_gradient_shape(self) -> None:
        """d_score() should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Laplace(params)
        y = np.array([0.5, 1.5])
        grad = dist.d_score(y)
        assert grad.shape == (2, 2)

    def test_gradient_finite_difference(self) -> None:
        """d_score() should match numerical finite differences."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        y = np.array([2.5, -0.5])
        dist = Laplace(params)
        grad = dist.d_score(y)

        eps = 1e-5
        for k in range(2):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps
            score_plus = Laplace(params_plus).score(y)
            score_minus = Laplace(params_minus).score(y)
            numerical = (score_plus - score_minus) / (2 * eps)
            np.testing.assert_allclose(grad[:, k], numerical, atol=1e-4)

    def test_gradient_formula(self) -> None:
        """d_score() should match hand-derived formulas."""
        loc, log_scale = 2.0, np.log(1.5)
        params = np.array([[loc, log_scale]])
        dist = Laplace(params)
        y = np.array([3.0])
        scale = np.exp(log_scale)

        grad = dist.d_score(y)
        # d/d(loc) = sign(loc - y) / scale
        assert grad[0, 0] == pytest.approx(np.sign(loc - y[0]) / scale)
        # d/d(log_scale) = 1 - |y - loc| / scale
        assert grad[0, 1] == pytest.approx(1.0 - abs(y[0] - loc) / scale)


class TestLaplaceFisherInfo:
    """Tests for Laplace.metric (Fisher Information)."""

    def test_metric_shape(self) -> None:
        """metric() should return [n_samples, 2, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Laplace(params)
        fi = dist.metric()
        assert fi.shape == (2, 2, 2)

    def test_metric_diagonal(self) -> None:
        """Laplace FI should be diagonal."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Laplace(params)
        fi = dist.metric()
        for i in range(2):
            assert fi[i, 0, 1] == 0.0
            assert fi[i, 1, 0] == 0.0

    def test_metric_values(self) -> None:
        """FI should be diag(1/scale^2, 1)."""
        scale = 1.5
        params = np.array([[0.0, np.log(scale)]])
        dist = Laplace(params)
        fi = dist.metric()
        assert fi[0, 0, 0] == pytest.approx(1.0 / scale**2)
        assert fi[0, 1, 1] == pytest.approx(1.0)

    def test_metric_positive_definite(self, rng: np.random.Generator) -> None:
        """FI should be positive definite for all samples."""
        params = np.column_stack([rng.normal(size=20), rng.normal(size=20)])
        dist = Laplace(params)
        fi = dist.metric()
        for i in range(20):
            eigvals = np.linalg.eigvalsh(fi[i])
            assert np.all(eigvals > 0)


class TestLaplaceNaturalGradient:
    """Tests for Laplace.natural_gradient."""

    def test_natural_gradient_shape(self) -> None:
        """natural_gradient() should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Laplace(params)
        y = np.array([0.5, 1.5])
        ng = dist.natural_gradient(y)
        assert ng.shape == (2, 2)

    def test_natural_gradient_equals_fi_inv_grad(self) -> None:
        """natural_gradient should equal FI^{-1} @ d_score."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        dist = Laplace(params)
        y = np.array([2.5, -0.5])
        grad = dist.d_score(y)
        fi = dist.metric()
        expected = np.linalg.solve(fi, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(dist.natural_gradient(y), expected, rtol=1e-10)

    def test_natural_gradient_equals_base_class_default(self) -> None:
        """Fast path should match base class generic implementation."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        dist = Laplace(params)
        y = np.array([2.5, -0.5])
        fast = dist.natural_gradient(y)
        generic = Distribution.natural_gradient(dist, y)
        np.testing.assert_allclose(fast, generic, rtol=1e-10)


class TestLaplaceSampling:
    """Tests for Laplace sampling, CDF, PPF, logpdf."""

    def test_sample_shape(self) -> None:
        """sample(n) should return [n, n_samples]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Laplace(params)
        s = dist.sample(100)
        assert s.shape == (100, 2)

    def test_mean_returns_loc(self) -> None:
        """mean() should return loc."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        dist = Laplace(params)
        np.testing.assert_allclose(dist.mean(), params[:, 0])

    def test_cdf_matches_scipy(self) -> None:
        """cdf() should match scipy."""
        params = np.array([[2.0, np.log(1.5)]])
        dist = Laplace(params)
        y = np.array([3.0])
        expected = sp_laplace.cdf(y, loc=2.0, scale=1.5)
        np.testing.assert_allclose(dist.cdf(y), expected, rtol=1e-10)

    def test_ppf_inverse_of_cdf(self) -> None:
        """ppf(cdf(y)) should return y."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        dist = Laplace(params)
        y = np.array([2.5, -0.5])
        np.testing.assert_allclose(dist.ppf(dist.cdf(y)), y, rtol=1e-10)

    def test_logpdf_matches_scipy(self) -> None:
        """logpdf() should match scipy."""
        params = np.array([[2.0, np.log(1.5)]])
        dist = Laplace(params)
        y = np.array([3.0])
        expected = sp_laplace.logpdf(y, loc=2.0, scale=1.5)
        np.testing.assert_allclose(dist.logpdf(y), expected, rtol=1e-10)

    def test_is_subclass_of_distribution(self) -> None:
        """Laplace should be a Distribution subclass."""
        assert issubclass(Laplace, Distribution)
