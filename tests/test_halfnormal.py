"""Unit tests for the HalfNormal distribution."""

import numpy as np
import pytest
from scipy.stats import halfnorm as sp_halfnorm

from ngboost_lightning.distributions.base import Distribution
from ngboost_lightning.distributions.halfnormal import HalfNormal


class TestHalfNormalFit:
    """Tests for HalfNormal.fit (initial parameter estimation)."""

    def test_fit_recovers_parameters(self, rng: np.random.Generator) -> None:
        """Fit should recover log_scale from generated data."""
        true_scale = 2.0
        y = sp_halfnorm.rvs(scale=true_scale, size=10_000, random_state=rng)
        params = HalfNormal.fit(y)
        assert params.shape == (1,)
        assert params[0] == pytest.approx(np.log(true_scale), abs=0.1)

    def test_fit_returns_finite(self, rng: np.random.Generator) -> None:
        """Fit should return finite values."""
        y = sp_halfnorm.rvs(scale=1.5, size=100, random_state=rng)
        params = HalfNormal.fit(y)
        assert np.all(np.isfinite(params))

    def test_fit_weighted(self, rng: np.random.Generator) -> None:
        """Fit with sample weights should return finite values."""
        y = sp_halfnorm.rvs(scale=1.0, size=200, random_state=rng)
        w = rng.uniform(0.5, 2.0, size=200)
        params = HalfNormal.fit(y, sample_weight=w)
        assert params.shape == (1,)
        assert np.all(np.isfinite(params))


class TestHalfNormalConstruction:
    """Tests for HalfNormal construction from params."""

    def test_from_params_shape(self) -> None:
        """HalfNormal should accept [n_samples, 1] params."""
        params = np.array([[0.0], [0.5], [-0.5]])
        dist = HalfNormal(params)
        assert len(dist) == 3
        assert dist.scale.shape == (3,)

    def test_log_scale_link(self) -> None:
        """Scale should be exp(log_scale)."""
        log_scale = 0.5
        params = np.array([[log_scale]])
        dist = HalfNormal(params)
        assert dist.scale[0] == pytest.approx(np.exp(log_scale))

    def test_getitem_slice(self) -> None:
        """Slicing should return a HalfNormal with fewer samples."""
        params = np.array([[0.0], [0.5], [1.0]])
        dist = HalfNormal(params)
        sub = dist[:2]
        assert isinstance(sub, HalfNormal)
        assert len(sub) == 2

    def test_getitem_int(self) -> None:
        """Integer indexing should return a single-sample HalfNormal."""
        params = np.array([[0.0], [0.5]])
        dist = HalfNormal(params)
        sub = dist[0]
        assert isinstance(sub, HalfNormal)
        assert len(sub) == 1


class TestHalfNormalScore:
    """Tests for HalfNormal.score (negative log-likelihood)."""

    def test_score_matches_scipy(self) -> None:
        """Score should match -scipy.logpdf."""
        params = np.array([[np.log(1.5)], [np.log(0.5)]])
        dist = HalfNormal(params)
        y = np.array([1.0, 0.3])
        expected = -sp_halfnorm.logpdf(y, scale=dist.scale)
        np.testing.assert_allclose(dist.score(y), expected, rtol=1e-10)

    def test_score_is_finite(self, rng: np.random.Generator) -> None:
        """Score should return finite values for positive y."""
        params = rng.normal(size=(50, 1))
        dist = HalfNormal(params)
        y = rng.uniform(0.01, 5.0, size=50)
        assert np.all(np.isfinite(dist.score(y)))

    def test_total_score_is_mean(self) -> None:
        """Total_score should return mean of score."""
        params = np.array([[0.0], [0.5], [-0.5]])
        dist = HalfNormal(params)
        y = np.array([0.5, 1.0, 0.2])
        expected = float(np.mean(dist.score(y)))
        assert dist.total_score(y) == pytest.approx(expected)


class TestHalfNormalGradient:
    """Tests for HalfNormal.d_score (gradient of NLL)."""

    def test_gradient_shape(self) -> None:
        """Gradient should return [n_samples, 1]."""
        params = np.array([[0.0], [0.5]])
        dist = HalfNormal(params)
        y = np.array([0.5, 1.5])
        grad = dist.d_score(y)
        assert grad.shape == (2, 1)

    def test_gradient_finite_difference(self) -> None:
        """Gradient should match numerical finite differences."""
        params = np.array([[np.log(1.5)], [np.log(0.5)]])
        y = np.array([1.0, 0.3])
        dist = HalfNormal(params)
        grad = dist.d_score(y)

        eps = 1e-5
        params_plus = params.copy()
        params_plus[:, 0] += eps
        params_minus = params.copy()
        params_minus[:, 0] -= eps
        score_plus = HalfNormal(params_plus).score(y)
        score_minus = HalfNormal(params_minus).score(y)
        numerical = (score_plus - score_minus) / (2 * eps)
        np.testing.assert_allclose(grad[:, 0], numerical, atol=1e-4)

    def test_gradient_formula(self) -> None:
        """Gradient should match hand-derived formula."""
        scale = 1.5
        params = np.array([[np.log(scale)]])
        dist = HalfNormal(params)
        y = np.array([2.0])
        grad = dist.d_score(y)
        expected = 1.0 - y[0] ** 2 / scale**2
        assert grad[0, 0] == pytest.approx(expected)


class TestHalfNormalFisherInfo:
    """Tests for HalfNormal.metric (Fisher Information)."""

    def test_metric_shape(self) -> None:
        """Metric should return [n_samples, 1, 1]."""
        params = np.array([[0.0], [0.5]])
        dist = HalfNormal(params)
        fi = dist.metric()
        assert fi.shape == (2, 1, 1)

    def test_metric_constant(self) -> None:
        """FI should be constant 2 regardless of scale."""
        params = np.array([[np.log(0.1)], [np.log(1.0)], [np.log(10.0)]])
        dist = HalfNormal(params)
        fi = dist.metric()
        for i in range(3):
            assert fi[i, 0, 0] == pytest.approx(2.0)

    def test_metric_positive_definite(self, rng: np.random.Generator) -> None:
        """FI should be positive definite for all samples."""
        params = rng.normal(size=(20, 1))
        dist = HalfNormal(params)
        fi = dist.metric()
        for i in range(20):
            assert fi[i, 0, 0] > 0


class TestHalfNormalNaturalGradient:
    """Tests for HalfNormal.natural_gradient."""

    def test_natural_gradient_shape(self) -> None:
        """Natural_gradient should return [n_samples, 1]."""
        params = np.array([[0.0], [0.5]])
        dist = HalfNormal(params)
        y = np.array([0.5, 1.5])
        ng = dist.natural_gradient(y)
        assert ng.shape == (2, 1)

    def test_natural_gradient_is_half_gradient(self) -> None:
        """Natural gradient should be d_score / 2 (FI = 2)."""
        params = np.array([[np.log(1.5)], [np.log(0.5)]])
        dist = HalfNormal(params)
        y = np.array([1.0, 0.3])
        grad = dist.d_score(y)
        ng = dist.natural_gradient(y)
        np.testing.assert_allclose(ng, grad / 2.0, rtol=1e-10)

    def test_natural_gradient_equals_base_class_default(self) -> None:
        """Fast path should match base class generic implementation."""
        params = np.array([[np.log(1.5)], [np.log(0.5)]])
        dist = HalfNormal(params)
        y = np.array([1.0, 0.3])
        fast = dist.natural_gradient(y)
        generic = Distribution.natural_gradient(dist, y)
        np.testing.assert_allclose(fast, generic, rtol=1e-10)


class TestHalfNormalSampling:
    """Tests for HalfNormal sampling, CDF, PPF, logpdf, mean."""

    def test_sample_shape(self) -> None:
        """Sample should return [n, n_samples]."""
        params = np.array([[0.0], [0.5]])
        dist = HalfNormal(params)
        s = dist.sample(100)
        assert s.shape == (100, 2)

    def test_mean_formula(self) -> None:
        """Mean should equal scale * sqrt(2/pi)."""
        scale = 2.0
        params = np.array([[np.log(scale)]])
        dist = HalfNormal(params)
        expected = scale * np.sqrt(2.0 / np.pi)
        np.testing.assert_allclose(dist.mean(), [expected], rtol=1e-10)

    def test_cdf_matches_scipy(self) -> None:
        """Cdf should match scipy."""
        scale = 1.5
        params = np.array([[np.log(scale)]])
        dist = HalfNormal(params)
        y = np.array([1.0])
        expected = sp_halfnorm.cdf(y, scale=scale)
        np.testing.assert_allclose(dist.cdf(y), expected, rtol=1e-10)

    def test_ppf_inverse_of_cdf(self) -> None:
        """Ppf(cdf(y)) should return y."""
        params = np.array([[np.log(1.5)], [np.log(0.5)]])
        dist = HalfNormal(params)
        y = np.array([1.0, 0.3])
        np.testing.assert_allclose(dist.ppf(dist.cdf(y)), y, rtol=1e-10)

    def test_logpdf_matches_scipy(self) -> None:
        """Logpdf should match scipy."""
        scale = 1.5
        params = np.array([[np.log(scale)]])
        dist = HalfNormal(params)
        y = np.array([1.0])
        expected = sp_halfnorm.logpdf(y, scale=scale)
        np.testing.assert_allclose(dist.logpdf(y), expected, rtol=1e-10)

    def test_is_subclass_of_distribution(self) -> None:
        """HalfNormal should be a Distribution subclass."""
        assert issubclass(HalfNormal, Distribution)
