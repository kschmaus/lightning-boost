"""Unit tests for the Weibull distribution."""

import numpy as np
import pytest
from scipy.special import gamma as sp_gamma_fn
from scipy.stats import weibull_min as sp_weibull

from ngboost_lightning.distributions.base import Distribution
from ngboost_lightning.distributions.weibull import Weibull


class TestWeibullFit:
    """Tests for Weibull.fit (initial parameter estimation)."""

    def test_fit_recovers_parameters(self, rng: np.random.Generator) -> None:
        """Fit should recover (log_shape, log_scale) from generated data."""
        true_shape, true_scale = 2.0, 3.0
        y = sp_weibull.rvs(
            c=true_shape, scale=true_scale, size=10_000, random_state=rng
        )
        params = Weibull.fit(y)
        assert params.shape == (2,)
        assert params[0] == pytest.approx(np.log(true_shape), abs=0.15)
        assert params[1] == pytest.approx(np.log(true_scale), abs=0.15)

    def test_fit_returns_finite(self, rng: np.random.Generator) -> None:
        """Fit should return finite values."""
        y = sp_weibull.rvs(c=1.5, scale=2.0, size=100, random_state=rng)
        params = Weibull.fit(y)
        assert np.all(np.isfinite(params))

    def test_fit_weighted(self, rng: np.random.Generator) -> None:
        """Fit with sample weights should return finite values."""
        y = sp_weibull.rvs(c=2.0, scale=1.0, size=200, random_state=rng)
        w = rng.uniform(0.5, 2.0, size=200)
        params = Weibull.fit(y, sample_weight=w)
        assert params.shape == (2,)
        assert np.all(np.isfinite(params))


class TestWeibullConstruction:
    """Tests for Weibull construction from params."""

    def test_from_params_shape(self) -> None:
        """Weibull should accept [n_samples, 2] params."""
        params = np.array([[0.5, 0.0], [1.0, 0.5], [0.0, -0.5]])
        dist = Weibull(params)
        assert len(dist) == 3
        assert dist.shape.shape == (3,)
        assert dist.scale.shape == (3,)

    def test_log_links(self) -> None:
        """Shape and scale should be exp(log_shape) and exp(log_scale)."""
        log_shape, log_scale = 0.5, 1.0
        params = np.array([[log_shape, log_scale]])
        dist = Weibull(params)
        assert dist.shape[0] == pytest.approx(np.exp(log_shape))
        assert dist.scale[0] == pytest.approx(np.exp(log_scale))

    def test_getitem_slice(self) -> None:
        """Slicing should return a Weibull with fewer samples."""
        params = np.array([[0.5, 0.0], [1.0, 0.5], [0.0, 1.0]])
        dist = Weibull(params)
        sub = dist[:2]
        assert isinstance(sub, Weibull)
        assert len(sub) == 2

    def test_getitem_int(self) -> None:
        """Integer indexing should return a single-sample Weibull."""
        params = np.array([[0.5, 0.0], [1.0, 0.5]])
        dist = Weibull(params)
        sub = dist[0]
        assert isinstance(sub, Weibull)
        assert len(sub) == 1


class TestWeibullScore:
    """Tests for Weibull.score (negative log-likelihood)."""

    def test_score_matches_scipy(self) -> None:
        """Score should match -scipy.logpdf."""
        params = np.array([[np.log(2.0), np.log(3.0)], [np.log(1.5), np.log(1.0)]])
        dist = Weibull(params)
        y = np.array([2.0, 0.5])
        expected = -sp_weibull.logpdf(y, c=dist.shape, scale=dist.scale)
        np.testing.assert_allclose(dist.score(y), expected, rtol=1e-10)

    def test_score_is_finite(self, rng: np.random.Generator) -> None:
        """Score should return finite values for positive y."""
        params = np.column_stack(
            [
                rng.uniform(0.1, 2.0, size=50),
                rng.uniform(-1.0, 1.0, size=50),
            ]
        )
        dist = Weibull(params)
        y = rng.uniform(0.1, 5.0, size=50)
        assert np.all(np.isfinite(dist.score(y)))

    def test_total_score_is_mean(self) -> None:
        """Total_score should return mean of score."""
        params = np.array([[np.log(2.0), np.log(3.0)], [np.log(1.5), np.log(1.0)]])
        dist = Weibull(params)
        y = np.array([2.0, 0.5])
        expected = float(np.mean(dist.score(y)))
        assert dist.total_score(y) == pytest.approx(expected)


class TestWeibullGradient:
    """Tests for Weibull.d_score (gradient of NLL)."""

    def test_gradient_shape(self) -> None:
        """Gradient should return [n_samples, 2]."""
        params = np.array([[np.log(2.0), np.log(3.0)]])
        dist = Weibull(params)
        y = np.array([2.5])
        grad = dist.d_score(y)
        assert grad.shape == (1, 2)

    def test_gradient_finite_difference(self) -> None:
        """Gradient should match numerical finite differences."""
        params = np.array([[np.log(2.0), np.log(3.0)], [np.log(1.5), np.log(1.0)]])
        y = np.array([2.0, 0.5])
        dist = Weibull(params)
        grad = dist.d_score(y)

        eps = 1e-5
        for k in range(2):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps
            score_plus = Weibull(params_plus).score(y)
            score_minus = Weibull(params_minus).score(y)
            numerical = (score_plus - score_minus) / (2 * eps)
            np.testing.assert_allclose(grad[:, k], numerical, atol=1e-4)

    def test_gradient_at_mode(self) -> None:
        """Gradient should be near zero at the mode for log_scale component."""
        # For Weibull with shape k > 1, mode = scale * ((k-1)/k)^(1/k)
        k, lam = 2.0, 3.0
        mode = lam * ((k - 1.0) / k) ** (1.0 / k)
        params = np.array([[np.log(k), np.log(lam)]])
        dist = Weibull(params)
        grad = dist.d_score(np.array([mode]))
        # At the mode, log_scale gradient: k*(1 - z^k) where z=mode/lam
        z = mode / lam
        expected_grad1 = k * (1.0 - z**k)
        assert grad[0, 1] == pytest.approx(expected_grad1, abs=1e-10)


class TestWeibullFisherInfo:
    """Tests for Weibull.metric (Fisher Information)."""

    def test_metric_shape(self) -> None:
        """Metric should return [n_samples, 2, 2]."""
        params = np.array([[np.log(2.0), np.log(3.0)], [np.log(1.5), np.log(1.0)]])
        dist = Weibull(params)
        fi = dist.metric()
        assert fi.shape == (2, 2, 2)

    def test_metric_non_diagonal(self) -> None:
        """Weibull FI should be non-diagonal (off-diag nonzero)."""
        params = np.array([[np.log(2.0), np.log(3.0)]])
        dist = Weibull(params)
        fi = dist.metric()
        assert fi[0, 0, 1] != 0.0
        assert fi[0, 1, 0] != 0.0

    def test_metric_symmetric(self) -> None:
        """FI should be symmetric."""
        params = np.array([[np.log(2.0), np.log(3.0)], [np.log(1.5), np.log(1.0)]])
        dist = Weibull(params)
        fi = dist.metric()
        for i in range(2):
            assert fi[i, 0, 1] == pytest.approx(fi[i, 1, 0])

    def test_metric_positive_definite(self, rng: np.random.Generator) -> None:
        """FI should be positive definite for all samples."""
        params = np.column_stack(
            [
                rng.uniform(0.1, 2.0, size=20),
                rng.uniform(-1.0, 1.0, size=20),
            ]
        )
        dist = Weibull(params)
        fi = dist.metric()
        for i in range(20):
            eigvals = np.linalg.eigvalsh(fi[i])
            assert np.all(eigvals > 0)

    def test_metric_values(self) -> None:
        """FI should match the known Weibull Fisher formulas."""
        k = 2.0
        params = np.array([[np.log(k), np.log(3.0)]])
        dist = Weibull(params)
        fi = dist.metric()
        gamma_const = 0.5772156649015329
        one_minus_gamma = 1.0 - gamma_const
        assert fi[0, 0, 0] == pytest.approx(np.pi**2 / 6.0 + one_minus_gamma**2)
        assert fi[0, 0, 1] == pytest.approx(-k * one_minus_gamma)
        assert fi[0, 1, 1] == pytest.approx(k**2)


class TestWeibullNaturalGradient:
    """Tests for Weibull natural gradient (base class solve)."""

    def test_natural_gradient_shape(self) -> None:
        """Natural_gradient should return [n_samples, 2]."""
        params = np.array([[np.log(2.0), np.log(3.0)]])
        dist = Weibull(params)
        y = np.array([2.5])
        ng = dist.natural_gradient(y)
        assert ng.shape == (1, 2)

    def test_natural_gradient_equals_fi_inv_grad(self) -> None:
        """Natural_gradient should equal FI^{-1} @ d_score."""
        params = np.array([[np.log(2.0), np.log(3.0)], [np.log(1.5), np.log(1.0)]])
        dist = Weibull(params)
        y = np.array([2.0, 0.5])
        grad = dist.d_score(y)
        fi = dist.metric()
        expected = np.linalg.solve(fi, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(dist.natural_gradient(y), expected, rtol=1e-10)


class TestWeibullSampling:
    """Tests for Weibull sampling, CDF, PPF, logpdf, mean."""

    def test_sample_shape(self) -> None:
        """Sample should return [n, n_samples]."""
        params = np.array([[np.log(2.0), np.log(3.0)], [np.log(1.5), np.log(1.0)]])
        dist = Weibull(params)
        s = dist.sample(100)
        assert s.shape == (100, 2)

    def test_mean_formula(self) -> None:
        """Mean should equal scale * Gamma(1 + 1/shape)."""
        k, lam = 2.0, 3.0
        params = np.array([[np.log(k), np.log(lam)]])
        dist = Weibull(params)
        expected = lam * sp_gamma_fn(1.0 + 1.0 / k)
        np.testing.assert_allclose(dist.mean(), [expected], rtol=1e-10)

    def test_cdf_matches_scipy(self) -> None:
        """Cdf should match scipy."""
        k, lam = 2.0, 3.0
        params = np.array([[np.log(k), np.log(lam)]])
        dist = Weibull(params)
        y = np.array([2.5])
        expected = sp_weibull.cdf(y, c=k, scale=lam)
        np.testing.assert_allclose(dist.cdf(y), expected, rtol=1e-10)

    def test_ppf_inverse_of_cdf(self) -> None:
        """Ppf(cdf(y)) should return y."""
        params = np.array([[np.log(2.0), np.log(3.0)], [np.log(1.5), np.log(1.0)]])
        dist = Weibull(params)
        y = np.array([2.0, 0.5])
        np.testing.assert_allclose(dist.ppf(dist.cdf(y)), y, rtol=1e-10)

    def test_logpdf_matches_scipy(self) -> None:
        """Logpdf should match scipy."""
        k, lam = 2.0, 3.0
        params = np.array([[np.log(k), np.log(lam)]])
        dist = Weibull(params)
        y = np.array([2.5])
        expected = sp_weibull.logpdf(y, c=k, scale=lam)
        np.testing.assert_allclose(dist.logpdf(y), expected, rtol=1e-10)

    def test_is_subclass_of_distribution(self) -> None:
        """Weibull should be a Distribution subclass."""
        assert issubclass(Weibull, Distribution)
