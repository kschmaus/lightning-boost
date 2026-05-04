"""Unit tests for the Cauchy distribution (via t_fixed_df(1))."""

import numpy as np
import pytest
from scipy.stats import cauchy as sp_cauchy

from ngboost_lightning.distributions.base import Distribution
from ngboost_lightning.distributions.studentt import Cauchy
from ngboost_lightning.distributions.studentt import StudentT
from ngboost_lightning.distributions.studentt import t_fixed_df


class TestCauchyFactory:
    """Tests for the Cauchy alias and its relationship to StudentT."""

    def test_cauchy_is_studentt_subclass(self) -> None:
        """Cauchy should be a StudentT subclass."""
        assert issubclass(Cauchy, StudentT)

    def test_cauchy_has_df_one(self) -> None:
        """Cauchy should have df=1."""
        assert Cauchy.df == 1.0

    def test_cauchy_equals_t_fixed_df_1(self) -> None:
        """Cauchy should behave identically to t_fixed_df(1)."""
        T1 = t_fixed_df(1)
        params = np.array([[0.0, 0.0]])
        y = np.array([1.0])
        score_cauchy = Cauchy(params).score(y)
        score_t1 = T1(params).score(y)
        np.testing.assert_allclose(score_cauchy, score_t1)

    def test_n_params_is_two(self) -> None:
        """Cauchy should have n_params=2."""
        assert Cauchy.n_params == 2

    def test_is_subclass_of_distribution(self) -> None:
        """Cauchy should be a Distribution subclass."""
        assert issubclass(Cauchy, Distribution)


class TestCauchyFit:
    """Tests for Cauchy.fit (initial parameter estimation)."""

    def test_fit_returns_finite(self, rng: np.random.Generator) -> None:
        """Fit should return finite values despite heavy tails."""
        y = sp_cauchy.rvs(loc=0, scale=1, size=200, random_state=rng)
        params = Cauchy.fit(y)
        assert params.shape == (2,)
        assert np.all(np.isfinite(params))

    def test_fit_with_weights(self, rng: np.random.Generator) -> None:
        """Fit with sample weights should return finite values."""
        y = sp_cauchy.rvs(loc=0, scale=1, size=200, random_state=rng)
        w = rng.uniform(0.5, 2.0, size=200)
        params = Cauchy.fit(y, sample_weight=w)
        assert params.shape == (2,)
        assert np.all(np.isfinite(params))


class TestCauchyScore:
    """Tests for Cauchy.score (negative log-likelihood)."""

    def test_score_matches_scipy_cauchy(self) -> None:
        """Score should match -scipy.stats.cauchy.logpdf."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        dist = Cauchy(params)
        y = np.array([2.5, -0.5])
        expected = -sp_cauchy.logpdf(y, loc=dist.loc, scale=dist.scale)
        np.testing.assert_allclose(dist.score(y), expected, rtol=1e-10)

    def test_score_is_finite(self, rng: np.random.Generator) -> None:
        """Score should return finite values."""
        params = np.column_stack(
            [
                rng.normal(size=50),
                rng.normal(size=50),
            ]
        )
        dist = Cauchy(params)
        y = rng.normal(size=50)
        assert np.all(np.isfinite(dist.score(y)))

    def test_total_score_is_mean(self) -> None:
        """Total_score should return mean of score."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]])
        dist = Cauchy(params)
        y = np.array([0.1, 0.9, -1.1])
        expected = float(np.mean(dist.score(y)))
        assert dist.total_score(y) == pytest.approx(expected)


class TestCauchyGradient:
    """Tests for Cauchy.d_score (gradient of NLL)."""

    def test_gradient_shape(self) -> None:
        """Gradient should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Cauchy(params)
        y = np.array([0.5, 1.5])
        grad = dist.d_score(y)
        assert grad.shape == (2, 2)

    def test_gradient_finite_difference(self) -> None:
        """Gradient should match numerical finite differences."""
        params = np.array(
            [
                [2.0, np.log(1.5)],
                [-1.0, np.log(0.5)],
            ]
        )
        y = np.array([2.5, -0.5])
        dist = Cauchy(params)
        grad = dist.d_score(y)

        eps = 1e-5
        for k in range(2):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps
            score_plus = Cauchy(params_plus).score(y)
            score_minus = Cauchy(params_minus).score(y)
            numerical = (score_plus - score_minus) / (2 * eps)
            np.testing.assert_allclose(grad[:, k], numerical, atol=1e-4)

    def test_gradient_zero_at_loc(self) -> None:
        """Location gradient should be zero when y == loc."""
        params = np.array([[2.0, np.log(1.5)]])
        dist = Cauchy(params)
        y = np.array([2.0])
        grad = dist.d_score(y)
        assert grad[0, 0] == pytest.approx(0.0, abs=1e-12)


class TestCauchyFisherInfo:
    """Tests for Cauchy.metric (Fisher Information at df=1)."""

    def test_metric_shape(self) -> None:
        """Metric should return [n_samples, 2, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Cauchy(params)
        fi = dist.metric()
        assert fi.shape == (2, 2, 2)

    def test_metric_diagonal(self) -> None:
        """Cauchy FI should be diagonal (inherited from StudentT)."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Cauchy(params)
        fi = dist.metric()
        for i in range(2):
            assert fi[i, 0, 1] == 0.0
            assert fi[i, 1, 0] == 0.0

    def test_metric_values_at_df1(self) -> None:
        """FI should match T-distribution Fisher at df=1."""
        scale = 1.5
        params = np.array([[0.0, np.log(scale)]])
        dist = Cauchy(params)
        fi = dist.metric()
        # FI[0,0] = (v+1)/((v+3)*s^2) = 2/(4*s^2) = 0.5/s^2
        assert fi[0, 0, 0] == pytest.approx(0.5 / scale**2)
        # FI[1,1] = 2*v/(v+3) = 2/4 = 0.5
        assert fi[0, 1, 1] == pytest.approx(0.5)

    def test_metric_positive_definite(self, rng: np.random.Generator) -> None:
        """FI should be positive definite for all samples."""
        params = np.column_stack(
            [
                rng.normal(size=20),
                rng.normal(size=20),
            ]
        )
        dist = Cauchy(params)
        fi = dist.metric()
        for i in range(20):
            eigvals = np.linalg.eigvalsh(fi[i])
            assert np.all(eigvals > 0)


class TestCauchyNaturalGradient:
    """Tests for Cauchy.natural_gradient."""

    def test_natural_gradient_shape(self) -> None:
        """Natural_gradient should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Cauchy(params)
        y = np.array([0.5, 1.5])
        ng = dist.natural_gradient(y)
        assert ng.shape == (2, 2)

    def test_natural_gradient_equals_fi_inv_grad(self) -> None:
        """Natural_gradient should equal FI^{-1} @ d_score."""
        params = np.array(
            [
                [2.0, np.log(1.5)],
                [-1.0, np.log(0.5)],
            ]
        )
        dist = Cauchy(params)
        y = np.array([2.5, -0.5])
        grad = dist.d_score(y)
        fi = dist.metric()
        expected = np.linalg.solve(fi, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(dist.natural_gradient(y), expected, rtol=1e-10)

    def test_natural_gradient_equals_base_class(self) -> None:
        """Fast path should match base class generic implementation."""
        params = np.array(
            [
                [2.0, np.log(1.5)],
                [-1.0, np.log(0.5)],
            ]
        )
        dist = Cauchy(params)
        y = np.array([2.5, -0.5])
        fast = dist.natural_gradient(y)
        generic = Distribution.natural_gradient(dist, y)
        np.testing.assert_allclose(fast, generic, rtol=1e-10)


class TestCauchySampling:
    """Tests for Cauchy sampling, CDF, PPF, logpdf."""

    def test_sample_shape(self) -> None:
        """Sample should return [n, n_samples]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Cauchy(params)
        s = dist.sample(100)
        assert s.shape == (100, 2)

    def test_mean_returns_loc(self) -> None:
        """Mean should return loc (even though Cauchy mean is undefined)."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        dist = Cauchy(params)
        # StudentT.mean() returns loc; technically undefined for df=1
        # but as a point prediction it's still the location parameter
        np.testing.assert_allclose(dist.mean(), params[:, 0])

    def test_cdf_matches_scipy(self) -> None:
        """Cdf should match scipy.stats.cauchy."""
        params = np.array([[2.0, np.log(1.5)]])
        dist = Cauchy(params)
        y = np.array([3.0])
        expected = sp_cauchy.cdf(y, loc=2.0, scale=1.5)
        np.testing.assert_allclose(dist.cdf(y), expected, rtol=1e-10)

    def test_ppf_inverse_of_cdf(self) -> None:
        """Ppf(cdf(y)) should return y."""
        params = np.array(
            [
                [2.0, np.log(1.5)],
                [-1.0, np.log(0.5)],
            ]
        )
        dist = Cauchy(params)
        y = np.array([2.5, -0.5])
        np.testing.assert_allclose(dist.ppf(dist.cdf(y)), y, rtol=1e-10)

    def test_logpdf_matches_scipy_cauchy(self) -> None:
        """Logpdf should match scipy.stats.cauchy."""
        params = np.array([[2.0, np.log(1.5)]])
        dist = Cauchy(params)
        y = np.array([3.0])
        expected = sp_cauchy.logpdf(y, loc=2.0, scale=1.5)
        np.testing.assert_allclose(dist.logpdf(y), expected, rtol=1e-10)
