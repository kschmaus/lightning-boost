"""Unit tests for the Student's T distribution."""

import numpy as np
import pytest
from scipy.stats import t as sp_t

from ngboost_lightning.distributions.base import Distribution
from ngboost_lightning.distributions.studentt import StudentT
from ngboost_lightning.distributions.studentt import StudentT3
from ngboost_lightning.distributions.studentt import t_fixed_df


class TestTFixedDfFactory:
    """Tests for the t_fixed_df factory function."""

    def test_creates_subclass(self) -> None:
        """Factory should return a StudentT subclass."""
        cls = t_fixed_df(5)
        assert issubclass(cls, StudentT)

    def test_df_baked_in(self) -> None:
        """Created class should have the requested df."""
        cls = t_fixed_df(7)
        assert cls.df == 7.0

    def test_studentt3_has_df3(self) -> None:
        """StudentT3 convenience alias should have df=3."""
        assert StudentT3.df == 3.0

    def test_different_df_gives_different_class(self) -> None:
        """Different df values should produce distinct classes."""
        cls5 = t_fixed_df(5)
        cls10 = t_fixed_df(10)
        assert cls5 is not cls10
        assert cls5.df != cls10.df

    def test_invalid_df_raises(self) -> None:
        """Non-positive df should raise ValueError."""
        with pytest.raises(ValueError, match="df must be > 0"):
            t_fixed_df(0)
        with pytest.raises(ValueError, match="df must be > 0"):
            t_fixed_df(-1)

    def test_n_params_is_two(self) -> None:
        """All StudentT subclasses should have n_params=2."""
        cls = t_fixed_df(5)
        assert cls.n_params == 2


class TestStudentTFit:
    """Tests for StudentT.fit (initial parameter estimation)."""

    def test_fit_recovers_parameters(self, rng: np.random.Generator) -> None:
        """Fit should recover approximate (loc, log_scale) from data."""
        true_loc, true_scale, df = 3.0, 2.0, 5.0
        cls = t_fixed_df(df)
        y = sp_t.rvs(
            df=df, loc=true_loc, scale=true_scale, size=10_000, random_state=rng
        )
        params = cls.fit(y)
        assert params.shape == (2,)
        assert params[0] == pytest.approx(true_loc, abs=0.2)
        # Scale correction: fit adjusts for df > 2
        assert np.isfinite(params[1])

    def test_fit_returns_finite(self, rng: np.random.Generator) -> None:
        """Fit should return finite values."""
        y = sp_t.rvs(df=3, size=100, random_state=rng)
        params = StudentT3.fit(y)
        assert np.all(np.isfinite(params))

    def test_fit_with_weights(self, rng: np.random.Generator) -> None:
        """Fit with sample weights should return finite values."""
        y = sp_t.rvs(df=3, size=200, random_state=rng)
        w = rng.uniform(0.5, 2.0, size=200)
        params = StudentT3.fit(y, sample_weight=w)
        assert params.shape == (2,)
        assert np.all(np.isfinite(params))

    def test_fit_df_correction(self, rng: np.random.Generator) -> None:
        """Fit with df>2 should apply variance correction."""
        true_scale, df = 2.0, 10.0
        cls = t_fixed_df(df)
        y = sp_t.rvs(df=df, loc=0, scale=true_scale, size=50_000, random_state=rng)
        params = cls.fit(y)
        recovered_scale = np.exp(params[1])
        assert recovered_scale == pytest.approx(true_scale, rel=0.15)


class TestStudentTConstruction:
    """Tests for StudentT construction from params."""

    def test_from_params_shape(self) -> None:
        """StudentT should accept [n_samples, 2] params."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]])
        dist = StudentT3(params)
        assert len(dist) == 3
        assert dist.loc.shape == (3,)
        assert dist.scale.shape == (3,)

    def test_log_scale_link(self) -> None:
        """Scale should be exp(log_scale)."""
        log_scale = 0.5
        params = np.array([[0.0, log_scale]])
        dist = StudentT3(params)
        assert dist.scale[0] == pytest.approx(np.exp(log_scale))

    def test_getitem_slice(self) -> None:
        """Slicing should return a StudentT with fewer samples."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        dist = StudentT3(params)
        sub = dist[:2]
        assert isinstance(sub, StudentT)
        assert len(sub) == 2

    def test_getitem_int(self) -> None:
        """Integer indexing should return a single-sample StudentT."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = StudentT3(params)
        sub = dist[0]
        assert isinstance(sub, StudentT)
        assert len(sub) == 1


class TestStudentTScore:
    """Tests for StudentT.score (negative log-likelihood)."""

    def test_score_matches_scipy(self) -> None:
        """Score should match -scipy.logpdf."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        dist = StudentT3(params)
        y = np.array([2.5, -0.5])
        expected = -sp_t.logpdf(y, df=3, loc=dist.loc, scale=dist.scale)
        np.testing.assert_allclose(dist.score(y), expected, rtol=1e-10)

    def test_score_is_finite(self, rng: np.random.Generator) -> None:
        """Score should return finite values."""
        params = np.column_stack([rng.normal(size=50), rng.normal(size=50)])
        dist = StudentT3(params)
        y = rng.normal(size=50)
        assert np.all(np.isfinite(dist.score(y)))

    def test_total_score_is_mean(self) -> None:
        """Total_score should return mean of score."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]])
        dist = StudentT3(params)
        y = np.array([0.1, 0.9, -1.1])
        expected = float(np.mean(dist.score(y)))
        assert dist.total_score(y) == pytest.approx(expected)

    def test_score_different_df(self) -> None:
        """Score with different df should differ."""
        params = np.array([[0.0, 0.0]])
        y = np.array([1.0])
        score3 = StudentT3(params).score(y)
        StudentT10 = t_fixed_df(10)
        score10 = StudentT10(params).score(y)
        assert score3 != pytest.approx(score10)


class TestStudentTGradient:
    """Tests for StudentT.d_score (gradient of NLL)."""

    def test_gradient_shape(self) -> None:
        """Gradient should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = StudentT3(params)
        y = np.array([0.5, 1.5])
        grad = dist.d_score(y)
        assert grad.shape == (2, 2)

    def test_gradient_finite_difference(self) -> None:
        """Gradient should match numerical finite differences."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        y = np.array([2.5, -0.5])
        dist = StudentT3(params)
        grad = dist.d_score(y)

        eps = 1e-5
        for k in range(2):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps
            score_plus = StudentT3(params_plus).score(y)
            score_minus = StudentT3(params_minus).score(y)
            numerical = (score_plus - score_minus) / (2 * eps)
            np.testing.assert_allclose(grad[:, k], numerical, atol=1e-4)

    def test_gradient_zero_at_loc(self) -> None:
        """Location gradient should be zero when y == loc."""
        params = np.array([[2.0, np.log(1.5)]])
        dist = StudentT3(params)
        y = np.array([2.0])
        grad = dist.d_score(y)
        assert grad[0, 0] == pytest.approx(0.0, abs=1e-12)


class TestStudentTFisherInfo:
    """Tests for StudentT.metric (Fisher Information)."""

    def test_metric_shape(self) -> None:
        """Metric should return [n_samples, 2, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = StudentT3(params)
        fi = dist.metric()
        assert fi.shape == (2, 2, 2)

    def test_metric_diagonal(self) -> None:
        """StudentT FI should be diagonal."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = StudentT3(params)
        fi = dist.metric()
        for i in range(2):
            assert fi[i, 0, 1] == 0.0
            assert fi[i, 1, 0] == 0.0

    def test_metric_values(self) -> None:
        """FI should match known StudentT Fisher formulas."""
        v = 3.0
        scale = 1.5
        params = np.array([[0.0, np.log(scale)]])
        dist = StudentT3(params)
        fi = dist.metric()
        assert fi[0, 0, 0] == pytest.approx((v + 1) / ((v + 3) * scale**2))
        assert fi[0, 1, 1] == pytest.approx(2 * v / (v + 3))

    def test_metric_positive_definite(self, rng: np.random.Generator) -> None:
        """FI should be positive definite for all samples."""
        params = np.column_stack([rng.normal(size=20), rng.normal(size=20)])
        dist = StudentT3(params)
        fi = dist.metric()
        for i in range(20):
            eigvals = np.linalg.eigvalsh(fi[i])
            assert np.all(eigvals > 0)


class TestStudentTNaturalGradient:
    """Tests for StudentT.natural_gradient."""

    def test_natural_gradient_shape(self) -> None:
        """Natural_gradient should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = StudentT3(params)
        y = np.array([0.5, 1.5])
        ng = dist.natural_gradient(y)
        assert ng.shape == (2, 2)

    def test_natural_gradient_equals_fi_inv_grad(self) -> None:
        """Natural_gradient should equal FI^{-1} @ d_score."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        dist = StudentT3(params)
        y = np.array([2.5, -0.5])
        grad = dist.d_score(y)
        fi = dist.metric()
        expected = np.linalg.solve(fi, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(dist.natural_gradient(y), expected, rtol=1e-10)

    def test_natural_gradient_equals_base_class_default(self) -> None:
        """Fast path should match base class generic implementation."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        dist = StudentT3(params)
        y = np.array([2.5, -0.5])
        fast = dist.natural_gradient(y)
        generic = Distribution.natural_gradient(dist, y)
        np.testing.assert_allclose(fast, generic, rtol=1e-10)


class TestStudentTSampling:
    """Tests for StudentT sampling, CDF, PPF, logpdf, mean."""

    def test_sample_shape(self) -> None:
        """Sample should return [n, n_samples]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = StudentT3(params)
        s = dist.sample(100)
        assert s.shape == (100, 2)

    def test_mean_returns_loc(self) -> None:
        """Mean should return loc (for df > 1)."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        dist = StudentT3(params)
        np.testing.assert_allclose(dist.mean(), params[:, 0])

    def test_cdf_matches_scipy(self) -> None:
        """Cdf should match scipy."""
        params = np.array([[2.0, np.log(1.5)]])
        dist = StudentT3(params)
        y = np.array([3.0])
        expected = sp_t.cdf(y, df=3, loc=2.0, scale=1.5)
        np.testing.assert_allclose(dist.cdf(y), expected, rtol=1e-10)

    def test_ppf_inverse_of_cdf(self) -> None:
        """Ppf(cdf(y)) should return y."""
        params = np.array([[2.0, np.log(1.5)], [-1.0, np.log(0.5)]])
        dist = StudentT3(params)
        y = np.array([2.5, -0.5])
        np.testing.assert_allclose(dist.ppf(dist.cdf(y)), y, rtol=1e-10)

    def test_logpdf_matches_scipy(self) -> None:
        """Logpdf should match scipy."""
        params = np.array([[2.0, np.log(1.5)]])
        dist = StudentT3(params)
        y = np.array([3.0])
        expected = sp_t.logpdf(y, df=3, loc=2.0, scale=1.5)
        np.testing.assert_allclose(dist.logpdf(y), expected, rtol=1e-10)

    def test_is_subclass_of_distribution(self) -> None:
        """StudentT should be a Distribution subclass."""
        assert issubclass(StudentT, Distribution)
        assert issubclass(StudentT3, Distribution)
