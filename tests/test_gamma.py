"""Unit tests for the Gamma distribution."""

import numpy as np
import pytest
from scipy.special import digamma
from scipy.special import polygamma
from scipy.stats import gamma as sp_gamma

from ngboost_lightning.distributions.gamma import Gamma


class TestGammaFit:
    """Tests for Gamma.fit (initial parameter estimation)."""

    def test_fit_recovers_parameters(self, rng: np.random.Generator) -> None:
        """fit() should recover (log_alpha, log_beta) from generated data."""
        true_alpha, true_beta = 3.0, 2.0
        y = rng.gamma(shape=true_alpha, scale=1.0 / true_beta, size=50_000)
        params = Gamma.fit(y)
        assert params.shape == (2,)
        assert params[0] == pytest.approx(np.log(true_alpha), abs=0.1)
        assert params[1] == pytest.approx(np.log(true_beta), abs=0.1)

    def test_fit_returns_finite(self, rng: np.random.Generator) -> None:
        """fit() should return finite values."""
        y = rng.gamma(shape=2.0, scale=1.0, size=100)
        params = Gamma.fit(y)
        assert np.all(np.isfinite(params))


class TestGammaConstruction:
    """Tests for Gamma construction from params."""

    def test_from_params_shape(self) -> None:
        """Gamma should accept [n_samples, 2] params."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [-0.5, 0.3]])
        dist = Gamma(params)
        assert len(dist) == 3
        assert dist.alpha.shape == (3,)
        assert dist.beta.shape == (3,)

    def test_exp_links(self) -> None:
        """Alpha and beta should be exp of internal params."""
        log_alpha, log_beta = 0.5, -0.3
        params = np.array([[log_alpha, log_beta]])
        dist = Gamma(params)
        assert dist.alpha[0] == pytest.approx(np.exp(log_alpha))
        assert dist.beta[0] == pytest.approx(np.exp(log_beta))

    def test_getitem_slice(self) -> None:
        """Slicing should return a Gamma with fewer samples."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        dist = Gamma(params)
        sliced = dist[1:]
        assert len(sliced) == 2
        assert sliced.alpha[0] == pytest.approx(np.exp(1.0))

    def test_getitem_int(self) -> None:
        """Integer indexing should return a single-sample Gamma."""
        params = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        dist = Gamma(params)
        single = dist[1]
        assert len(single) == 1
        assert single.alpha[0] == pytest.approx(np.exp(1.0))


class TestGammaScore:
    """Tests for score (negative log-likelihood)."""

    def test_score_matches_scipy(self, rng: np.random.Generator) -> None:
        """score() should match -scipy.stats.gamma.logpdf()."""
        alpha, beta = 3.0, 2.0
        params = np.tile([np.log(alpha), np.log(beta)], (100, 1))
        dist = Gamma(params)
        y = rng.gamma(shape=alpha, scale=1.0 / beta, size=100)

        expected = -sp_gamma.logpdf(y, a=alpha, scale=1.0 / beta)
        np.testing.assert_allclose(dist.score(y), expected, rtol=1e-10)

    def test_score_is_finite(self, rng: np.random.Generator) -> None:
        """NLL should always be finite for finite positive inputs."""
        params = np.tile([0.0, 0.0], (50, 1))  # alpha=1, beta=1
        dist = Gamma(params)
        y = rng.gamma(shape=1.0, scale=1.0, size=50)
        assert np.all(np.isfinite(dist.score(y)))

    def test_total_score_is_mean(self, rng: np.random.Generator) -> None:
        """total_score should be the mean of per-sample scores."""
        params = np.tile([np.log(2.0), np.log(1.0)], (100, 1))
        dist = Gamma(params)
        y = rng.gamma(shape=2.0, scale=1.0, size=100)
        expected = np.mean(dist.score(y))
        assert dist.total_score(y) == pytest.approx(float(expected))


class TestGammaGradient:
    """Tests for d_score (analytical gradient)."""

    def test_gradient_shape(self) -> None:
        """d_score should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Gamma(params)
        y = np.array([0.5, 1.5])
        grad = dist.d_score(y)
        assert grad.shape == (2, 2)

    def test_gradient_finite_difference(self, rng: np.random.Generator) -> None:
        """Analytical gradient should match finite-difference approximation."""
        n = 50
        log_alpha, log_beta = np.log(3.0), np.log(2.0)
        params = np.tile([log_alpha, log_beta], (n, 1))
        y = rng.gamma(shape=3.0, scale=0.5, size=n)

        dist = Gamma(params)
        analytical = dist.d_score(y)

        eps = 1e-5
        for k in range(2):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps

            score_plus = Gamma(params_plus).score(y)
            score_minus = Gamma(params_minus).score(y)
            numerical = (score_plus - score_minus) / (2 * eps)

            np.testing.assert_allclose(
                analytical[:, k],
                numerical,
                atol=1e-6,
                rtol=1e-4,
                err_msg=f"Gradient mismatch for parameter {k}",
            )

    def test_gradient_formula(self) -> None:
        """Verify gradient matches hand-derived formula."""
        alpha, beta = 3.0, 2.0
        params = np.array([[np.log(alpha), np.log(beta)]])
        dist = Gamma(params)
        y = np.array([1.5])

        grad = dist.d_score(y)
        expected_0 = alpha * (digamma(alpha) - np.log(beta) - np.log(y[0]))
        expected_1 = beta * y[0] - alpha

        assert grad[0, 0] == pytest.approx(expected_0)
        assert grad[0, 1] == pytest.approx(expected_1)


class TestGammaFisherInfo:
    """Tests for metric (Fisher Information)."""

    def test_metric_shape(self) -> None:
        """metric() should return [n_samples, 2, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Gamma(params)
        fi = dist.metric()
        assert fi.shape == (2, 2, 2)

    def test_metric_is_not_diagonal(self) -> None:
        """For Gamma, Fisher Information should have non-zero off-diagonals."""
        params = np.array([[np.log(3.0), np.log(2.0)]])
        dist = Gamma(params)
        fi = dist.metric()
        assert fi[0, 0, 1] != pytest.approx(0.0)
        assert fi[0, 1, 0] != pytest.approx(0.0)

    def test_metric_values(self) -> None:
        """FI should match the known formula for log-parameterization."""
        alpha, beta = 3.0, 2.0
        params = np.array([[np.log(alpha), np.log(beta)]])
        dist = Gamma(params)
        fi = dist.metric()

        trigamma = polygamma(1, alpha)
        assert fi[0, 0, 0] == pytest.approx(alpha**2 * trigamma)
        assert fi[0, 0, 1] == pytest.approx(-alpha)
        assert fi[0, 1, 0] == pytest.approx(-alpha)
        assert fi[0, 1, 1] == pytest.approx(alpha)

    def test_metric_positive_definite(self, rng: np.random.Generator) -> None:
        """Fisher Information should be positive definite for all samples."""
        n = 50
        params = np.column_stack(
            [
                rng.uniform(-1, 2, size=n),  # log_alpha
                rng.uniform(-1, 2, size=n),  # log_beta
            ]
        )
        dist = Gamma(params)
        fi = dist.metric()
        for i in range(n):
            eigvals = np.linalg.eigvalsh(fi[i])
            assert np.all(eigvals > 0), f"FI not positive definite at sample {i}"


class TestGammaNaturalGradient:
    """Tests for natural_gradient (FI^{-1} @ d_score)."""

    def test_natural_gradient_shape(self) -> None:
        """natural_gradient should return [n_samples, 2]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Gamma(params)
        y = np.array([0.5, 1.5])
        ng = dist.natural_gradient(y)
        assert ng.shape == (2, 2)

    def test_natural_gradient_equals_fi_inv_grad(
        self, rng: np.random.Generator
    ) -> None:
        """natural_gradient should equal FI^{-1} @ d_score."""
        n = 50
        params = np.column_stack(
            [
                rng.uniform(-1, 2, size=n),
                rng.uniform(-1, 2, size=n),
            ]
        )
        dist = Gamma(params)
        y = rng.gamma(shape=2.0, scale=1.0, size=n)

        ng = dist.natural_gradient(y)
        grad = dist.d_score(y)
        fi = dist.metric()

        expected = np.linalg.solve(fi, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(ng, expected, rtol=1e-10)

    def test_uses_base_class_implementation(self) -> None:
        """Gamma should NOT override natural_gradient — it uses the base class."""
        assert "natural_gradient" not in Gamma.__dict__


class TestGammaSampling:
    """Tests for sample() and statistical methods."""

    def test_sample_shape(self) -> None:
        """sample(n) should return shape [n, n_samples]."""
        params = np.array([[0.0, 0.0], [1.0, 0.5]])
        dist = Gamma(params)
        samples = dist.sample(100)
        assert samples.shape == (100, 2)

    def test_mean_formula(self) -> None:
        """mean() should return alpha / beta."""
        alpha, beta = 3.0, 2.0
        params = np.array([[np.log(alpha), np.log(beta)]])
        dist = Gamma(params)
        assert dist.mean()[0] == pytest.approx(alpha / beta)

    def test_cdf_matches_scipy(self) -> None:
        """cdf() should match scipy.stats.gamma.cdf."""
        alpha, beta = 2.0, 1.0
        params = np.array([[np.log(alpha), np.log(beta)]])
        dist = Gamma(params)
        y = np.array([2.0])
        expected = sp_gamma.cdf(y, a=alpha, scale=1.0 / beta)
        np.testing.assert_allclose(dist.cdf(y), expected, rtol=1e-10)

    def test_ppf_inverse_of_cdf(self, rng: np.random.Generator) -> None:
        """ppf(cdf(y)) should return y."""
        alpha, beta = 3.0, 2.0
        params = np.tile([np.log(alpha), np.log(beta)], (20, 1))
        dist = Gamma(params)
        y = rng.gamma(shape=alpha, scale=1.0 / beta, size=20)
        roundtrip = dist.ppf(dist.cdf(y))
        np.testing.assert_allclose(roundtrip, y, rtol=1e-10)

    def test_logpdf_matches_scipy(self) -> None:
        """logpdf() should match scipy.stats.gamma.logpdf."""
        alpha, beta = 3.0, 2.0
        params = np.array([[np.log(alpha), np.log(beta)]])
        dist = Gamma(params)
        y = np.array([1.5])
        expected = sp_gamma.logpdf(y, a=alpha, scale=1.0 / beta)
        np.testing.assert_allclose(dist.logpdf(y), expected, rtol=1e-10)
