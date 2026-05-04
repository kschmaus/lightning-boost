"""Unit tests for the Bernoulli distribution (binary categorical)."""

import numpy as np
import pytest

from ngboost_lightning.distributions.base import Distribution
from ngboost_lightning.distributions.categorical import Bernoulli
from tests._constants import SEED


class TestBernoulliFit:
    """Tests for Bernoulli.fit (initial parameter estimation)."""

    def test_fit_recovers_probability(self) -> None:
        """fit() should recover logit = log(p1/p0) from class labels."""
        y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        params = Bernoulli.fit(y)
        assert params.shape == (1,)
        # p1=0.6, p0=0.4 => logit = log(0.6/0.4)
        expected = np.log(0.6 / 0.4)
        assert params[0] == pytest.approx(expected, abs=1e-8)

    def test_fit_returns_finite(self, rng: np.random.Generator) -> None:
        """fit() should return finite values for random binary data."""
        y = rng.choice([0.0, 1.0], size=200)
        params = Bernoulli.fit(y)
        assert np.all(np.isfinite(params))


class TestBernoulliConstruction:
    """Tests for Bernoulli construction from params."""

    def test_from_params_shape(self) -> None:
        """Bernoulli should accept [n_samples, 1] params and produce [n, 2] probs."""
        params = np.array([[0.0], [1.0], [-1.0]])
        dist = Bernoulli(params)
        assert len(dist) == 3
        assert dist.probs.shape == (3, 2)

    def test_sigmoid_link(self) -> None:
        """probs[:, 1] should equal sigmoid(logit)."""
        logits = np.array([[0.0], [2.0], [-2.0]])
        dist = Bernoulli(logits)
        expected = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        np.testing.assert_allclose(dist.probs[:, 1], expected, rtol=1e-10)

    def test_probs_sum_to_one(self, rng: np.random.Generator) -> None:
        """Row probabilities should sum to 1."""
        params = rng.normal(size=(20, 1))
        dist = Bernoulli(params)
        np.testing.assert_allclose(dist.probs.sum(axis=1), 1.0, atol=1e-12)

    def test_getitem_slice(self) -> None:
        """Slicing should return a Bernoulli with fewer samples."""
        params = np.array([[0.0], [1.0], [2.0]])
        dist = Bernoulli(params)
        sliced = dist[0:2]
        assert len(sliced) == 2
        np.testing.assert_allclose(sliced.probs, dist.probs[:2], rtol=1e-12)

    def test_getitem_int(self) -> None:
        """Integer indexing should return a single-sample distribution."""
        params = np.array([[0.0], [1.0], [2.0]])
        dist = Bernoulli(params)
        single = dist[1]
        assert len(single) == 1
        np.testing.assert_allclose(single.probs, dist.probs[1:2], rtol=1e-12)


class TestBernoulliScore:
    """Tests for score (negative log-likelihood)."""

    def test_score_matches_cross_entropy(self) -> None:
        """score() should equal manually computed -log(p_{y_i})."""
        params = np.array([[1.0], [-0.5]])
        dist = Bernoulli(params)
        y = np.array([1.0, 0.0])
        p = dist.probs
        expected = np.array([-np.log(p[0, 1]), -np.log(p[1, 0])])
        np.testing.assert_allclose(dist.score(y), expected, rtol=1e-10)

    def test_score_is_finite(self, rng: np.random.Generator) -> None:
        """NLL should be finite for valid binary inputs."""
        params = rng.normal(size=(50, 1))
        dist = Bernoulli(params)
        y = rng.choice([0.0, 1.0], size=50)
        assert np.all(np.isfinite(dist.score(y)))

    def test_total_score_is_mean(self, rng: np.random.Generator) -> None:
        """total_score should be the mean of per-sample scores."""
        params = rng.normal(size=(100, 1))
        dist = Bernoulli(params)
        y = rng.choice([0.0, 1.0], size=100)
        expected = float(np.mean(dist.score(y)))
        assert dist.total_score(y) == pytest.approx(expected)


class TestBernoulliGradient:
    """Tests for d_score (analytical gradient)."""

    def test_gradient_shape(self) -> None:
        """d_score should return [n_samples, 1]."""
        params = np.array([[0.0], [1.0]])
        dist = Bernoulli(params)
        y = np.array([0.0, 1.0])
        grad = dist.d_score(y)
        assert grad.shape == (2, 1)

    def test_gradient_finite_difference(self) -> None:
        """Gradient should match finite differences."""
        rng = np.random.default_rng(SEED)
        params = rng.normal(size=(10, 1))
        dist = Bernoulli(params)
        y = rng.choice([0.0, 1.0], size=10)
        grad = dist.d_score(y)
        eps = 1e-5
        for k in range(1):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps
            score_plus = Bernoulli(params_plus).score(y)
            score_minus = Bernoulli(params_minus).score(y)
            fd = (score_plus - score_minus) / (2 * eps)
            np.testing.assert_allclose(grad[:, k], fd, atol=1e-4)

    def test_gradient_formula(self) -> None:
        """d_score[:, 0] should equal p1 - I(y==1)."""
        params = np.array([[0.5], [-1.0], [2.0]])
        dist = Bernoulli(params)
        y = np.array([1.0, 0.0, 1.0])
        grad = dist.d_score(y)
        p1 = dist.probs[:, 1]
        indicator = (y == 1.0).astype(float)
        expected = p1 - indicator
        np.testing.assert_allclose(grad[:, 0], expected, rtol=1e-10)


class TestBernoulliFisherInfo:
    """Tests for metric (Fisher Information)."""

    def test_metric_shape(self) -> None:
        """metric() should return [n_samples, 1, 1]."""
        params = np.array([[0.0], [1.0]])
        dist = Bernoulli(params)
        fi = dist.metric()
        assert fi.shape == (2, 1, 1)

    def test_metric_values(self) -> None:
        """metric[:, 0, 0] should equal p1 * (1 - p1)."""
        params = np.array([[0.0], [1.5], [-2.0]])
        dist = Bernoulli(params)
        fi = dist.metric()
        p1 = dist.probs[:, 1]
        expected = p1 * (1.0 - p1)
        np.testing.assert_allclose(fi[:, 0, 0], expected, rtol=1e-10)

    def test_metric_positive(self, rng: np.random.Generator) -> None:
        """All Fisher Information values should be positive."""
        params = rng.normal(size=(50, 1))
        dist = Bernoulli(params)
        fi = dist.metric()
        assert np.all(fi > 0)


class TestBernoulliNaturalGradient:
    """Tests for natural_gradient (FI^{-1} @ d_score)."""

    def test_natural_gradient_shape(self) -> None:
        """natural_gradient should return [n_samples, 1]."""
        params = np.array([[0.0], [1.0]])
        dist = Bernoulli(params)
        y = np.array([0.0, 1.0])
        ng = dist.natural_gradient(y)
        assert ng.shape == (2, 1)

    def test_natural_gradient_equals_fi_inv_grad(
        self,
        rng: np.random.Generator,
    ) -> None:
        """natural_gradient should equal FI^{-1} @ d_score."""
        params = rng.normal(size=(30, 1))
        dist = Bernoulli(params)
        y = rng.choice([0.0, 1.0], size=30)
        ng = dist.natural_gradient(y)
        grad = dist.d_score(y)
        fi = dist.metric()
        expected = np.linalg.solve(fi, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(ng, expected, rtol=1e-10)

    def test_natural_gradient_equals_base_class_default(
        self,
        rng: np.random.Generator,
    ) -> None:
        """Categorical delegates to base class linalg.solve; results should match."""
        params = rng.normal(size=(30, 1))
        dist = Bernoulli(params)
        y = rng.choice([0.0, 1.0], size=30)
        fast = dist.natural_gradient(y)
        base_result = Distribution.natural_gradient(dist, y)
        np.testing.assert_allclose(fast, base_result, rtol=1e-10)


class TestBernoulliSampling:
    """Tests for sample() and statistical methods."""

    def test_sample_shape(self) -> None:
        """sample(n) should return shape [n, n_samples]."""
        params = np.array([[0.0], [1.0]])
        dist = Bernoulli(params)
        samples = dist.sample(100)
        assert samples.shape == (100, 2)

    def test_sample_values_binary(self, rng: np.random.Generator) -> None:
        """All samples should be in {0, 1}."""
        params = rng.normal(size=(5, 1))
        dist = Bernoulli(params)
        samples = dist.sample(200)
        unique_vals = np.unique(samples)
        assert set(unique_vals).issubset({0.0, 1.0})

    def test_cdf(self) -> None:
        """cdf(0) should equal probs[:, 0] and cdf(1) should equal 1.0."""
        params = np.array([[0.5], [-1.0], [2.0]])
        dist = Bernoulli(params)
        cdf0 = dist.cdf(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(cdf0, dist.probs[:, 0], rtol=1e-12)
        cdf1 = dist.cdf(np.array([1.0, 1.0, 1.0]))
        np.testing.assert_allclose(cdf1, 1.0, atol=1e-12)

    def test_ppf_returns_integers(self, rng: np.random.Generator) -> None:
        """Ppf should return values in {0, 1}."""
        params = rng.normal(size=(20, 1))
        dist = Bernoulli(params)
        q = rng.uniform(0.0, 1.0, size=20)
        result = dist.ppf(q)
        unique_vals = np.unique(result)
        assert set(unique_vals).issubset({0.0, 1.0})

    def test_logpdf_matches_score(self, rng: np.random.Generator) -> None:
        """logpdf(y) should equal -score(y)."""
        params = rng.normal(size=(30, 1))
        dist = Bernoulli(params)
        y = rng.choice([0.0, 1.0], size=30)
        np.testing.assert_allclose(dist.logpdf(y), -dist.score(y), rtol=1e-10)

    def test_mean_returns_probs(self) -> None:
        """mean() should return the full probability matrix."""
        params = np.array([[0.0], [1.0], [-1.0]])
        dist = Bernoulli(params)
        np.testing.assert_array_equal(dist.mean(), dist.probs)
