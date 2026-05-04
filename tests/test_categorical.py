"""Unit tests for Categorical multiclass distribution (K=3 and K=5)."""

import numpy as np
import pytest

from ngboost_lightning.distributions.categorical import Bernoulli
from ngboost_lightning.distributions.categorical import Categorical
from ngboost_lightning.distributions.categorical import k_categorical
from tests._constants import SEED


class TestCategoricalFactory:
    """Tests for the k_categorical factory function."""

    def test_k_categorical_2(self) -> None:
        """k_categorical(2) should have n_params=1 and K=2."""
        cls = k_categorical(2)
        assert cls.n_params == 1
        assert cls.K == 2

    def test_k_categorical_3(self) -> None:
        """k_categorical(3) should have n_params=2 and K=3."""
        cls = k_categorical(3)
        assert cls.n_params == 2
        assert cls.K == 3

    def test_k_categorical_5(self) -> None:
        """k_categorical(5) should have n_params=4 and K=5."""
        cls = k_categorical(5)
        assert cls.n_params == 4
        assert cls.K == 5

    def test_bernoulli_alias(self) -> None:
        """Bernoulli should be a K=2 Categorical subclass."""
        assert Bernoulli.n_params == 1
        assert Bernoulli.K == 2
        assert issubclass(Bernoulli, Categorical)

    def test_invalid_k_raises(self) -> None:
        """k_categorical(1) should raise ValueError."""
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            k_categorical(1)

    def test_different_k_different_classes(self) -> None:
        """Different K values should produce different classes."""
        assert k_categorical(3) is not k_categorical(4)


class TestCategoricalFitK3:
    """Tests for Categorical.fit with K=3."""

    def test_fit_uniform(self) -> None:
        """Uniform labels should yield logits close to zero."""
        Cat3 = k_categorical(3)
        y = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        logits = Cat3.fit(y)
        assert logits.shape == (2,)
        np.testing.assert_allclose(logits, [0.0, 0.0], atol=1e-6)

    def test_fit_skewed(self) -> None:
        """Skewed labels should yield logits reflecting class proportions."""
        Cat3 = k_categorical(3)
        # 50% class 0, 30% class 1, 20% class 2
        y = np.array([0.0] * 50 + [1.0] * 30 + [2.0] * 20)
        logits = Cat3.fit(y)
        assert logits.shape == (2,)
        # logit_k = log(p_k / p_0)
        expected_0 = np.log(0.3 / 0.5)
        expected_1 = np.log(0.2 / 0.5)
        np.testing.assert_allclose(logits[0], expected_0, atol=1e-6)
        np.testing.assert_allclose(logits[1], expected_1, atol=1e-6)

    def test_fit_returns_finite(self, rng: np.random.Generator) -> None:
        """fit() should return finite values for random labels."""
        Cat3 = k_categorical(3)
        y = rng.choice([0.0, 1.0, 2.0], size=100)
        logits = Cat3.fit(y)
        assert np.all(np.isfinite(logits))


class TestCategoricalScoreK3:
    """Tests for score (negative log-likelihood) with K=3."""

    def test_score_matches_manual(self) -> None:
        """Score should match manually computed -log(p_{y_i})."""
        Cat3 = k_categorical(3)
        params = np.array([[0.0, 0.0], [1.0, -0.5]])
        dist = Cat3(params)
        y = np.array([1.0, 2.0])
        scores = dist.score(y)
        # Manual: -log(probs[i, y_i])
        expected = -np.log(dist.probs[np.arange(2), y.astype(int)] + 1e-15)
        np.testing.assert_allclose(scores, expected, rtol=1e-10)

    def test_score_is_finite(self, rng: np.random.Generator) -> None:
        """All scores should be finite."""
        Cat3 = k_categorical(3)
        params = rng.normal(size=(20, 2)) * 0.5
        dist = Cat3(params)
        y = rng.choice([0.0, 1.0, 2.0], size=20)
        assert np.all(np.isfinite(dist.score(y)))

    def test_total_score_is_mean(self, rng: np.random.Generator) -> None:
        """total_score should be the mean of per-sample scores."""
        Cat3 = k_categorical(3)
        params = rng.normal(size=(50, 2)) * 0.5
        dist = Cat3(params)
        y = rng.choice([0.0, 1.0, 2.0], size=50)
        expected = float(np.mean(dist.score(y)))
        assert dist.total_score(y) == pytest.approx(expected)


class TestCategoricalGradientK3:
    """Tests for d_score (gradient) with K=3."""

    def test_gradient_shape(self) -> None:
        """d_score should return [n_samples, 2] for K=3."""
        Cat3 = k_categorical(3)
        params = np.array([[0.0, 0.0], [1.0, -0.5]])
        dist = Cat3(params)
        y = np.array([0.0, 1.0])
        grad = dist.d_score(y)
        assert grad.shape == (2, 2)

    def test_gradient_finite_difference(self) -> None:
        """Gradient should match finite differences."""
        rng = np.random.default_rng(SEED)
        Cat3 = k_categorical(3)
        params = rng.normal(size=(10, 2)) * 0.5
        dist = Cat3(params)
        y = rng.choice([0.0, 1.0, 2.0], size=10)
        grad = dist.d_score(y)
        eps = 1e-5
        for k in range(2):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps
            fd = (Cat3(params_plus).score(y) - Cat3(params_minus).score(y)) / (2 * eps)
            np.testing.assert_allclose(grad[:, k], fd, atol=1e-4)

    def test_gradient_formula(self) -> None:
        """d_score[:, k] should equal p_{k+1} - I(y == k+1)."""
        Cat3 = k_categorical(3)
        params = np.array([[0.5, -0.3], [0.0, 1.0], [-1.0, 0.5]])
        dist = Cat3(params)
        y = np.array([0.0, 1.0, 2.0])
        grad = dist.d_score(y)
        for k in range(2):
            expected = dist.probs[:, k + 1] - (y == (k + 1)).astype(float)
            np.testing.assert_allclose(grad[:, k], expected, rtol=1e-10)


class TestCategoricalFisherInfoK3:
    """Tests for metric (Fisher Information) with K=3."""

    def test_metric_shape(self) -> None:
        """metric() should return [n_samples, 2, 2] for K=3."""
        Cat3 = k_categorical(3)
        params = np.array([[0.0, 0.0], [1.0, -0.5]])
        dist = Cat3(params)
        fi = dist.metric()
        assert fi.shape == (2, 2, 2)

    def test_metric_is_not_diagonal(self) -> None:
        """Off-diagonal elements should be non-zero for K=3."""
        Cat3 = k_categorical(3)
        params = np.array([[0.5, -0.3]])
        dist = Cat3(params)
        fi = dist.metric()
        assert fi[0, 0, 1] != 0.0
        assert fi[0, 1, 0] != 0.0

    def test_metric_positive_definite(self, rng: np.random.Generator) -> None:
        """Fisher Information should be positive definite for all samples."""
        Cat3 = k_categorical(3)
        params = rng.normal(size=(20, 2)) * 0.5
        dist = Cat3(params)
        fi = dist.metric()
        for i in range(20):
            eigvals = np.linalg.eigvalsh(fi[i])
            assert np.all(eigvals > 0), f"FI not positive definite at sample {i}"

    def test_metric_values(self) -> None:
        """FI should equal diag(p[1:]) - p[1:] @ p[1:]^T for first sample."""
        Cat3 = k_categorical(3)
        params = np.array([[0.5, -0.3]])
        dist = Cat3(params)
        fi = dist.metric()
        p = dist.probs[0, 1:]  # [p_1, p_2]
        expected = np.diag(p) - np.outer(p, p)
        np.testing.assert_allclose(fi[0], expected, rtol=1e-10)


class TestCategoricalNaturalGradientK3:
    """Tests for natural_gradient with K=3."""

    def test_natural_gradient_shape(self) -> None:
        """natural_gradient should return [n_samples, 2] for K=3."""
        Cat3 = k_categorical(3)
        params = np.array([[0.0, 0.0], [1.0, -0.5]])
        dist = Cat3(params)
        y = np.array([0.0, 1.0])
        ng = dist.natural_gradient(y)
        assert ng.shape == (2, 2)

    def test_natural_gradient_equals_fi_inv_grad(
        self, rng: np.random.Generator
    ) -> None:
        """natural_gradient should equal FI^{-1} @ d_score."""
        Cat3 = k_categorical(3)
        params = rng.normal(size=(20, 2)) * 0.5
        dist = Cat3(params)
        y = rng.choice([0.0, 1.0, 2.0], size=20)
        ng = dist.natural_gradient(y)
        grad = dist.d_score(y)
        fi = dist.metric()
        expected = np.linalg.solve(fi, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(ng, expected, rtol=1e-10)


class TestCategoricalK5:
    """Tests for Categorical with K=5."""

    def test_gradient_finite_difference_k5(self) -> None:
        """Gradient should match finite differences for K=5."""
        rng = np.random.default_rng(SEED)
        Cat5 = k_categorical(5)
        params = rng.normal(size=(10, 4)) * 0.5
        dist = Cat5(params)
        y = rng.choice([0.0, 1.0, 2.0, 3.0, 4.0], size=10)
        grad = dist.d_score(y)
        eps = 1e-5
        for k in range(4):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps
            fd = (Cat5(params_plus).score(y) - Cat5(params_minus).score(y)) / (2 * eps)
            np.testing.assert_allclose(
                grad[:, k], fd, atol=1e-4, err_msg=f"K=5 gradient mismatch at k={k}"
            )

    def test_metric_shape_k5(self) -> None:
        """metric() should return [n_samples, 4, 4] for K=5."""
        Cat5 = k_categorical(5)
        params = np.zeros((5, 4))
        dist = Cat5(params)
        fi = dist.metric()
        assert fi.shape == (5, 4, 4)

    def test_metric_positive_definite_k5(self, rng: np.random.Generator) -> None:
        """Fisher Information should be positive definite for K=5."""
        Cat5 = k_categorical(5)
        params = rng.normal(size=(15, 4)) * 0.5
        dist = Cat5(params)
        fi = dist.metric()
        for i in range(15):
            eigvals = np.linalg.eigvalsh(fi[i])
            assert np.all(eigvals > 0), f"FI not positive definite at sample {i}"

    def test_natural_gradient_k5(self, rng: np.random.Generator) -> None:
        """natural_gradient should match solve for K=5."""
        Cat5 = k_categorical(5)
        params = rng.normal(size=(15, 4)) * 0.5
        dist = Cat5(params)
        y = rng.choice([0.0, 1.0, 2.0, 3.0, 4.0], size=15)
        ng = dist.natural_gradient(y)
        grad = dist.d_score(y)
        fi = dist.metric()
        expected = np.linalg.solve(fi, grad[..., np.newaxis])[..., 0]
        np.testing.assert_allclose(ng, expected, rtol=1e-10)


class TestCategoricalSampling:
    """Tests for sample() and statistical methods with K=3."""

    def test_sample_shape(self) -> None:
        """sample(100) should return shape [100, n_samples]."""
        Cat3 = k_categorical(3)
        params = np.array([[0.0, 0.0], [1.0, -0.5], [0.5, 0.5]])
        dist = Cat3(params)
        samples = dist.sample(100)
        assert samples.shape == (100, 3)

    def test_sample_values_valid(self) -> None:
        """All samples should be in {0, 1, 2}."""
        Cat3 = k_categorical(3)
        params = np.array([[0.0, 0.0], [1.0, -0.5]])
        dist = Cat3(params)
        samples = dist.sample(200)
        assert np.all(np.isin(samples, [0.0, 1.0, 2.0]))

    def test_cdf_at_last_class(self) -> None:
        """cdf(K-1) should equal 1.0."""
        Cat3 = k_categorical(3)
        params = np.array([[0.5, -0.3], [0.0, 1.0]])
        dist = Cat3(params)
        y = np.array([2.0, 2.0])
        np.testing.assert_allclose(dist.cdf(y), 1.0, rtol=1e-10)

    def test_ppf_returns_integers(self) -> None:
        """Ppf should return integer-valued results."""
        Cat3 = k_categorical(3)
        params = np.array([[0.0, 0.0], [1.0, -0.5]])
        dist = Cat3(params)
        q = np.array([0.5, 0.9])
        result = dist.ppf(q)
        np.testing.assert_array_equal(result, np.floor(result))

    def test_logpdf_matches_neg_score(self, rng: np.random.Generator) -> None:
        """logpdf(y) should equal -score(y)."""
        Cat3 = k_categorical(3)
        params = rng.normal(size=(20, 2)) * 0.5
        dist = Cat3(params)
        y = rng.choice([0.0, 1.0, 2.0], size=20)
        np.testing.assert_allclose(dist.logpdf(y), -dist.score(y), rtol=1e-10)

    def test_mean_returns_probs(self) -> None:
        """mean() should return probs with shape [n_samples, K]."""
        Cat3 = k_categorical(3)
        params = np.array([[0.0, 0.0], [1.0, -0.5]])
        dist = Cat3(params)
        m = dist.mean()
        assert m.shape == (2, 3)
        np.testing.assert_allclose(m, dist.probs, rtol=1e-10)
