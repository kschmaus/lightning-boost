"""Categorical distribution for ngboost-lightning (binary and multiclass).

Provides a ``k_categorical(K)`` factory that creates a Categorical distribution
class with ``n_params = K - 1`` for K classes. A convenience alias
``Bernoulli = k_categorical(2)`` covers binary classification.

Internal parameters are logits ``[logit_1, ..., logit_{K-1}]`` with class 0
as the reference (logit_0 = 0). Probabilities are computed via softmax over
all K logits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import softmax

from ngboost_lightning.distributions.base import Distribution

if TYPE_CHECKING:
    from numpy.typing import NDArray


def k_categorical(n_classes: int) -> type[Categorical]:
    """Create a Categorical distribution class for K classes.

    Args:
        n_classes: Number of classes (must be >= 2).

    Returns:
        A ``Categorical`` subclass with ``n_params = K - 1``.

    Examples:
        >>> Bernoulli = k_categorical(2)
        >>> Bernoulli.n_params
        1
        >>> Cat5 = k_categorical(5)
        >>> Cat5.n_params
        4
    """
    if n_classes < 2:
        msg = f"n_classes must be >= 2, got {n_classes}"
        raise ValueError(msg)

    # Dynamically create a subclass with K and n_params baked in
    cls = type(
        f"Categorical{n_classes}",
        (Categorical,),
        {"K": n_classes, "n_params": n_classes - 1},
    )
    # Preserve module for pickling
    cls.__module__ = __name__
    return cls  # type: ignore[return-value,unused-ignore]


class Categorical(Distribution):
    """Categorical distribution with softmax-logit parameterization.

    Do not instantiate this class directly. Use ``k_categorical(K)`` to
    create a class for a specific number of classes, or use the
    ``Bernoulli`` alias for binary classification.

    Internal parameters are ``[logit_1, ..., logit_{K-1}]`` with class 0
    as the reference (``logit_0 = 0``). Probabilities are computed via
    softmax over all K logits.

    Attributes:
        K: Number of classes (set by ``k_categorical``).
        n_params: ``K - 1``.
        probs: Class probabilities, shape ``[n_samples, K]``.
        logits: Full logit vector (including reference), shape ``[n_samples, K]``.
    """

    K: int
    n_params: int

    def __init__(self, params: NDArray[np.floating]) -> None:
        """Construct Categorical from internal logit parameters.

        Args:
            params: Internal parameters, shape ``[n_samples, K-1]``.
                These are logits for classes 1..K-1 (class 0 logit is 0).
        """
        n_samples = params.shape[0]
        self._params = params

        # Full logits: prepend 0 for reference class
        self.logits: NDArray[np.floating] = np.zeros(
            (n_samples, self.K), dtype=params.dtype
        )
        self.logits[:, 1:] = params

        # Probabilities via softmax over classes (axis=1)
        self.probs: NDArray[np.floating] = softmax(self.logits, axis=1)

    @staticmethod
    def fit(
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Estimate initial logits from class labels.

        Computes (weighted) class frequencies and converts to log-odds
        relative to class 0: ``logit_k = log(p_k / p_0)`` for k=1..K-1.

        Args:
            y: Integer class labels, shape ``[n_samples]``.
            sample_weight: Per-sample weights, shape ``[n_samples]``.

        Returns:
            Logit vector, shape ``[K-1]``.
        """
        y_int = y.astype(np.intp)
        classes = np.unique(y_int)
        if sample_weight is None:
            _classes, counts = np.unique(y_int, return_counts=True)
            p = counts / len(y_int)
        else:
            w = np.asarray(sample_weight, dtype=np.float64)
            p = np.array([w[y_int == c].sum() for c in classes])
            p = p / p.sum()
        # Log-odds relative to class 0
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p = np.clip(p, eps, 1.0 - eps)
        result: NDArray[np.floating] = np.log(p[1:]) - np.log(p[0])
        return result

    def score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample categorical cross-entropy (negative log-likelihood).

        Args:
            y: Integer class labels, shape ``[n_samples]``.

        Returns:
            NLL values, shape ``[n_samples]``.
        """
        y_int = y.astype(np.intp)
        n = len(y_int)
        # -log(p_{y_i}) for each sample
        result: NDArray[np.floating] = -np.log(self.probs[np.arange(n), y_int] + 1e-15)
        return result

    def d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Gradient of NLL w.r.t. logits [logit_1, ..., logit_{K-1}].

        For the softmax-logit parameterization:
            d(NLL)/d(logit_k) = p_k - I(y == k)  for k = 1..K-1

        Args:
            y: Integer class labels, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, K-1]``.
        """
        y_int = y.astype(np.intp)

        # p_k for k=1..K-1
        grad = self.probs[:, 1:].copy()

        # Subtract indicator: I(y == k) for k=1..K-1
        for k in range(1, self.K):
            mask = y_int == k
            grad[mask, k - 1] -= 1.0

        return grad

    def metric(self) -> NDArray[np.floating]:
        """Fisher Information for softmax-logit parameterization.

        For classes 1..K-1 (excluding reference class 0):
            FI = diag(p[1:]) - p[1:] @ p[1:]^T

        This is non-diagonal for K > 2.

        Returns:
            FI tensor, shape ``[n_samples, K-1, K-1]``.
        """
        n = len(self._params)
        k = self.K - 1  # n_params
        p = self.probs[:, 1:]  # [n, K-1]

        fi = np.zeros((n, k, k))
        # Diagonal: p_k
        for j in range(k):
            fi[:, j, j] = p[:, j]
        # Off-diagonal: -p_j * p_l
        fi -= p[:, :, np.newaxis] * p[:, np.newaxis, :]

        return fi

    def natural_gradient(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Natural gradient via base class solve (non-diagonal Fisher).

        For K=2 (Bernoulli), the Fisher is scalar ``p*(1-p)`` and this
        reduces to ``(p - y) / (p * (1-p))``.

        For K>2, uses ``np.linalg.solve(FI, d_score)``.

        Args:
            y: Integer class labels, shape ``[n_samples]``.

        Returns:
            Natural gradient, shape ``[n_samples, K-1]``.
        """
        # Delegate to base class which does np.linalg.solve
        return super().natural_gradient(y)

    def mean(self) -> NDArray[np.floating]:
        """Class probabilities (the "mean" of a categorical).

        Returns:
            Probability matrix, shape ``[n_samples, K]``.

        Note:
            This returns a 2D array unlike regression distributions which
            return 1D. The classifier wrapper handles conversion to class
            labels.
        """
        return self.probs

    def sample(self, n: int) -> NDArray[np.floating]:
        """Draw n samples per distribution instance.

        Args:
            n: Number of samples to draw.

        Returns:
            Class labels, shape ``[n, n_samples]``.
        """
        n_samples = len(self._params)
        result = np.empty((n, n_samples), dtype=np.float64)
        for i in range(n_samples):
            result[:, i] = np.random.choice(self.K, size=n, p=self.probs[i])
        return result

    def cdf(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Cumulative distribution function for categorical.

        Returns the cumulative probability up to and including class y:
        ``P(Y <= y) = sum(probs[:, :y+1])``.

        Args:
            y: Integer class labels, shape ``[n_samples]``.

        Returns:
            CDF values, shape ``[n_samples]``.
        """
        y_int = y.astype(np.intp)
        n = len(y_int)
        result = np.zeros(n)
        for i in range(n):
            result[i] = self.probs[i, : y_int[i] + 1].sum()
        return result

    def ppf(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        """Percent point function (inverse CDF) for categorical.

        Returns the smallest class label k such that ``P(Y <= k) >= q``.

        Args:
            q: Quantiles, values in [0, 1], shape ``[n_samples]``.

        Returns:
            Integer class labels, shape ``[n_samples]``.
        """
        n = len(q)
        result = np.zeros(n, dtype=np.float64)
        cum_probs = np.cumsum(self.probs, axis=1)
        for i in range(n):
            result[i] = np.searchsorted(cum_probs[i], q[i])
        return result

    def logpdf(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Log probability mass function.

        Args:
            y: Integer class labels, shape ``[n_samples]``.

        Returns:
            Log-PMF values, shape ``[n_samples]``.
        """
        y_int = y.astype(np.intp)
        n = len(y_int)
        result: NDArray[np.floating] = np.log(self.probs[np.arange(n), y_int] + 1e-15)
        return result

    def __getitem__(
        self,
        key: int | slice | NDArray[np.integer],
    ) -> Categorical:
        """Slice to a subset of samples."""
        sliced = self._params[key]
        if sliced.ndim == 1:
            sliced = sliced[np.newaxis, :]
        return type(self)(sliced)

    def __len__(self) -> int:
        """Number of samples in this distribution instance."""
        return len(self._params)


Bernoulli = k_categorical(2)
"""Bernoulli distribution for binary classification.

Convenience alias for ``k_categorical(2)``. Has ``n_params = 1`` with
a single logit parameter. ``probs[:, 1]`` gives the probability of
class 1.
"""
