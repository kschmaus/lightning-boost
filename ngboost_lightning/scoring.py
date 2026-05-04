"""Scoring rules for natural gradient boosting.

A scoring rule maps a predicted distribution and observed target to a scalar
loss.  The engine uses the scoring rule's ``score``, ``d_score``, ``metric``,
and ``natural_gradient`` methods during the boosting loop.

Two scoring rules are provided:

* :class:`LogScore` — negative log-likelihood (default, equivalent to MLE).
* :class:`CRPScore` — Continuous Ranked Probability Score, a proper scoring
  rule that rewards well-calibrated predictive CDFs.

CRPS is only defined for continuous (and discrete-count) distributions, not
for categorical classification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ngboost_lightning.distributions.base import Distribution


@runtime_checkable
class ScoringRule(Protocol):
    """Protocol for pluggable scoring rules."""

    def score(
        self,
        dist: Distribution,
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Per-sample score.

        Args:
            dist: Predicted distribution instance.
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Score values, shape ``[n_samples]``. Lower is better.
        """
        ...

    def d_score(
        self,
        dist: Distribution,
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Gradient of score w.r.t. distribution parameters.

        Args:
            dist: Predicted distribution instance.
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, n_params]``.
        """
        ...

    def metric(
        self,
        dist: Distribution,
    ) -> NDArray[np.floating]:
        """Riemannian metric tensor for the natural gradient.

        Args:
            dist: Predicted distribution instance.

        Returns:
            Metric tensor, shape ``[n_samples, n_params, n_params]``.
        """
        ...

    def natural_gradient(
        self,
        dist: Distribution,
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Natural gradient: metric^{-1} @ d_score.

        Args:
            dist: Predicted distribution instance.
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Natural gradient, shape ``[n_samples, n_params]``.
        """
        ...

    def total_score(
        self,
        dist: Distribution,
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> float:
        """Weighted mean score across all samples.

        Args:
            dist: Predicted distribution instance.
            y: Observed target values, shape ``[n_samples]``.
            sample_weight: Per-sample weights, shape ``[n_samples]``.
                If ``None``, all samples are weighted equally.

        Returns:
            Scalar (weighted) mean score.
        """
        ...


class LogScore:
    """Negative log-likelihood scoring rule (default).

    Delegates to the distribution's existing ``score``, ``d_score``,
    ``metric``, and ``natural_gradient`` methods.  This is a zero-cost
    wrapper that preserves backward compatibility.
    """

    def score(
        self,
        dist: Distribution,
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Per-sample NLL."""
        return dist.score(y)

    def d_score(
        self,
        dist: Distribution,
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Gradient of NLL w.r.t. parameters."""
        return dist.d_score(y)

    def metric(
        self,
        dist: Distribution,
    ) -> NDArray[np.floating]:
        """Fisher Information matrix."""
        return dist.metric()

    def natural_gradient(
        self,
        dist: Distribution,
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Natural gradient using Fisher Information."""
        return dist.natural_gradient(y)

    def total_score(
        self,
        dist: Distribution,
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> float:
        """Weighted mean NLL."""
        return float(np.average(dist.score(y), weights=sample_weight))


class CRPScore:
    r"""Continuous Ranked Probability Score.

    Dispatches to ``crps_score``, ``crps_d_score``, and ``crps_metric``
    methods on each distribution.  Raises ``NotImplementedError`` if the
    distribution does not implement CRPS (e.g. Categorical).

    The CRPS for a distribution with CDF *F* and observation *y* is:

    .. math::

        CRPS(F, y) = \int_{-\infty}^{\infty} (F(x) - \mathbf{1}(x \geq y))^2 dx

    Each distribution provides a closed-form implementation.
    """

    def score(
        self,
        dist: Distribution,
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Per-sample CRPS."""
        return dist.crps_score(y)

    def d_score(
        self,
        dist: Distribution,
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Gradient of CRPS w.r.t. parameters."""
        return dist.crps_d_score(y)

    def metric(
        self,
        dist: Distribution,
    ) -> NDArray[np.floating]:
        """Riemannian metric for CRPS natural gradient."""
        return dist.crps_metric()

    def natural_gradient(
        self,
        dist: Distribution,
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Natural gradient under the CRPS metric.

        Default: solve ``metric @ nat_grad = d_score`` per sample.
        Distributions may override ``crps_natural_gradient`` for a
        fast path.
        """
        if hasattr(dist, "crps_natural_gradient"):
            ng: NDArray[np.floating] = dist.crps_natural_gradient(y)
            return ng
        grad = dist.crps_d_score(y)
        met = dist.crps_metric()
        result: NDArray[np.floating] = np.linalg.solve(met, grad[..., np.newaxis])[
            ..., 0
        ]
        return result

    def total_score(
        self,
        dist: Distribution,
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> float:
        """Weighted mean CRPS."""
        return float(np.average(dist.crps_score(y), weights=sample_weight))
