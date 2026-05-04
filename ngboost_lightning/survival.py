"""Survival analysis support for ngboost-lightning.

Provides right-censored data handling, the ``CensoredLogScore`` scoring
rule, and the ``Y_from_censored`` helper for creating structured target
arrays.

Censoring convention: ``event=1`` means the time is observed (uncensored),
``event=0`` means right-censored (the true event time is unknown, but
greater than the observed time).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ngboost_lightning.distributions.base import Distribution

#: Structured dtype for censored survival targets.
SURVIVAL_DTYPE = np.dtype([("Event", "?"), ("Time", "<f8")])

_EPS = 1e-10


def Y_from_censored(
    T: NDArray[np.floating],
    E: NDArray[np.integer] | NDArray[np.bool_],
) -> NDArray[np.void]:
    """Create a structured target array from times and event indicators.

    Args:
        T: Times to event or censoring, shape ``[n_samples]``.
        E: Event indicators, shape ``[n_samples]``.
            ``E[i] = 1`` (or True) means ``T[i]`` is an observed event time.
            ``E[i] = 0`` (or False) means ``T[i]`` is a right-censored time.

    Returns:
        Structured array with fields ``'Event'`` (bool) and ``'Time'``
        (float64), shape ``[n_samples]``.
    """
    T = np.asarray(T, dtype=np.float64)
    E = np.asarray(E, dtype=bool)
    n = len(T)
    Y = np.empty(n, dtype=SURVIVAL_DTYPE)
    Y["Event"] = E
    Y["Time"] = T
    return Y


def _is_censored_y(y: NDArray[np.floating] | NDArray[np.void]) -> bool:
    """Check if y is a structured censored-survival array."""
    return (
        hasattr(y, "dtype") and y.dtype.names is not None and "Event" in y.dtype.names
    )


class CensoredLogScore:
    """Censored negative log-likelihood scoring rule.

    For uncensored observations (event=1):
        loss = -logpdf(T)

    For right-censored observations (event=0):
        loss = -logsf(T)    (= -log(1 - CDF(T)))

    The metric (Fisher Information) is the same as uncensored LogScore —
    this matches NGBoost's approach.
    """

    def score(
        self,
        dist: Distribution,
        y: NDArray[np.void],
    ) -> NDArray[np.floating]:
        """Per-sample censored NLL.

        Args:
            dist: Predicted distribution instance.
            y: Structured target array with 'Event' and 'Time' fields.

        Returns:
            Censored NLL values, shape ``[n_samples]``.
        """
        E = y["Event"].astype(np.float64)
        T = y["Time"]
        uncens = dist.logpdf(T)
        cens = dist.logsf(T)
        return -(E * uncens + (1.0 - E) * cens)

    def d_score(
        self,
        dist: Distribution,
        y: NDArray[np.void],
    ) -> NDArray[np.floating]:
        """Gradient of censored NLL w.r.t. distribution parameters.

        Uses the uncensored analytical gradient for observed samples and
        numerical finite differences for censored samples. This avoids
        distribution-specific censored gradient derivations while remaining
        accurate.

        Args:
            dist: Predicted distribution instance.
            y: Structured target array with 'Event' and 'Time' fields.

        Returns:
            Gradient array, shape ``[n_samples, n_params]``.
        """
        E = y["Event"]
        T = y["Time"]
        n = len(T)
        n_params = dist._params.shape[1]

        # Uncensored gradient: standard d_score
        uncens_grad = dist.d_score(T)

        # Censored gradient via finite differences on -logsf
        eps = 1e-5
        cens_grad = np.zeros((n, n_params))
        for k in range(n_params):
            params_plus = dist._params.copy()
            params_plus[:, k] += eps
            params_minus = dist._params.copy()
            params_minus[:, k] -= eps

            logsf_plus = type(dist)(params_plus).logsf(T)
            logsf_minus = type(dist)(params_minus).logsf(T)
            cens_grad[:, k] = -(logsf_plus - logsf_minus) / (2.0 * eps)

        # Blend: E * uncens_grad + (1-E) * cens_grad
        E_f = E.astype(np.float64)[:, np.newaxis]
        return E_f * uncens_grad + (1.0 - E_f) * cens_grad

    def metric(
        self,
        dist: Distribution,
    ) -> NDArray[np.floating]:
        """Fisher Information matrix (same as uncensored).

        Args:
            dist: Predicted distribution instance.

        Returns:
            Metric tensor, shape ``[n_samples, n_params, n_params]``.
        """
        return dist.metric()

    def natural_gradient(
        self,
        dist: Distribution,
        y: NDArray[np.void],
    ) -> NDArray[np.floating]:
        """Natural gradient for censored NLL.

        Args:
            dist: Predicted distribution instance.
            y: Structured target array with 'Event' and 'Time' fields.

        Returns:
            Natural gradient, shape ``[n_samples, n_params]``.
        """
        grad = self.d_score(dist, y)
        fi = self.metric(dist)
        result: NDArray[np.floating] = np.linalg.solve(fi, grad[..., np.newaxis])[
            ..., 0
        ]
        return result

    def total_score(
        self,
        dist: Distribution,
        y: NDArray[np.void],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> float:
        """Weighted mean censored NLL.

        Args:
            dist: Predicted distribution instance.
            y: Structured target array with 'Event' and 'Time' fields.
            sample_weight: Per-sample weights.

        Returns:
            Scalar (weighted) mean censored NLL.
        """
        return float(np.average(self.score(dist, y), weights=sample_weight))
