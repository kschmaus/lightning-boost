"""Exponential distribution for ngboost-lightning."""

from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.stats import expon as sp_expon

from ngboost_lightning.distributions.base import Distribution


class Exponential(Distribution):
    """Exponential distribution with log-rate parameterization.

    Internal parameter is ``[log_rate]`` where ``rate = exp(log_rate)``.
    The log-link for rate avoids constrained optimization during boosting.

    Note on Fisher Information:
        For Exponential(log_rate), the Fisher Information is ``[[1]]``
        (the identity) for every sample. This means the natural gradient
        equals the ordinary gradient.

    Attributes:
        n_params: Always 1 (log_rate).
        rate: Rate values, shape ``[n_samples]``.
    """

    n_params = 1

    def __init__(self, params: NDArray[np.floating]) -> None:
        """Construct Exponential from internal parameters.

        Args:
            params: Internal parameters, shape ``[n_samples, 1]``.
                Column 0 is log(rate).
        """
        self.log_rate: NDArray[np.floating] = params[:, 0]
        self.rate: NDArray[np.floating] = np.exp(self.log_rate)
        self._dist = sp_expon(scale=1.0 / self.rate)
        self._params = params

    @staticmethod
    def fit(
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Estimate initial log_rate from target data.

        Args:
            y: Target values, shape ``[n_samples]``.
            sample_weight: Per-sample weights, shape ``[n_samples]``.

        Returns:
            Parameter vector ``[log(rate)]``, shape ``[1]``.
        """
        rate = 1.0 / max(float(np.average(y, weights=sample_weight)), 1e-10)
        return np.array([np.log(rate)])

    def score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample negative log-likelihood.

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            NLL values, shape ``[n_samples]``.
        """
        return -self._dist.logpdf(y)

    def d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Analytical gradient of NLL w.r.t. [log_rate].

        Derivation:
            NLL = -log(rate) + rate * y
                = -log_rate + exp(log_rate) * y

            d(NLL)/d(log_rate) = -1 + exp(log_rate) * y
                               = -1 + rate * y
                               = -(1 - rate * y)

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, 1]``.
        """
        n = len(y)
        grad = np.empty((n, 1))
        grad[:, 0] = -1.0 + self.rate * y
        return grad

    def metric(self) -> NDArray[np.floating]:
        """Fisher Information: ``[[1]]`` for each sample.

        Returns:
            FI tensor, shape ``[n_samples, 1, 1]``.
        """
        n = len(self.rate)
        fi = np.ones((n, 1, 1))
        return fi

    def natural_gradient(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Natural gradient (fast path since FI is identity).

        Since the Fisher Information is the identity matrix, the natural
        gradient equals the ordinary gradient.

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Natural gradient, shape ``[n_samples, 1]``.
        """
        return self.d_score(y)

    def mean(self) -> NDArray[np.floating]:
        """Conditional mean (point prediction).

        Returns:
            Mean values ``1/rate``, shape ``[n_samples]``.
        """
        return 1.0 / self.rate

    def sample(self, n: int) -> NDArray[np.floating]:
        """Draw n samples per distribution instance.

        Args:
            n: Number of samples to draw.

        Returns:
            Samples, shape ``[n, n_samples]``.
        """
        return self._dist.rvs(size=(n, len(self)))

    def cdf(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Cumulative distribution function.

        Args:
            y: Values at which to evaluate the CDF.

        Returns:
            CDF values, same shape as ``y``.
        """
        return self._dist.cdf(y)

    def ppf(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        """Percent point function (inverse CDF / quantile function).

        Args:
            q: Quantiles, values in [0, 1].

        Returns:
            Values at the given quantiles, same shape as ``q``.
        """
        return self._dist.ppf(q)

    def logpdf(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Log probability density function.

        Args:
            y: Values at which to evaluate.

        Returns:
            Log-density values, same shape as ``y``.
        """
        return self._dist.logpdf(y)

    def logsf(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Log survival function: log(1 - CDF(y)).

        For Exponential(rate), ``logsf(y) = -rate * y``.

        Args:
            y: Values at which to evaluate.

        Returns:
            Log-survival values, same shape as ``y``.
        """
        return self._dist.logsf(y)

    # --- CRPS scoring rule ---

    def crps_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample CRPS for Exponential.

        Closed form (uncensored):
            ``CRPS = y + 2*scale*exp(-y/scale) - 1.5*scale``
        where ``scale = 1/rate``.

        Derivation:
            ``CRPS = int_0^y F(x)^2 dx + int_y^inf (1 - F(x))^2 dx``
            Part 1 gives ``y + 2*s*exp(-y/s) - s/2*exp(-2y/s) - 1.5*s``
            Part 2 gives ``s/2*exp(-2y/s)``
            The ``exp(-2y/s)`` terms cancel.

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            CRPS values, shape ``[n_samples]``.
        """
        scale = 1.0 / self.rate
        result: NDArray[np.floating] = (
            y + 2.0 * scale * np.exp(-y * self.rate) - 1.5 * scale
        )
        return result

    def crps_d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Gradient of CRPS w.r.t. [log_rate].

        Derivation:
            Let ``s = 1/rate = scale``.
                d(CRPS)/d(s) = 2*exp(-y/s)*(1 + y/s) - 1.5

            Chain rule to log_rate:
                d(s)/d(log_rate) = -s
                d(CRPS)/d(log_rate) = d(CRPS)/d(s) * (-s)

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, 1]``.
        """
        scale = 1.0 / self.rate
        exp_neg = np.exp(-y * self.rate)
        # d(CRPS)/d(scale)
        d_scale = 2.0 * exp_neg * (1.0 + y / scale) - 1.5
        # Chain to log_rate
        n = len(y)
        grad = np.empty((n, 1))
        grad[:, 0] = d_scale * (-scale)
        return grad

    def crps_metric(self) -> NDArray[np.floating]:
        """Riemannian metric for CRPS natural gradient.

        The CRPS metric in the scale parameterization is the constant
        ``29/108`` (derived analytically).  Applying the Jacobian for the
        log_rate parameterization ``ds/d(log_rate) = -s``:
            ``M_log_rate = s^2 * 29/108 = (29/108) * scale^2``

        Returns:
            Metric tensor, shape ``[n_samples, 1, 1]``.
        """
        n = len(self.rate)
        scale = 1.0 / self.rate
        met = np.empty((n, 1, 1))
        met[:, 0, 0] = (29.0 / 108.0) * scale**2
        return met

    def crps_natural_gradient(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Natural gradient under CRPS metric (scalar fast path).

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Natural gradient, shape ``[n_samples, 1]``.
        """
        grad = self.crps_d_score(y)
        scale = 1.0 / self.rate
        n = len(y)
        nat_grad = np.empty((n, 1))
        nat_grad[:, 0] = grad[:, 0] / ((29.0 / 108.0) * scale**2)
        return nat_grad

    def __getitem__(
        self,
        key: int | slice | NDArray[np.integer],
    ) -> Self:
        """Slice to a subset of samples."""
        sliced = self._params[key]
        if sliced.ndim == 1:
            sliced = sliced[np.newaxis, :]
        return type(self)(sliced)

    def __len__(self) -> int:
        """Number of samples in this distribution instance."""
        return len(self.rate)
