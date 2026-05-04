"""Weibull distribution for ngboost-lightning."""

from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.stats import weibull_min as sp_weibull

from ngboost_lightning.distributions.base import Distribution

# Euler-Mascheroni constant
_EULER_GAMMA = 0.5772156649015329


class Weibull(Distribution):
    """Weibull (minimum) distribution with log-link parameterization.

    Internal parameters are ``[log_shape, log_scale]`` where
    ``shape = exp(log_shape)`` (k) and ``scale = exp(log_scale)`` (lambda).

    The Weibull PDF is
    ``f(y) = (k/lambda) * (y/lambda)^(k-1) * exp(-(y/lambda)^k)``
    for ``y >= 0``.

    Note on Fisher Information:
        For Weibull(log_shape, log_scale), the Fisher Information is
        **non-diagonal**. This distribution relies on the base class
        ``np.linalg.solve`` for the natural gradient.

    Attributes:
        n_params: Always 2 (log_shape and log_scale).
        shape: Shape parameter values (k), shape ``[n_samples]``.
        scale: Scale parameter values (lambda), shape ``[n_samples]``.
    """

    n_params = 2

    def __init__(self, params: NDArray[np.floating]) -> None:
        """Construct Weibull from internal parameters.

        Args:
            params: Internal parameters, shape ``[n_samples, 2]``.
                Column 0 is log(shape), column 1 is log(scale).
        """
        self.log_shape: NDArray[np.floating] = params[:, 0]
        self.log_scale: NDArray[np.floating] = params[:, 1]
        self.shape: NDArray[np.floating] = np.exp(self.log_shape)
        self.scale: NDArray[np.floating] = np.exp(self.log_scale)
        # scipy weibull_min(c, scale=scale) has shape param c
        self._dist = sp_weibull(c=self.shape, scale=self.scale)
        self._params = params

    @staticmethod
    def fit(
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Estimate initial (log_shape, log_scale) from positive data.

        Uses scipy's MLE for unweighted data, and a simple moment-based
        approximation for weighted data.

        Args:
            y: Target values, shape ``[n_samples]``. Must be positive.
            sample_weight: Per-sample weights, shape ``[n_samples]``.

        Returns:
            Parameter vector ``[log(shape), log(scale)]``, shape ``[2]``.
        """
        y_pos = np.maximum(y, 1e-10)
        if sample_weight is None:
            # scipy MLE: weibull_min.fit with floc=0 (fixed location)
            shape_hat, _, scale_hat = sp_weibull.fit(y_pos, floc=0)
        else:
            # Weighted moment approximation via log-moments
            w = np.asarray(sample_weight, dtype=np.float64)
            log_y = np.log(y_pos)
            mean_log = float(np.average(log_y, weights=w))
            var_log = float(np.average((log_y - mean_log) ** 2, weights=w))
            # For Weibull: Var(log Y) = pi^2 / (6 * k^2)
            shape_hat = max(np.pi / np.sqrt(6.0 * max(var_log, 1e-10)), 1e-6)
            # E[log Y] = log(lambda) - gamma/k
            scale_hat = max(np.exp(mean_log + _EULER_GAMMA / shape_hat), 1e-10)
        shape_hat = max(float(shape_hat), 1e-6)
        scale_hat = max(float(scale_hat), 1e-10)
        return np.array([np.log(shape_hat), np.log(scale_hat)])

    def score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample negative log-likelihood.

        Args:
            y: Observed target values, shape ``[n_samples]``. Must be > 0.

        Returns:
            NLL values, shape ``[n_samples]``.
        """
        return -self._dist.logpdf(y)

    def d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Analytical gradient of NLL w.r.t. [log_shape, log_scale].

        Derivation (let k=shape, lam=scale, z=y/lam):
            NLL = -log(k) + log(lam) - (k-1)*log(z) + z^k

            d(NLL)/d(k) = -1/k - log(z) + z^k * log(z)
            d(NLL)/d(log_k) = k * d(NLL)/d(k)
                = -1 + k * log(z) * (z^k - 1)

            d(NLL)/d(log_lam) = 1 + (k-1) - k * z^k
                = k * (1 - z^k)

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, 2]``.
        """
        k = self.shape
        lam = self.scale
        z = y / lam
        log_z = np.log(np.maximum(z, 1e-300))
        z_k = z**k

        n = len(y)
        grad = np.empty((n, 2))
        grad[:, 0] = -1.0 + k * log_z * (z_k - 1.0)
        grad[:, 1] = k * (1.0 - z_k)
        return grad

    def metric(self) -> NDArray[np.floating]:
        """Fisher Information for (log_shape, log_scale) parameterization.

        For Weibull with standard params (k, lam), the FI is:
            FI_kk = (pi^2/6 + (1-gamma)^2) / k^2
            FI_kl = -(1-gamma) / lam
            FI_ll = k^2 / lam^2

        Applying the Jacobian J = diag(k, lam) for log-links:
            FI_log = J^T @ FI @ J

            FI_log[0,0] = k^2 * FI_kk = pi^2/6 + (1-gamma)^2
            FI_log[0,1] = k * lam * FI_kl = -k*(1-gamma)
            FI_log[1,0] = FI_log[0,1]
            FI_log[1,1] = lam^2 * FI_ll = k^2

        Returns:
            FI tensor, shape ``[n_samples, 2, 2]``.
        """
        n = len(self.shape)
        k = self.shape
        one_minus_gamma = 1.0 - _EULER_GAMMA
        fi = np.empty((n, 2, 2))
        fi[:, 0, 0] = np.pi**2 / 6.0 + one_minus_gamma**2
        fi[:, 0, 1] = -k * one_minus_gamma
        fi[:, 1, 0] = fi[:, 0, 1]
        fi[:, 1, 1] = k**2
        return fi

    def mean(self) -> NDArray[np.floating]:
        """Conditional mean: scale * Gamma(1 + 1/shape).

        Returns:
            Mean values, shape ``[n_samples]``.
        """
        from scipy.special import gamma as sp_gamma_fn

        result: NDArray[np.floating] = self.scale * sp_gamma_fn(1.0 + 1.0 / self.shape)
        return result

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

        For Weibull(k, lam), ``logsf(y) = -(y/lam)^k``.

        Args:
            y: Values at which to evaluate.

        Returns:
            Log-survival values, same shape as ``y``.
        """
        return self._dist.logsf(y)

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
        return len(self.shape)
