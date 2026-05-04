"""Half-Normal distribution for ngboost-lightning."""

from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.stats import halfnorm as sp_halfnorm

from ngboost_lightning.distributions.base import Distribution


class HalfNormal(Distribution):
    """Half-Normal distribution with log-scale parameterization.

    Internal parameter is ``[log_scale]`` where ``scale = exp(log_scale)``.
    The PDF is ``sqrt(2/pi) / scale * exp(-y^2 / (2*scale^2))`` for ``y >= 0``.

    Note on Fisher Information:
        For HalfNormal(log_scale), the Fisher Information is the constant
        ``[[2]]`` for every sample. This means the natural gradient is simply
        ``d_score / 2``.

    Attributes:
        n_params: Always 1 (log_scale).
        scale: Scale parameter values, shape ``[n_samples]``.
    """

    n_params = 1

    def __init__(self, params: NDArray[np.floating]) -> None:
        """Construct HalfNormal from internal parameters.

        Args:
            params: Internal parameters, shape ``[n_samples, 1]``.
                Column 0 is log(scale).
        """
        self.log_scale: NDArray[np.floating] = params[:, 0]
        self.scale: NDArray[np.floating] = np.exp(self.log_scale)
        self._dist = sp_halfnorm(scale=self.scale)
        self._params = params

    @staticmethod
    def fit(
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Estimate initial log_scale from non-negative target data.

        Uses ``scale = sqrt(E[y^2])`` which is the MLE for HalfNormal.

        Args:
            y: Target values, shape ``[n_samples]``. Should be >= 0.
            sample_weight: Per-sample weights, shape ``[n_samples]``.

        Returns:
            Parameter vector ``[log(scale)]``, shape ``[1]``.
        """
        mean_sq = float(np.average(y**2, weights=sample_weight))
        scale = max(np.sqrt(mean_sq), 1e-6)
        return np.array([np.log(scale)])

    def score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample negative log-likelihood.

        Args:
            y: Observed target values, shape ``[n_samples]``. Must be >= 0.

        Returns:
            NLL values, shape ``[n_samples]``.
        """
        return -self._dist.logpdf(y)

    def d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Analytical gradient of NLL w.r.t. [log_scale].

        Derivation:
            NLL = -log(sqrt(2/pi)) + log(scale) + y^2 / (2*scale^2)
            d(NLL)/d(scale) = 1/scale - y^2/scale^3
            d(NLL)/d(log_scale) = scale * d(NLL)/d(scale)
                = 1 - y^2/scale^2

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, 1]``.
        """
        n = len(y)
        grad = np.empty((n, 1))
        grad[:, 0] = 1.0 - y**2 / self.scale**2
        return grad

    def metric(self) -> NDArray[np.floating]:
        """Fisher Information: ``[[2]]`` for each sample.

        Derivation:
            E[(d(NLL)/d(log_scale))^2] = E[(1 - Y^2/s^2)^2]
            = 1 - 2*E[Y^2]/s^2 + E[Y^4]/s^4
            = 1 - 2 + 3 = 2

        Returns:
            FI tensor, shape ``[n_samples, 1, 1]``.
        """
        n = len(self.scale)
        fi = np.full((n, 1, 1), 2.0)
        return fi

    def natural_gradient(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Natural gradient (fast path since FI is constant 2).

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Natural gradient, shape ``[n_samples, 1]``.
        """
        return self.d_score(y) / 2.0

    def mean(self) -> NDArray[np.floating]:
        """Conditional mean: ``scale * sqrt(2/pi)``.

        Returns:
            Mean values, shape ``[n_samples]``.
        """
        result: NDArray[np.floating] = self.scale * np.sqrt(2.0 / np.pi)
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
        return len(self.scale)
