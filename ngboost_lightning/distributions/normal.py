"""Normal distribution for ngboost-lightning."""

from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm as sp_norm

from ngboost_lightning.distributions.base import Distribution


class Normal(Distribution):
    """Normal (Gaussian) distribution with log-scale parameterization.

    Internal parameters are ``[mean, log_scale]`` where
    ``scale = exp(log_scale)``. The log-link for scale avoids constrained
    optimization during boosting.

    Note on Fisher Information:
        For Normal(mean, log_scale), the Fisher Information is diagonal:
        ``diag(1/scale^2, 2)``. This means the external natural gradient
        (full FI inverse) and diagonal approximation give identical results.
        This equivalence does NOT hold for non-Normal distributions.

    Attributes:
        n_params: Always 2 (mean and log_scale).
        loc: Mean values, shape ``[n_samples]``.
        scale: Standard deviation values, shape ``[n_samples]``.
        var: Variance values, shape ``[n_samples]``.
    """

    n_params = 2

    def __init__(self, params: NDArray[np.floating]) -> None:
        """Construct Normal from internal parameters.

        Args:
            params: Internal parameters, shape ``[n_samples, 2]``.
                Column 0 is mean, column 1 is log(scale).
        """
        self.loc: NDArray[np.floating] = params[:, 0]
        log_scale: NDArray[np.floating] = params[:, 1]
        self.scale: NDArray[np.floating] = np.exp(log_scale)
        self.var: NDArray[np.floating] = self.scale**2
        self._dist = sp_norm(loc=self.loc, scale=self.scale)
        self._params = params

    @staticmethod
    def fit(
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Estimate initial (mean, log_scale) from target data.

        Args:
            y: Target values, shape ``[n_samples]``.
            sample_weight: Per-sample weights, shape ``[n_samples]``.

        Returns:
            Parameter vector ``[mean, log(std)]``, shape ``[2]``.
        """
        mean = float(np.average(y, weights=sample_weight))
        var = float(np.average((y - mean) ** 2, weights=sample_weight))
        scale = max(np.sqrt(var), 1e-6)
        return np.array([mean, np.log(scale)])

    def score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample negative log-likelihood.

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            NLL values, shape ``[n_samples]``.
        """
        return -self._dist.logpdf(y)

    def d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Analytical gradient of NLL w.r.t. [mean, log_scale].

        Derivation:
            NLL = -log N(y | loc, scale)
                = 0.5*log(2*pi) + log(scale) + 0.5*((y - loc)/scale)^2

            d(NLL)/d(loc) = (loc - y) / scale^2
            d(NLL)/d(log_scale) = 1 - ((y - loc) / scale)^2
                (chain rule: d(NLL)/d(log_scale)
                 = d(NLL)/d(scale) * d(scale)/d(log_scale)
                 = d(NLL)/d(scale) * scale)

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, 2]``.
        """
        n = len(y)
        grad = np.empty((n, 2))
        grad[:, 0] = (self.loc - y) / self.var
        grad[:, 1] = 1.0 - ((self.loc - y) ** 2) / self.var
        return grad

    def metric(self) -> NDArray[np.floating]:
        """Fisher Information: diag(1/scale^2, 2) for each sample.

        Returns:
            FI tensor, shape ``[n_samples, 2, 2]``.
        """
        n = len(self.loc)
        fi = np.zeros((n, 2, 2))
        fi[:, 0, 0] = 1.0 / self.var
        fi[:, 1, 1] = 2.0
        return fi

    def natural_gradient(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Natural gradient via diagonal Fisher (fast path).

        Since the Fisher Information for Normal is diagonal, the natural
        gradient is simply element-wise division:
            nat_grad[:, 0] = d_score[:, 0] / (1/scale^2)
                           = d_score[:, 0] * scale^2
            nat_grad[:, 1] = d_score[:, 1] / 2

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Natural gradient, shape ``[n_samples, 2]``.
        """
        grad = self.d_score(y)
        nat_grad = np.empty_like(grad)
        nat_grad[:, 0] = grad[:, 0] * self.var
        nat_grad[:, 1] = grad[:, 1] / 2.0
        return nat_grad

    def mean(self) -> NDArray[np.floating]:
        """Conditional mean (point prediction).

        Returns:
            Mean values, shape ``[n_samples]``.
        """
        return self.loc

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

    # --- CRPS scoring rule ---

    def crps_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample CRPS for Normal.

        Closed form (Gneiting & Raftery 2007):
            ``CRPS = scale * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))``
        where ``z = (y - loc) / scale``.

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            CRPS values, shape ``[n_samples]``.
        """
        z = (y - self.loc) / self.scale
        result: NDArray[np.floating] = self.scale * (
            z * (2.0 * sp_norm.cdf(z) - 1.0)
            + 2.0 * sp_norm.pdf(z)
            - 1.0 / np.sqrt(np.pi)
        )
        return result

    def crps_d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Gradient of CRPS w.r.t. [mean, log_scale].

        Derivation:
            d(CRPS)/d(loc) = -(2*Phi(z) - 1)
            d(CRPS)/d(log_scale) = CRPS + (y - loc) * d(CRPS)/d(loc)
                (chain rule through ``z`` and the leading ``scale`` factor)

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, 2]``.
        """
        z = (y - self.loc) / self.scale
        n = len(y)
        grad = np.empty((n, 2))
        grad[:, 0] = -(2.0 * sp_norm.cdf(z) - 1.0)
        grad[:, 1] = self.crps_score(y) + (y - self.loc) * grad[:, 0]
        return grad

    def crps_metric(self) -> NDArray[np.floating]:
        """Riemannian metric for CRPS natural gradient.

        For Normal(loc, log_scale), the CRPS metric is:
            ``diag(1/sqrt(pi), scale^2/sqrt(pi)) / (2*sqrt(pi))``

        Simplified from NGBoost's ``NormalCRPScore.metric``.

        Returns:
            Metric tensor, shape ``[n_samples, 2, 2]``.
        """
        n = len(self.loc)
        inv_2_sqrt_pi = 1.0 / (2.0 * np.sqrt(np.pi))
        met = np.zeros((n, 2, 2))
        met[:, 0, 0] = 2.0 * inv_2_sqrt_pi
        met[:, 1, 1] = self.var * inv_2_sqrt_pi
        return met

    def crps_natural_gradient(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Natural gradient under CRPS metric (fast diagonal path).

        Since the metric is diagonal:
            nat_grad[:, 0] = d_score[:, 0] / met[0, 0]
            nat_grad[:, 1] = d_score[:, 1] / met[1, 1]

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Natural gradient, shape ``[n_samples, 2]``.
        """
        grad = self.crps_d_score(y)
        inv_2_sqrt_pi = 1.0 / (2.0 * np.sqrt(np.pi))
        nat_grad = np.empty_like(grad)
        nat_grad[:, 0] = grad[:, 0] / (2.0 * inv_2_sqrt_pi)
        nat_grad[:, 1] = grad[:, 1] / (self.var * inv_2_sqrt_pi)
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
        return len(self.loc)
