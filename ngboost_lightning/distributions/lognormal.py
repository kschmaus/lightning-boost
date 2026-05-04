"""Log-Normal distribution for ngboost-lightning."""

from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.stats import lognorm as sp_lognorm
from scipy.stats import norm as sp_norm

from ngboost_lightning.distributions.base import Distribution


class LogNormal(Distribution):
    """Log-Normal distribution with (mu, log_sigma) parameterization.

    Internal parameters are ``[mu, log_sigma]`` where the underlying normal
    has mean ``mu`` and scale ``sigma = exp(log_sigma)``. A random variable
    ``Y ~ LogNormal(mu, sigma)`` means ``log(Y) ~ Normal(mu, sigma)``.

    scipy.stats.lognorm convention: ``lognorm(s=sigma, scale=exp(mu))``.

    Note on Fisher Information:
        For LogNormal(mu, log_sigma), the Fisher Information is diagonal:
        ``diag(1/sigma^2, 2)``. This is identical to the Normal case because
        the sufficient statistics in log-space are the same.

    Attributes:
        n_params: Always 2 (mu and log_sigma).
        mu: Mean of log(Y), shape ``[n_samples]``.
        sigma: Std dev of log(Y), shape ``[n_samples]``.
        var: Variance of log(Y), shape ``[n_samples]``.
    """

    n_params = 2

    def __init__(self, params: NDArray[np.floating]) -> None:
        """Construct LogNormal from internal parameters.

        Args:
            params: Internal parameters, shape ``[n_samples, 2]``.
                Column 0 is mu, column 1 is log(sigma).
        """
        self.mu: NDArray[np.floating] = params[:, 0]
        log_sigma: NDArray[np.floating] = params[:, 1]
        self.sigma: NDArray[np.floating] = np.exp(log_sigma)
        self.var: NDArray[np.floating] = self.sigma**2
        self._dist = sp_lognorm(s=self.sigma, scale=np.exp(self.mu))
        self._params = params

    @staticmethod
    def fit(
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Estimate initial (mu, log_sigma) from target data.

        Args:
            y: Target values, shape ``[n_samples]``. Must be positive.
            sample_weight: Per-sample weights, shape ``[n_samples]``.

        Returns:
            Parameter vector ``[mu, log(sigma)]``, shape ``[2]``.
        """
        log_y = np.log(np.maximum(y, 1e-10))
        mu = float(np.average(log_y, weights=sample_weight))
        var = float(np.average((log_y - mu) ** 2, weights=sample_weight))
        sigma = max(np.sqrt(var), 1e-6)
        return np.array([mu, np.log(sigma)])

    def score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample negative log-likelihood.

        Args:
            y: Observed target values, shape ``[n_samples]``. Must be positive.

        Returns:
            NLL values, shape ``[n_samples]``.
        """
        return -self._dist.logpdf(y)

    def d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Analytical gradient of NLL w.r.t. [mu, log_sigma].

        Derivation:
            NLL = -log f(y | mu, sigma)
                = log(y) + 0.5*log(2*pi) + log(sigma) + 0.5*((log(y) - mu)/sigma)^2

            d(NLL)/d(mu) = -(log(y) - mu) / sigma^2
            d(NLL)/d(log_sigma) = 1 - ((log(y) - mu) / sigma)^2
                (chain rule: d(NLL)/d(log_sigma)
                 = d(NLL)/d(sigma) * d(sigma)/d(log_sigma)
                 = d(NLL)/d(sigma) * sigma)

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, 2]``.
        """
        n = len(y)
        grad = np.empty((n, 2))
        log_y = np.log(y)
        grad[:, 0] = -(log_y - self.mu) / self.var
        grad[:, 1] = 1.0 - ((log_y - self.mu) ** 2) / self.var
        return grad

    def metric(self) -> NDArray[np.floating]:
        """Fisher Information: diag(1/sigma^2, 2) for each sample.

        Returns:
            FI tensor, shape ``[n_samples, 2, 2]``.
        """
        n = len(self.mu)
        fi = np.zeros((n, 2, 2))
        fi[:, 0, 0] = 1.0 / self.var
        fi[:, 1, 1] = 2.0
        return fi

    def natural_gradient(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Natural gradient via diagonal Fisher (fast path).

        Since the Fisher Information for LogNormal is diagonal, the natural
        gradient is simply element-wise division:
            nat_grad[:, 0] = d_score[:, 0] / (1/sigma^2)
                           = d_score[:, 0] * sigma^2
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
            Mean values ``exp(mu + sigma^2/2)``, shape ``[n_samples]``.
        """
        return np.exp(self.mu + self.var / 2.0)

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

        Args:
            y: Values at which to evaluate.

        Returns:
            Log-survival values, same shape as ``y``.
        """
        return self._dist.logsf(y)

    # --- CRPS scoring rule ---

    def crps_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample CRPS for LogNormal.

        Since ``log(Y) ~ Normal(mu, sigma)``, the CRPS is computed in
        log-space with ``z = (log(y) - mu) / sigma``:

            ``CRPS = sigma * (z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))``

        This follows the uncensored path of NGBoost's
        ``LogNormalCRPScoreCensored``.

        Args:
            y: Observed target values, shape ``[n_samples]``. Must be positive.

        Returns:
            CRPS values, shape ``[n_samples]``.
        """
        z = (np.log(y) - self.mu) / self.sigma
        result: NDArray[np.floating] = self.sigma * (
            z * (2.0 * sp_norm.cdf(z) - 1.0)
            + 2.0 * sp_norm.pdf(z)
            - 1.0 / np.sqrt(np.pi)
        )
        return result

    def crps_d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Gradient of CRPS w.r.t. [mu, log_sigma].

        Derivation (same structure as Normal CRPS gradient in log-space):
            d(CRPS)/d(mu) = -(2*Phi(z) - 1)
            d(CRPS)/d(log_sigma) = CRPS + (log(y) - mu) * d(CRPS)/d(mu)

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, 2]``.
        """
        z = (np.log(y) - self.mu) / self.sigma
        n = len(y)
        grad = np.empty((n, 2))
        grad[:, 0] = -(2.0 * sp_norm.cdf(z) - 1.0)
        grad[:, 1] = self.crps_score(y) + (np.log(y) - self.mu) * grad[:, 0]
        return grad

    def crps_metric(self) -> NDArray[np.floating]:
        """Riemannian metric for CRPS natural gradient.

        Identical structure to Normal's CRPS metric (diagonal), with
        ``sigma`` playing the role of ``scale``:
            ``diag(2/(2*sqrt(pi)), sigma^2/(2*sqrt(pi)))``

        Returns:
            Metric tensor, shape ``[n_samples, 2, 2]``.
        """
        n = len(self.mu)
        inv_2_sqrt_pi = 1.0 / (2.0 * np.sqrt(np.pi))
        met = np.zeros((n, 2, 2))
        met[:, 0, 0] = 2.0 * inv_2_sqrt_pi
        met[:, 1, 1] = self.var * inv_2_sqrt_pi
        return met

    def crps_natural_gradient(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Natural gradient under CRPS metric (fast diagonal path).

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
        return len(self.mu)
