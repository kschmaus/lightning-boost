"""Gamma distribution for ngboost-lightning."""

from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.special import beta as sp_beta
from scipy.special import digamma
from scipy.special import polygamma
from scipy.stats import gamma as sp_gamma

from ngboost_lightning.distributions.base import Distribution


class Gamma(Distribution):
    """Gamma distribution with log-link parameterization.

    Internal parameters are ``[log_alpha, log_beta]`` where
    ``alpha = exp(log_alpha)`` is the shape parameter and
    ``beta = exp(log_beta)`` is the rate parameter. The log-links keep
    both parameters positive during unconstrained boosting.

    Note on Fisher Information:
        For Gamma(log_alpha, log_beta), the Fisher Information is
        **non-diagonal**. The off-diagonal entries are ``-alpha``.
        This distribution relies on the base class ``np.linalg.solve``
        for the natural gradient — no fast-path override.

    Attributes:
        n_params: Always 2 (log_alpha and log_beta).
        alpha: Shape parameter values, shape ``[n_samples]``.
        beta: Rate parameter values, shape ``[n_samples]``.
    """

    n_params = 2

    def __init__(self, params: NDArray[np.floating]) -> None:
        """Construct Gamma from internal parameters.

        Args:
            params: Internal parameters, shape ``[n_samples, 2]``.
                Column 0 is log(alpha), column 1 is log(beta).
        """
        self.log_alpha: NDArray[np.floating] = params[:, 0]
        self.log_beta: NDArray[np.floating] = params[:, 1]
        self.alpha: NDArray[np.floating] = np.exp(self.log_alpha)
        self.beta: NDArray[np.floating] = np.exp(self.log_beta)
        self._dist = sp_gamma(a=self.alpha, scale=1.0 / self.beta)
        self._params = params

    @staticmethod
    def fit(
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Estimate initial (log_alpha, log_beta) via method of moments.

        Args:
            y: Target values, shape ``[n_samples]``. Must be positive.
            sample_weight: Per-sample weights, shape ``[n_samples]``.

        Returns:
            Parameter vector ``[log(alpha), log(beta)]``, shape ``[2]``.
        """
        mean_y = float(np.average(y, weights=sample_weight))
        var_y = float(np.average((y - mean_y) ** 2, weights=sample_weight))
        beta = np.maximum(mean_y / max(var_y, 1e-10), 1e-6)
        alpha = np.maximum(mean_y * beta, 1e-6)
        return np.array([np.log(alpha), np.log(beta)])

    def score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample negative log-likelihood.

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            NLL values, shape ``[n_samples]``.
        """
        return -self._dist.logpdf(y)

    def d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Analytical gradient of NLL w.r.t. [log_alpha, log_beta].

        Derivation:
            NLL = -(alpha-1)*log(y) + beta*y - alpha*log(beta) + gammaln(alpha)

            d(NLL)/d(alpha) = -log(y) - log(beta) + digamma(alpha)
            d(NLL)/d(log_alpha) = d(NLL)/d(alpha) * alpha
                = alpha * (digamma(alpha) - log(beta) - log(y))

            d(NLL)/d(beta) = y - alpha/beta
            d(NLL)/d(log_beta) = d(NLL)/d(beta) * beta
                = beta*y - alpha

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, 2]``.
        """
        n = len(y)
        grad = np.empty((n, 2))
        grad[:, 0] = self.alpha * (digamma(self.alpha) - self.log_beta - np.log(y))
        grad[:, 1] = self.beta * y - self.alpha
        return grad

    def metric(self) -> NDArray[np.floating]:
        """Fisher Information for (log_alpha, log_beta) parameterization.

        The standard-parameterization Fisher Information for Gamma(alpha, beta)
        is::

            FI = [[trigamma(alpha), -1 / beta], [-1 / beta, alpha / beta ^ 2]]

        Applying the Jacobian ``J = diag(alpha, beta)`` for the log-link::

            FI_log = J ^ T @ FI @ J

        gives::

            FI_log = [[alpha ^ 2 * trigamma(alpha), -alpha], [-alpha, alpha]]

        Returns:
            FI tensor, shape ``[n_samples, 2, 2]``.
        """
        n = len(self.alpha)
        fi = np.empty((n, 2, 2))
        trigamma = polygamma(1, self.alpha)
        fi[:, 0, 0] = self.alpha**2 * trigamma
        fi[:, 0, 1] = -self.alpha
        fi[:, 1, 0] = -self.alpha
        fi[:, 1, 1] = self.alpha
        return fi

    def mean(self) -> NDArray[np.floating]:
        """Conditional mean (point prediction): alpha / beta.

        Returns:
            Mean values, shape ``[n_samples]``.
        """
        return self.alpha / self.beta

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
        """Per-sample CRPS for Gamma.

        Closed form (Scheuerer & Hamill 2015):
            ``CRPS = y*(2*F(y;a,b) - 1)
                     - (a/b)*(2*F(y;a+1,b) - 1)
                     - 1/(b * Beta(a, 0.5))``

        where ``a = alpha`` (shape), ``b = beta`` (rate),
        ``F(y;a,b)`` is the Gamma CDF with shape ``a`` and rate ``b``.

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            CRPS values, shape ``[n_samples]``.
        """
        a = self.alpha
        b = self.beta
        # F(y; shape=a, rate=b) = F(y; shape=a, scale=1/b)
        cdf_a = sp_gamma.cdf(y, a=a, scale=1.0 / b)
        cdf_a1 = sp_gamma.cdf(y, a=a + 1.0, scale=1.0 / b)
        return (
            y * (2.0 * cdf_a - 1.0)
            - (a / b) * (2.0 * cdf_a1 - 1.0)
            - 1.0 / (b * sp_beta(a, 0.5))
        )

    def crps_d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Gradient of CRPS w.r.t. [log_alpha, log_beta].

        Uses central finite differences on the internal parameters for
        robustness.  The analytical gradient of the Gamma CDF w.r.t. the
        shape parameter involves the derivative of the regularized
        incomplete gamma function, which is numerically delicate.  Finite
        differences are accurate to ``O(eps^2)`` and fast enough for
        the boosting loop since CRPS evaluation is cheap.

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, 2]``.
        """
        eps = 1e-5
        n = len(y)
        grad = np.empty((n, 2))
        params = self._params.copy()

        for k in range(2):
            params_plus = params.copy()
            params_plus[:, k] += eps
            params_minus = params.copy()
            params_minus[:, k] -= eps
            score_plus = type(self)(params_plus).crps_score(y)
            score_minus = type(self)(params_minus).crps_score(y)
            grad[:, k] = (score_plus - score_minus) / (2.0 * eps)

        return grad

    def crps_metric(self) -> NDArray[np.floating]:
        """Riemannian metric for CRPS natural gradient (MC estimate).

        Uses a Monte Carlo estimate: sample from the distribution, compute
        CRPS gradients, and average the outer product.

        Returns:
            Metric tensor, shape ``[n_samples, 2, 2]``.
        """
        n_mc = 50
        n = len(self.alpha)
        met = np.zeros((n, 2, 2))
        rng = np.random.default_rng(42)
        for _ in range(n_mc):
            y_sample = rng.gamma(self.alpha, 1.0 / self.beta)
            g = self.crps_d_score(y_sample)
            # Outer product: g[:, :, None] @ g[:, None, :]
            met += np.einsum("ij,ik->ijk", g, g)
        met /= n_mc
        return met

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
        return len(self.alpha)
