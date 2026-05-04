"""Poisson distribution for ngboost-lightning."""

from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.special import i0e
from scipy.special import i1e
from scipy.stats import poisson as sp_poisson

from ngboost_lightning.distributions.base import Distribution


class Poisson(Distribution):
    """Poisson distribution with log-rate parameterization.

    Internal parameter is ``[log_rate]`` where ``rate = exp(log_rate)``.
    The log-link ensures rate stays positive during unconstrained boosting.

    This is a discrete distribution: score uses logPMF, not logPDF.

    Attributes:
        n_params: Always 1 (log_rate).
        rate: Poisson rate (lambda) values, shape ``[n_samples]``.
    """

    n_params = 1

    def __init__(self, params: NDArray[np.floating]) -> None:
        """Construct Poisson from internal parameters.

        Args:
            params: Internal parameters, shape ``[n_samples, 1]``.
                Column 0 is log(rate).
        """
        log_rate: NDArray[np.floating] = params[:, 0]
        self.rate: NDArray[np.floating] = np.exp(log_rate)
        self._dist = sp_poisson(mu=self.rate)
        self._params = params

    @staticmethod
    def fit(
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Estimate initial log_rate from target data.

        Args:
            y: Target values (counts), shape ``[n_samples]``.
            sample_weight: Per-sample weights, shape ``[n_samples]``.

        Returns:
            Parameter vector ``[log(rate)]``, shape ``[1]``.
        """
        rate = max(float(np.average(y, weights=sample_weight)), 1e-6)
        return np.array([np.log(rate)])

    def score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample negative log-likelihood (using logPMF).

        Args:
            y: Observed count values, shape ``[n_samples]``.

        Returns:
            NLL values, shape ``[n_samples]``.
        """
        return -self._dist.logpmf(y)

    def d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Analytical gradient of NLL w.r.t. [log_rate].

        Derivation:
            NLL = -(y * log(rate) - rate - gammaln(y + 1))
            d(NLL)/d(rate) = -(y / rate - 1) = 1 - y / rate
            d(NLL)/d(log_rate) = d(NLL)/d(rate) * d(rate)/d(log_rate)
                               = (1 - y / rate) * rate
                               = rate - y

        Args:
            y: Observed count values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, 1]``.
        """
        n = len(y)
        grad = np.empty((n, 1))
        grad[:, 0] = self.rate - y
        return grad

    def metric(self) -> NDArray[np.floating]:
        """Fisher Information for Poisson with log-rate parameterization.

        For Poisson(rate), Var(Y) = rate, and the Fisher information
        w.r.t. rate is 1/rate. Applying the chain rule for the log-rate
        parameterization: FI(log_rate) = rate^2 * (1/rate) = rate.

        Returns:
            FI tensor, shape ``[n_samples, 1, 1]``.
        """
        n = len(self.rate)
        fi = np.empty((n, 1, 1))
        fi[:, 0, 0] = self.rate
        return fi

    def natural_gradient(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Natural gradient via scalar Fisher (fast path).

        Since n_params=1, the natural gradient is simply d_score / rate:
            nat_grad[:, 0] = (rate - y) / rate = 1 - y / rate

        Args:
            y: Observed count values, shape ``[n_samples]``.

        Returns:
            Natural gradient, shape ``[n_samples, 1]``.
        """
        n = len(y)
        nat_grad = np.empty((n, 1))
        nat_grad[:, 0] = 1.0 - y / self.rate
        return nat_grad

    def mean(self) -> NDArray[np.floating]:
        """Conditional mean (point prediction).

        Returns:
            Rate values, shape ``[n_samples]``.
        """
        return self.rate

    def sample(self, n: int) -> NDArray[np.floating]:
        """Draw n samples per distribution instance.

        Args:
            n: Number of samples to draw.

        Returns:
            Samples, shape ``[n, n_samples]``.
        """
        return self._dist.rvs(size=(n, len(self)))

    def cdf(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Cumulative distribution function P(X <= y).

        Args:
            y: Values at which to evaluate the CDF.

        Returns:
            CDF values, same shape as ``y``.
        """
        return self._dist.cdf(y)

    def ppf(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        """Percent point function (inverse CDF / quantile function).

        For discrete distributions, returns the smallest integer k such
        that CDF(k) >= q.

        Args:
            q: Quantiles, values in [0, 1].

        Returns:
            Integer-valued quantiles, same shape as ``q``.
        """
        return self._dist.ppf(q)

    def logpdf(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Log probability mass function.

        Note: For ABC compatibility this method is named ``logpdf``, but
        for this discrete distribution it returns the log-PMF.

        Args:
            y: Values at which to evaluate.

        Returns:
            Log-PMF values, same shape as ``y``.
        """
        return self._dist.logpmf(y)

    # --- CRPS scoring rule ---

    def crps_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample CRPS for Poisson.

        Closed form (Czado, Gneiting & Held 2009):
            ``CRPS = (y - lam)*(2*F(y; lam) - 1)
                     + 2*lam*f(floor(y); lam)
                     - lam*exp(-2*lam)*(I_0(2*lam) + I_1(2*lam))``

        where ``lam = rate``, ``F`` is the Poisson CDF, ``f`` is the PMF,
        and ``I_0``, ``I_1`` are modified Bessel functions of the first kind.

        Uses exponentially-scaled Bessel functions (``i0e``, ``i1e``) to
        avoid overflow for large ``lam``:
            ``exp(-2*lam)*I_k(2*lam) = i_ke(2*lam)``

        Args:
            y: Observed count values, shape ``[n_samples]``.

        Returns:
            CRPS values, shape ``[n_samples]``.
        """
        lam = self.rate
        y_floor = np.floor(y)
        cdf_y = self._dist.cdf(y_floor)
        pmf_y = self._dist.pmf(y_floor)
        two_lam = 2.0 * lam
        # exp(-2*lam) * I_0(2*lam) + exp(-2*lam) * I_1(2*lam)
        bessel_term = i0e(two_lam) + i1e(two_lam)
        result: NDArray[np.floating] = (
            (y - lam) * (2.0 * cdf_y - 1.0) + 2.0 * lam * pmf_y - lam * bessel_term
        )
        return result

    def crps_d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Gradient of CRPS w.r.t. [log_rate].

        Uses central finite differences on the internal parameter.
        The analytical gradient involves derivatives of both the Poisson
        CDF and modified Bessel functions, making finite differences
        the more robust approach.

        Args:
            y: Observed count values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, 1]``.
        """
        eps = 1e-5
        params = self._params.copy()
        params_plus = params.copy()
        params_plus[:, 0] += eps
        params_minus = params.copy()
        params_minus[:, 0] -= eps
        score_plus = type(self)(params_plus).crps_score(y)
        score_minus = type(self)(params_minus).crps_score(y)
        n = len(y)
        grad = np.empty((n, 1))
        grad[:, 0] = (score_plus - score_minus) / (2.0 * eps)
        return grad

    def crps_metric(self) -> NDArray[np.floating]:
        """Riemannian metric for CRPS natural gradient (MC estimate).

        Uses a Monte Carlo estimate: sample from the distribution, compute
        CRPS gradients, and average the outer product.

        Returns:
            Metric tensor, shape ``[n_samples, 1, 1]``.
        """
        n_mc = 50
        n = len(self.rate)
        met = np.zeros((n, 1, 1))
        rng = np.random.default_rng(42)
        for _ in range(n_mc):
            y_sample = rng.poisson(self.rate).astype(np.float64)
            g = self.crps_d_score(y_sample)
            met[:, 0, 0] += g[:, 0] ** 2
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
        return len(self.rate)
