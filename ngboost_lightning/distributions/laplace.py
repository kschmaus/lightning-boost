"""Laplace distribution for ngboost-lightning."""

from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.stats import laplace as sp_laplace

from ngboost_lightning.distributions.base import Distribution


class Laplace(Distribution):
    """Laplace distribution with log-scale parameterization.

    Internal parameters are ``[loc, log_scale]`` where
    ``scale = exp(log_scale)``.  The log-link for scale avoids constrained
    optimization during boosting.

    The Laplace PDF is ``f(y) = 1/(2*b) * exp(-|y - loc| / b)`` where
    ``b = scale``.

    Attributes:
        n_params: Always 2 (loc and log_scale).
        loc: Location values, shape ``[n_samples]``.
        scale: Scale (diversity) values, shape ``[n_samples]``.
    """

    n_params = 2

    def __init__(self, params: NDArray[np.floating]) -> None:
        """Construct Laplace from internal parameters.

        Args:
            params: Internal parameters, shape ``[n_samples, 2]``.
                Column 0 is loc, column 1 is log(scale).
        """
        self.loc: NDArray[np.floating] = params[:, 0]
        log_scale: NDArray[np.floating] = params[:, 1]
        self.scale: NDArray[np.floating] = np.exp(log_scale)
        self._dist = sp_laplace(loc=self.loc, scale=self.scale)
        self._params = params

    @staticmethod
    def fit(
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Estimate initial (loc, log_scale) from target data.

        Uses the weighted median for loc and weighted MAD for scale, which
        are the MLEs for Laplace.

        Args:
            y: Target values, shape ``[n_samples]``.
            sample_weight: Per-sample weights, shape ``[n_samples]``.

        Returns:
            Parameter vector ``[loc, log(scale)]``, shape ``[2]``.
        """
        if sample_weight is None:
            loc = float(np.median(y))
        else:
            # Weighted median
            w = np.asarray(sample_weight, dtype=np.float64)
            sorted_idx = np.argsort(y)
            y_sorted = y[sorted_idx]
            w_sorted = w[sorted_idx]
            cumw = np.cumsum(w_sorted)
            half = cumw[-1] / 2.0
            loc = float(y_sorted[np.searchsorted(cumw, half)])

        mad = float(np.average(np.abs(y - loc), weights=sample_weight))
        scale = max(mad, 1e-6)
        return np.array([loc, np.log(scale)])

    def score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample negative log-likelihood.

        NLL = log(2*scale) + |y - loc| / scale

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            NLL values, shape ``[n_samples]``.
        """
        return -self._dist.logpdf(y)

    def d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Analytical gradient of NLL w.r.t. [loc, log_scale].

        Derivation:
            NLL = log(2) + log(scale) + |y - loc| / scale

            d(NLL)/d(loc) = sign(loc - y) / scale
            d(NLL)/d(log_scale) = 1 - |y - loc| / scale
                (chain rule: d(NLL)/d(log_scale)
                 = d(NLL)/d(scale) * scale
                 = (-|y - loc| / scale^2) * scale
                 = 1 - |y - loc| / scale)

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, 2]``.
        """
        n = len(y)
        grad = np.empty((n, 2))
        grad[:, 0] = np.sign(self.loc - y) / self.scale
        grad[:, 1] = 1.0 - np.abs(y - self.loc) / self.scale
        return grad

    def metric(self) -> NDArray[np.floating]:
        """Fisher Information: diag(1/scale^2, 1) for each sample.

        For Laplace(loc, log_scale):
            FI[0,0] = 1/scale^2 (from the |y-loc|/scale term)
            FI[1,1] = 1 (from the log(scale) + |y-loc|/scale terms)
            FI is diagonal because the cross-derivatives vanish in
            expectation (sign(y-loc) is symmetric).

        Returns:
            FI tensor, shape ``[n_samples, 2, 2]``.
        """
        n = len(self.loc)
        fi = np.zeros((n, 2, 2))
        fi[:, 0, 0] = 1.0 / self.scale**2
        fi[:, 1, 1] = 1.0
        return fi

    def natural_gradient(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Natural gradient via diagonal Fisher (fast path).

        Since FI is diagonal:
            nat_grad[:, 0] = d_score[:, 0] * scale^2
            nat_grad[:, 1] = d_score[:, 1] / 1.0

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Natural gradient, shape ``[n_samples, 2]``.
        """
        grad = self.d_score(y)
        nat_grad = np.empty_like(grad)
        nat_grad[:, 0] = grad[:, 0] * self.scale**2
        nat_grad[:, 1] = grad[:, 1]
        return nat_grad

    def mean(self) -> NDArray[np.floating]:
        """Conditional mean (= loc for Laplace).

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
        """Per-sample CRPS for Laplace.

        Closed form:
            ``CRPS = |y - loc| + scale * exp(-|y - loc|/scale) - 0.75*scale``

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            CRPS values, shape ``[n_samples]``.
        """
        abs_diff = np.abs(y - self.loc)
        result: NDArray[np.floating] = (
            abs_diff + self.scale * np.exp(-abs_diff / self.scale) - 0.75 * self.scale
        )
        return result

    def crps_d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Gradient of CRPS w.r.t. [loc, log_scale].

        Derivation (let a = |y - loc|, b = scale):
            CRPS = a + b*exp(-a/b) - 0.75*b

            d(CRPS)/d(loc) = sign(loc - y) * (1 - exp(-a/b))
            d(CRPS)/d(log_scale) = scale * (exp(-a/b)*(1 + a/b) - 0.75)

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, 2]``.
        """
        abs_diff = np.abs(y - self.loc)
        ratio = abs_diff / self.scale
        exp_neg = np.exp(-ratio)

        n = len(y)
        grad = np.empty((n, 2))
        grad[:, 0] = np.sign(self.loc - y) * (1.0 - exp_neg)
        grad[:, 1] = self.scale * (exp_neg * (1.0 + ratio) - 0.75)
        return grad

    def crps_metric(self) -> NDArray[np.floating]:
        """Riemannian metric for CRPS natural gradient.

        For Laplace(loc, log_scale), the CRPS metric is diagonal:
            met[0,0] = 0.5 (E[1 - exp(-|y-loc|/b)]^2 under Laplace)
            met[1,1] = 0.25 * scale^2
                (from E[(exp(-a/b)*(1+a/b) - 0.75)^2] * scale^2)

        Derived from the expected outer product of crps_d_score.

        Returns:
            Metric tensor, shape ``[n_samples, 2, 2]``.
        """
        n = len(self.loc)
        met = np.zeros((n, 2, 2))
        met[:, 0, 0] = 0.5
        met[:, 1, 1] = 0.25 * self.scale**2
        return met

    def crps_natural_gradient(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Natural gradient under CRPS metric (fast diagonal path).

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Natural gradient, shape ``[n_samples, 2]``.
        """
        grad = self.crps_d_score(y)
        nat_grad = np.empty_like(grad)
        nat_grad[:, 0] = grad[:, 0] / 0.5
        nat_grad[:, 1] = grad[:, 1] / (0.25 * self.scale**2)
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
