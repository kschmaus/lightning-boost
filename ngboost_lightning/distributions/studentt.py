"""Student's T distribution with fixed degrees of freedom for ngboost-lightning.

Provides a ``t_fixed_df(df)`` factory that creates a StudentT distribution
class with the given degrees of freedom baked in.  A convenience alias
``StudentT3 = t_fixed_df(3)`` covers the most common use case.

Internal parameters are ``[loc, log_scale]`` with ``df`` fixed at class
creation time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import t as sp_t

from ngboost_lightning.distributions.base import Distribution

if TYPE_CHECKING:
    from numpy.typing import NDArray


def t_fixed_df(df: float) -> type[StudentT]:
    """Create a StudentT distribution class with fixed degrees of freedom.

    Args:
        df: Degrees of freedom (must be > 0).

    Returns:
        A ``StudentT`` subclass with ``df`` baked in.

    Examples:
        >>> StudentT3 = t_fixed_df(3)
        >>> StudentT3.df
        3
        >>> StudentT3.n_params
        2
    """
    if df <= 0:
        msg = f"df must be > 0, got {df}"
        raise ValueError(msg)

    cls = type(
        f"StudentT_df{df}",
        (StudentT,),
        {"df": float(df)},
    )
    cls.__module__ = __name__
    return cls  # type: ignore[return-value,unused-ignore]


class StudentT(Distribution):
    """Student's T distribution with fixed df and log-scale parameterization.

    Internal parameters are ``[loc, log_scale]`` where
    ``scale = exp(log_scale)``.  The degrees of freedom ``df`` is a class
    attribute set by the ``t_fixed_df`` factory.

    Do not instantiate ``StudentT`` directly — use ``t_fixed_df(df)``
    to create a subclass with the desired df.

    Attributes:
        n_params: Always 2 (loc and log_scale).
        df: Degrees of freedom (class attribute, set by factory).
        loc: Location values, shape ``[n_samples]``.
        scale: Scale values, shape ``[n_samples]``.
    """

    n_params = 2
    df: float = 3.0  # default; overridden by factory

    def __init__(self, params: NDArray[np.floating]) -> None:
        """Construct StudentT from internal parameters.

        Args:
            params: Internal parameters, shape ``[n_samples, 2]``.
                Column 0 is loc, column 1 is log(scale).
        """
        self.loc: NDArray[np.floating] = params[:, 0]
        log_scale: NDArray[np.floating] = params[:, 1]
        self.scale: NDArray[np.floating] = np.exp(log_scale)
        self._dist = sp_t(df=self.df, loc=self.loc, scale=self.scale)
        self._params = params

    @classmethod
    def fit(
        cls,
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Estimate initial (loc, log_scale) from target data.

        Uses weighted mean for loc and a robust scale estimate from the
        weighted variance adjusted for the T distribution's heavier tails.

        Args:
            y: Target values, shape ``[n_samples]``.
            sample_weight: Per-sample weights, shape ``[n_samples]``.

        Returns:
            Parameter vector ``[loc, log(scale)]``, shape ``[2]``.
        """
        loc = float(np.average(y, weights=sample_weight))
        var = float(np.average((y - loc) ** 2, weights=sample_weight))
        # For T distribution, Var = scale^2 * df/(df-2) when df > 2
        # So scale = sqrt(var * (df-2)/df). For df <= 2, just use sqrt(var).
        v = cls.df
        if v > 2:
            scale = max(np.sqrt(var * (v - 2.0) / v), 1e-6)
        else:
            scale = max(np.sqrt(var), 1e-6)
        return np.array([loc, np.log(scale)])

    def score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample negative log-likelihood.

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            NLL values, shape ``[n_samples]``.
        """
        return -self._dist.logpdf(y)

    def d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Analytical gradient of NLL w.r.t. [loc, log_scale].

        Derivation (let v=df, s=scale, z=(y-loc)/s):
            NLL = const + log(s) + (v+1)/2 * log(1 + z^2/v)

            d(NLL)/d(loc) = (v+1) * (loc - y) / (v * s^2 + (y - loc)^2)
            d(NLL)/d(log_scale) = 1 - (v+1) * z^2 / (v + z^2)

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, 2]``.
        """
        v = self.df
        z = (y - self.loc) / self.scale
        z2 = z**2

        n = len(y)
        grad = np.empty((n, 2))
        # d/d(loc) = (v+1) * (loc - y) / (v * s^2 + (y-loc)^2)
        grad[:, 0] = (
            (v + 1.0) * (self.loc - y) / (v * self.scale**2 + (y - self.loc) ** 2)
        )
        # d/d(log_scale) = 1 - (v+1)*z^2 / (v + z^2)
        grad[:, 1] = 1.0 - (v + 1.0) * z2 / (v + z2)
        return grad

    def metric(self) -> NDArray[np.floating]:
        """Fisher Information for (loc, log_scale) parameterization.

        For T(v, loc, log_scale):
            FI[0,0] = (v+1) / ((v+3) * scale^2)
            FI[1,1] = 2*v / (v+3)
            FI is diagonal (cross-derivatives vanish by symmetry).

        Returns:
            FI tensor, shape ``[n_samples, 2, 2]``.
        """
        v = self.df
        n = len(self.loc)
        fi = np.zeros((n, 2, 2))
        fi[:, 0, 0] = (v + 1.0) / ((v + 3.0) * self.scale**2)
        fi[:, 1, 1] = 2.0 * v / (v + 3.0)
        return fi

    def natural_gradient(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Natural gradient via diagonal Fisher (fast path).

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Natural gradient, shape ``[n_samples, 2]``.
        """
        grad = self.d_score(y)
        v = self.df
        nat_grad = np.empty_like(grad)
        nat_grad[:, 0] = grad[:, 0] * (v + 3.0) * self.scale**2 / (v + 1.0)
        nat_grad[:, 1] = grad[:, 1] * (v + 3.0) / (2.0 * v)
        return nat_grad

    def mean(self) -> NDArray[np.floating]:
        """Conditional mean (= loc for T distribution when df > 1).

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

    def __getitem__(
        self,
        key: int | slice | NDArray[np.integer],
    ) -> StudentT:
        """Slice to a subset of samples."""
        sliced = self._params[key]
        if sliced.ndim == 1:
            sliced = sliced[np.newaxis, :]
        return type(self)(sliced)

    def __len__(self) -> int:
        """Number of samples in this distribution instance."""
        return len(self.loc)


# Convenience aliases
StudentT3 = t_fixed_df(3)
Cauchy = t_fixed_df(1)
