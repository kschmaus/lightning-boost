"""Base distribution abstraction for ngboost-lightning."""

from abc import ABC
from abc import abstractmethod
from typing import Self

import numpy as np
from numpy.typing import NDArray


class Distribution(ABC):
    """Abstract base class for probability distributions.

    A Distribution represents a parametric family of distributions. Instances
    hold arrays of parameters (one set per sample) and provide methods for
    scoring, gradient computation, and sampling.

    Subclasses must define:
        - ``n_params``: number of internal parameters
        - ``fit(y)``: estimate initial parameters from data
        - ``score(y)``: per-sample negative log-likelihood
        - ``d_score(y)``: gradient of score w.r.t. internal params
        - ``metric()``: Fisher Information matrix
        - ``mean()``: point prediction (conditional mean)
        - ``sample(n)``: draw random samples
        - ``cdf(y)``: cumulative distribution function
        - ``ppf(q)``: percent point function (inverse CDF)
        - ``logpdf(y)``: log probability density (or mass) function

    Note on scoring rules:
        The ``score``/``d_score``/``metric`` methods implement the LogScore
        (negative log-likelihood).  CRPS is supported via separate
        ``crps_score``/``crps_d_score``/``crps_metric`` methods — see
        :mod:`ngboost_lightning.scoring` for the strategy-pattern wiring.

    Attributes:
        n_params: Number of internal (unconstrained) parameters.
    """

    n_params: int
    _params: NDArray[np.floating]

    @abstractmethod
    def __init__(self, params: NDArray[np.floating]) -> None:
        """Construct distribution from internal parameters.

        Args:
            params: Internal parameters, shape ``[n_samples, n_params]``.
        """

    @staticmethod
    @abstractmethod
    def fit(
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Estimate initial parameters from target data.

        Args:
            y: Target values, shape ``[n_samples]``.
            sample_weight: Per-sample weights, shape ``[n_samples]``.
                If ``None``, all samples are weighted equally.

        Returns:
            Internal parameter vector, shape ``[n_params]``. These are the
            starting point for boosting (analogous to NGBoost's
            ``init_params``).
        """

    @abstractmethod
    def score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample negative log-likelihood (LogScore).

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            NLL values, shape ``[n_samples]``. Lower is better.
        """

    @abstractmethod
    def d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Gradient of score w.r.t. internal parameters.

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, n_params]``.
        """

    @abstractmethod
    def metric(self) -> NDArray[np.floating]:
        """Fisher Information matrix.

        Returns:
            FI tensor, shape ``[n_samples, n_params, n_params]``.

        Note:
            For Normal with (mean, log_scale) parameterization, the FI is
            diagonal: ``diag(1/scale^2, 2)``. The full
            ``[n_samples, n_params, n_params]`` return shape is required
            to support non-diagonal Fisher matrices in future distributions.
        """

    @abstractmethod
    def mean(self) -> NDArray[np.floating]:
        """Conditional mean (point prediction).

        Returns:
            Mean values, shape ``[n_samples]``.
        """

    @abstractmethod
    def sample(self, n: int) -> NDArray[np.floating]:
        """Draw random samples from the distribution.

        Args:
            n: Number of samples to draw per distribution instance.

        Returns:
            Samples, shape ``[n, n_samples]``.
        """

    @abstractmethod
    def cdf(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Cumulative distribution function.

        Args:
            y: Values at which to evaluate the CDF.

        Returns:
            CDF values, same shape as ``y``.
        """

    @abstractmethod
    def ppf(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        """Percent point function (inverse CDF / quantile function).

        Args:
            q: Quantiles, values in [0, 1].

        Returns:
            Values at the given quantiles, same shape as ``q``.
        """

    @abstractmethod
    def logpdf(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Log probability density (or mass) function.

        For continuous distributions this is the log-PDF. For discrete
        distributions (e.g. Poisson) this returns the log-PMF.

        Args:
            y: Values at which to evaluate.

        Returns:
            Log-density (or log-mass) values, same shape as ``y``.
        """

    def logsf(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Log survival function: log(1 - CDF(y)).

        Required for right-censored survival analysis. Subclasses that
        support censored observations (Exponential, LogNormal, Weibull)
        should override this with scipy's numerically stable ``logsf``.

        Args:
            y: Values at which to evaluate.

        Returns:
            Log-survival values, same shape as ``y``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement logsf.")

    def crps_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-sample CRPS (Continuous Ranked Probability Score).

        Subclasses that support CRPS must override this method.

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            CRPS values, shape ``[n_samples]``. Lower is better.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement CRPS scoring."
        )

    def crps_d_score(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Gradient of CRPS w.r.t. internal parameters.

        Subclasses that support CRPS must override this method.

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Gradient array, shape ``[n_samples, n_params]``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement CRPS scoring."
        )

    def crps_metric(self) -> NDArray[np.floating]:
        """Riemannian metric tensor for the CRPS natural gradient.

        This is NOT the Fisher Information (which is the metric for LogScore).
        Each distribution must derive its own CRPS metric.

        Returns:
            Metric tensor, shape ``[n_samples, n_params, n_params]``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement CRPS scoring."
        )

    def natural_gradient(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute the natural gradient: FI^{-1} @ d_score.

        Default implementation solves the linear system. Subclasses with
        diagonal Fisher (e.g. Normal) can override for efficiency.

        Args:
            y: Observed target values, shape ``[n_samples]``.

        Returns:
            Natural gradient, shape ``[n_samples, n_params]``.
        """
        grad = self.d_score(y)
        fi = self.metric()
        # Solve FI @ nat_grad = grad for each sample
        # fi: [n_samples, n_params, n_params], grad: [n_samples, n_params]
        # np.linalg.solve with 2D b treats it as (M, K) matrix columns, not
        # a batch of vectors. Expand to [n, n_params, 1] then squeeze back.
        result: NDArray[np.floating] = np.linalg.solve(fi, grad[..., np.newaxis])[
            ..., 0
        ]
        return result

    def total_score(
        self,
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> float:
        """Weighted mean negative log-likelihood across all samples.

        Args:
            y: Observed target values, shape ``[n_samples]``.
            sample_weight: Per-sample weights, shape ``[n_samples]``.
                If ``None``, all samples are weighted equally.

        Returns:
            Scalar (weighted) mean NLL.
        """
        return float(np.average(self.score(y), weights=sample_weight))

    def __getitem__(
        self,
        key: int | slice | NDArray[np.integer],
    ) -> Self:
        """Slice the distribution (select a subset of samples).

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of samples in this distribution instance."""
        return len(self.mean())
