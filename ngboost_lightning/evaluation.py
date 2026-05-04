"""Evaluation utilities for probabilistic predictions.

Provides diagnostic tools for assessing calibration and discrimination
of predicted distributions, including PIT-based calibration for
regression and concordance index for survival analysis.

Plot functions require matplotlib (optional dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ngboost_lightning.distributions.base import Distribution


def pit_values(
    pred_dist: Distribution,
    y: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Probability Integral Transform values.

    For a well-calibrated model, PIT values ``F(y)`` should follow
    a Uniform(0, 1) distribution.

    Args:
        pred_dist: Predicted distribution instance (one per sample).
        y: Observed target values, shape ``[n_samples]``.

    Returns:
        PIT values ``F_i(y_i)``, shape ``[n_samples]``, in ``[0, 1]``.
    """
    result: NDArray[np.floating] = np.asarray(pred_dist.cdf(y), dtype=np.float64)
    return result


def calibration_regression(
    pred_dist: Distribution,
    y: NDArray[np.floating],
    bins: int = 11,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """PIT-based calibration curve for regression.

    For each quantile level ``q`` in ``linspace(0, 1, bins)``, computes
    the fraction of observations falling below the predicted ``q``-th
    quantile.  A perfectly calibrated model has
    ``observed_fractions == expected_quantiles``.

    Args:
        pred_dist: Predicted distribution instance (one per sample).
        y: Observed target values, shape ``[n_samples]``.
        bins: Number of equally spaced quantile levels (including 0 and 1).

    Returns:
        Tuple ``(expected_quantiles, observed_fractions)``, each shape
        ``[bins]``.
    """
    y = np.asarray(y, dtype=np.float64)
    expected = np.linspace(0.0, 1.0, bins)
    observed = np.empty_like(expected)
    for i, q in enumerate(expected):
        quantile_vals = pred_dist.ppf(np.full(len(y), q))
        observed[i] = float(np.mean(y <= quantile_vals))
    return expected, observed


def calibration_error(
    expected: NDArray[np.floating],
    observed: NDArray[np.floating],
) -> float:
    """Mean squared calibration error.

    Args:
        expected: Expected quantile levels, shape ``[n_bins]``.
        observed: Observed fractions, shape ``[n_bins]``.

    Returns:
        Mean squared error between expected and observed.
    """
    expected = np.asarray(expected, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    return float(np.mean((expected - observed) ** 2))


def concordance_index(
    predicted_times: NDArray[np.floating],
    event_times: NDArray[np.floating],
    event_observed: NDArray[np.bool_],
) -> float:
    """Harrell's concordance index (C-statistic).

    Measures the fraction of comparable pairs where the predicted
    survival time correctly orders the observed event times.
    A pair ``(i, j)`` is comparable if the earlier event is uncensored.

    Args:
        predicted_times: Predicted survival times (e.g. median),
            shape ``[n_samples]``.
        event_times: Observed times, shape ``[n_samples]``.
        event_observed: Boolean event indicator (``True`` = observed),
            shape ``[n_samples]``.

    Returns:
        C-index in ``[0, 1]``.  ``0.5`` is random, ``1.0`` is perfect
        concordance.
    """
    predicted_times = np.asarray(predicted_times, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)
    event_observed = np.asarray(event_observed, dtype=bool)

    n = len(event_times)
    concordant = 0.0
    discordant = 0.0
    tied_risk = 0.0

    for i in range(n):
        if not event_observed[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # Pair is comparable if i had an event and T_i < T_j,
            # or if i had an event and T_i == T_j and j is censored.
            comparable = event_times[i] < event_times[j] or (
                event_times[i] == event_times[j] and not event_observed[j]
            )
            if not comparable:
                continue

            if predicted_times[i] < predicted_times[j]:
                concordant += 1.0
            elif predicted_times[i] > predicted_times[j]:
                discordant += 1.0
            else:
                tied_risk += 1.0

    total = concordant + discordant + tied_risk
    if total == 0.0:
        return 0.5
    return (concordant + 0.5 * tied_risk) / total


def calibration_survival(
    pred_dist: Distribution,
    event_times: NDArray[np.floating],
    event_observed: NDArray[np.bool_],
    bins: int = 10,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """D-calibration for survival models.

    Bins samples by their predicted survival probability at the observed
    time, then compares the predicted probability against the observed
    event rate within each bin.

    Args:
        pred_dist: Predicted distribution instance (one per sample).
        event_times: Observed times, shape ``[n_samples]``.
        event_observed: Boolean event indicator (``True`` = observed),
            shape ``[n_samples]``.
        bins: Number of bins.

    Returns:
        Tuple ``(predicted_probs, observed_rates)``, each shape
        ``[bins]``.  For a well-calibrated model these should be close.
    """
    T = np.asarray(event_times, dtype=np.float64)  # noqa: N806
    E = np.asarray(event_observed, dtype=bool)  # noqa: N806

    # Predicted probability of event by observed time: F(T) = 1 - S(T)
    predicted_event_prob = np.asarray(pred_dist.cdf(T), dtype=np.float64)

    # Bin by predicted event probability
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    predicted_probs = np.empty(bins)
    observed_rates = np.empty(bins)

    for i in range(bins):
        mask = (predicted_event_prob >= bin_edges[i]) & (
            predicted_event_prob < bin_edges[i + 1]
        )
        if i == bins - 1:
            # Include right edge in last bin
            mask = mask | (predicted_event_prob == bin_edges[i + 1])

        if np.sum(mask) == 0:
            predicted_probs[i] = (bin_edges[i] + bin_edges[i + 1]) / 2.0
            observed_rates[i] = np.nan
        else:
            predicted_probs[i] = float(np.mean(predicted_event_prob[mask]))
            observed_rates[i] = float(np.mean(E[mask]))

    return predicted_probs, observed_rates


def plot_pit_histogram(
    pred_dist: Distribution,
    y: NDArray[np.floating],
    bins: int = 20,
    ax: Any = None,
) -> Any:
    """Histogram of PIT values with uniform reference line.

    Requires matplotlib.

    Args:
        pred_dist: Predicted distribution instance (one per sample).
        y: Observed target values, shape ``[n_samples]``.
        bins: Number of histogram bins.
        ax: Matplotlib axes to plot on.  If ``None``, creates a new figure.

    Returns:
        Matplotlib ``Axes`` object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "matplotlib is required for plot_pit_histogram. "
            "Install it with: pip install matplotlib"
        ) from err

    pits = pit_values(pred_dist, y)

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(pits, bins=bins, density=True, alpha=0.7, edgecolor="black")
    ax.axhline(y=1.0, color="red", linestyle="--", label="Uniform")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title("PIT Histogram")
    ax.legend()
    ax.set_xlim(0, 1)

    return ax


def plot_calibration_curve(
    pred_dist: Distribution,
    y: NDArray[np.floating],
    bins: int = 11,
    ax: Any = None,
) -> Any:
    """Calibration curve: expected vs observed quantile fractions.

    Requires matplotlib.

    Args:
        pred_dist: Predicted distribution instance (one per sample).
        y: Observed target values, shape ``[n_samples]``.
        bins: Number of quantile levels.
        ax: Matplotlib axes to plot on.  If ``None``, creates a new figure.

    Returns:
        Matplotlib ``Axes`` object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "matplotlib is required for plot_calibration_curve. "
            "Install it with: pip install matplotlib"
        ) from err

    expected, observed = calibration_regression(pred_dist, y, bins=bins)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax.plot(expected, observed, "bo-", label="Model")
    ax.set_xlabel("Expected quantile")
    ax.set_ylabel("Observed fraction")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ax
