"""Unit tests for evaluation utilities."""

import numpy as np
import pytest
from scipy.stats import kstest

from ngboost_lightning.distributions.exponential import Exponential
from ngboost_lightning.distributions.normal import Normal
from ngboost_lightning.evaluation import calibration_error
from ngboost_lightning.evaluation import calibration_regression
from ngboost_lightning.evaluation import calibration_survival
from ngboost_lightning.evaluation import concordance_index
from ngboost_lightning.evaluation import pit_values
from ngboost_lightning.evaluation import plot_calibration_curve
from ngboost_lightning.evaluation import plot_pit_histogram

SEED = 42


# ---- helpers ----


def _make_well_calibrated_normal(
    n: int = 2000,
    seed: int = SEED,
) -> tuple[Normal, np.ndarray]:
    """Create a well-calibrated Normal distribution and matching samples.

    Generates samples from known (loc, scale) pairs, then builds a
    Normal distribution with those exact parameters. PIT values should
    be Uniform(0, 1).
    """
    rng = np.random.default_rng(seed)
    locs = rng.standard_normal(n)
    scales = np.abs(rng.standard_normal(n)) + 0.5
    y = rng.normal(locs, scales)
    params = np.column_stack([locs, np.log(scales)])
    dist = Normal(params)
    return dist, y


def _make_miscalibrated_normal(
    n: int = 2000,
    seed: int = SEED,
) -> tuple[Normal, np.ndarray]:
    """Create a miscalibrated Normal (underestimates variance).

    Uses correct locs but wrong (too small) scales, producing
    overconfident predictions.
    """
    rng = np.random.default_rng(seed)
    true_locs = rng.standard_normal(n)
    true_scales = np.abs(rng.standard_normal(n)) + 1.0
    y = rng.normal(true_locs, true_scales)
    # Use half the true scale → overconfident
    wrong_scales = true_scales * 0.5
    params = np.column_stack([true_locs, np.log(wrong_scales)])
    dist = Normal(params)
    return dist, y


# ---- PIT values ----


class TestPitValues:
    """Tests for pit_values function."""

    def test_shape(self) -> None:
        """PIT values have the same shape as y."""
        dist, y = _make_well_calibrated_normal(n=100)
        pits = pit_values(dist, y)
        assert pits.shape == y.shape

    def test_values_in_unit_interval(self) -> None:
        """PIT values should be in [0, 1]."""
        dist, y = _make_well_calibrated_normal(n=500)
        pits = pit_values(dist, y)
        assert np.all(pits >= 0.0)
        assert np.all(pits <= 1.0)

    def test_uniform_for_well_calibrated(self) -> None:
        """PIT values of well-calibrated model follow Uniform(0,1)."""
        dist, y = _make_well_calibrated_normal(n=5000)
        pits = pit_values(dist, y)
        # KS test: null hypothesis is uniform
        stat = kstest(pits, "uniform").statistic
        assert stat < 0.05  # should not reject uniformity


# ---- Calibration regression ----


class TestCalibrationRegression:
    """Tests for calibration_regression function."""

    def test_output_shapes(self) -> None:
        """Expected and observed arrays have correct shapes."""
        dist, y = _make_well_calibrated_normal(n=200)
        expected, observed = calibration_regression(dist, y, bins=11)
        assert expected.shape == (11,)
        assert observed.shape == (11,)

    def test_expected_is_linspace(self) -> None:
        """Expected quantiles are linspace(0, 1, bins)."""
        dist, y = _make_well_calibrated_normal(n=100)
        expected, _ = calibration_regression(dist, y, bins=11)
        np.testing.assert_allclose(expected, np.linspace(0, 1, 11))

    def test_well_calibrated_close_to_diagonal(self) -> None:
        """Well-calibrated model: observed ≈ expected."""
        dist, y = _make_well_calibrated_normal(n=5000)
        expected, observed = calibration_regression(dist, y, bins=11)
        # Exclude endpoints (0 and 1 are trivially matched)
        interior = slice(1, -1)
        np.testing.assert_allclose(observed[interior], expected[interior], atol=0.05)

    def test_miscalibrated_deviates(self) -> None:
        """Miscalibrated model: observed differs from expected."""
        dist, y = _make_miscalibrated_normal(n=5000)
        expected, observed = calibration_regression(dist, y, bins=11)
        # Overconfident model: more observations fall outside predicted
        # quantiles, so calibration error should be noticeable
        error = calibration_error(expected, observed)
        assert error > 0.005


# ---- Calibration error ----


class TestCalibrationError:
    """Tests for calibration_error function."""

    def test_perfect_calibration(self) -> None:
        """Identical arrays produce zero error."""
        x = np.linspace(0, 1, 11)
        assert calibration_error(x, x) == pytest.approx(0.0)

    def test_positive_for_mismatch(self) -> None:
        """Different arrays produce positive error."""
        x = np.linspace(0, 1, 11)
        y_off = x + 0.1
        assert calibration_error(x, y_off) > 0.0

    def test_well_calibrated_low_error(self) -> None:
        """Well-calibrated model has low calibration error."""
        dist, y = _make_well_calibrated_normal(n=5000)
        expected, observed = calibration_regression(dist, y, bins=11)
        error = calibration_error(expected, observed)
        assert error < 0.002

    def test_miscalibrated_higher_error(self) -> None:
        """Miscalibrated model has higher calibration error."""
        dist_good, y_good = _make_well_calibrated_normal(n=5000)
        dist_bad, y_bad = _make_miscalibrated_normal(n=5000)

        exp_good, obs_good = calibration_regression(dist_good, y_good, bins=11)
        exp_bad, obs_bad = calibration_regression(dist_bad, y_bad, bins=11)

        error_good = calibration_error(exp_good, obs_good)
        error_bad = calibration_error(exp_bad, obs_bad)

        assert error_bad > error_good


# ---- Concordance index ----


class TestConcordanceIndex:
    """Tests for concordance_index function."""

    def test_perfect_concordance(self) -> None:
        """Perfectly ordered predictions: CI = 1.0."""
        times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        events = np.array([True, True, True, True, True])
        # Perfect: predicted order matches observed order
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ci = concordance_index(predicted, times, events)
        assert ci == pytest.approx(1.0)

    def test_reversed_concordance(self) -> None:
        """Reversed predictions: CI = 0.0."""
        times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        events = np.array([True, True, True, True, True])
        # Reversed: shorter predicted time for longer actual time
        predicted = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        ci = concordance_index(predicted, times, events)
        assert ci == pytest.approx(0.0)

    def test_random_near_half(self) -> None:
        """Random predictions: CI ≈ 0.5."""
        rng = np.random.default_rng(SEED)
        n = 500
        times = rng.exponential(5.0, n)
        events = np.ones(n, dtype=bool)
        predicted = rng.exponential(5.0, n)
        ci = concordance_index(predicted, times, events)
        assert abs(ci - 0.5) < 0.1

    def test_matches_lifelines(self) -> None:
        """Result matches lifelines implementation."""
        from lifelines.utils import concordance_index as lifelines_ci

        rng = np.random.default_rng(SEED)
        n = 100
        times = rng.exponential(3.0, n)
        events = rng.random(n) > 0.3
        predicted = times + rng.normal(0, 1.0, n)

        our_ci = concordance_index(predicted, times, events)
        ll_ci = lifelines_ci(times, predicted, events)
        assert our_ci == pytest.approx(ll_ci, abs=0.01)

    def test_no_comparable_pairs(self) -> None:
        """All censored: returns 0.5 (no comparable pairs)."""
        times = np.array([1.0, 2.0, 3.0])
        events = np.array([False, False, False])
        predicted = np.array([3.0, 2.0, 1.0])
        ci = concordance_index(predicted, times, events)
        assert ci == pytest.approx(0.5)


# ---- Calibration survival ----


class TestCalibrationSurvival:
    """Tests for calibration_survival function."""

    def test_output_shapes(self) -> None:
        """Returns arrays of correct shape."""
        rng = np.random.default_rng(SEED)
        n = 200
        rates = np.abs(rng.standard_normal(n)) + 0.5
        params = np.column_stack([-np.log(rates)])
        dist = Exponential(params)
        T = rng.exponential(1.0 / rates)
        E = np.ones(n, dtype=bool)

        pred_probs, obs_rates = calibration_survival(dist, T, E, bins=5)
        assert pred_probs.shape == (5,)
        assert obs_rates.shape == (5,)

    def test_fully_observed_reasonable(self) -> None:
        """With all events observed, calibration should be reasonable."""
        rng = np.random.default_rng(SEED)
        n = 1000
        rate = 1.0
        params = np.full((n, 1), np.log(rate))
        dist = Exponential(params)
        T = rng.exponential(1.0 / rate, n)
        E = np.ones(n, dtype=bool)

        _pred_probs, obs_rates = calibration_survival(dist, T, E, bins=5)
        # For a well-calibrated model, predicted probs and observed rates
        # should be in the same ballpark (both increasing across bins)
        valid = ~np.isnan(obs_rates)
        assert np.sum(valid) >= 3


# ---- Plot functions ----


class TestPlotFunctions:
    """Tests for plot functions (require matplotlib)."""

    def test_plot_pit_histogram_returns_axes(self) -> None:
        """plot_pit_histogram returns a matplotlib Axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dist, y = _make_well_calibrated_normal(n=100)
        ax = plot_pit_histogram(dist, y, bins=10)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_plot_pit_histogram_with_ax(self) -> None:
        """plot_pit_histogram works with a provided axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dist, y = _make_well_calibrated_normal(n=100)
        _fig, ax_in = plt.subplots()
        ax_out = plot_pit_histogram(dist, y, bins=10, ax=ax_in)
        assert ax_out is ax_in
        plt.close("all")

    def test_plot_calibration_curve_returns_axes(self) -> None:
        """plot_calibration_curve returns a matplotlib Axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dist, y = _make_well_calibrated_normal(n=100)
        ax = plot_calibration_curve(dist, y, bins=11)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_plot_calibration_curve_with_ax(self) -> None:
        """plot_calibration_curve works with a provided axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dist, y = _make_well_calibrated_normal(n=100)
        _fig, ax_in = plt.subplots()
        ax_out = plot_calibration_curve(dist, y, bins=11, ax=ax_in)
        assert ax_out is ax_in
        plt.close("all")
