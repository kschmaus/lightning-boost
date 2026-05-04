"""Shared utilities for lightning-boost benchmarks."""

import gc
import platform
import time
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# ── Constants ────────────────────────────────────────────────────────
SEED = 24601
N_TRIALS = 3
N_ESTIMATORS = 200
LEARNING_RATE = 0.05
NUM_LEAVES = 31
N_FEATURES = 10


def make_regression_data(
    n_samples: int,
    n_features: int = N_FEATURES,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic regression data with 80/20 train/test split.

    Args:
        n_samples: Total number of samples.
        n_features: Number of features.
        seed: Random seed.

    Returns:
        (X_train, X_test, y_train, y_test) tuple.
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        noise=10.0,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def make_binary_data(
    n_samples: int,
    n_features: int = N_FEATURES,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic binary classification data with 80/20 split.

    Args:
        n_samples: Total number of samples.
        n_features: Number of features.
        seed: Random seed.

    Returns:
        (X_train, X_test, y_train, y_test) tuple.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def time_it(
    func: object,
    n_trials: int = N_TRIALS,
) -> tuple[float, float]:
    """Time a callable over multiple trials.

    Runs gc.collect() between trials to reduce noise.

    Args:
        func: Zero-argument callable to time.
        n_trials: Number of repeated trials.

    Returns:
        (mean_seconds, std_seconds) tuple.
    """
    times = []
    for _ in range(n_trials):
        gc.collect()
        start = time.perf_counter()
        func()  # type: ignore[operator]
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print a formatted ASCII table to stdout.

    Args:
        title: Table title printed above the table.
        headers: Column header strings.
        rows: List of rows, each a list of string values.
    """
    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Format
    sep = "  ".join("-" * w for w in widths)
    header_line = "  ".join(h.rjust(w) for h, w in zip(headers, widths, strict=True))

    print(f"\n{title}")
    print(header_line)
    print(sep)
    for row in rows:
        print("  ".join(cell.rjust(w) for cell, w in zip(row, widths, strict=True)))
    print()


def save_results(data: dict[str, Any], name: str) -> Path:
    """Save benchmark results to a timestamped YAML file.

    Creates ``benchmarks/results/<name>_<timestamp>.yaml``.

    Args:
        data: Results dictionary to serialize.
        name: Base name for the output file (e.g. ``"comparison"``).

    Returns:
        Path to the written YAML file.
    """
    import lightgbm
    import sklearn
    import yaml

    import ngboost_lightning

    data["metadata"] = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ngboost_lightning": ngboost_lightning.__version__,
        "lightgbm": lightgbm.__version__,
        "scikit_learn": sklearn.__version__,
    }

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"{name}_{ts}.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    print(f"Results saved to {path}")
    return path
