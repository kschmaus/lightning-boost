"""Shared test fixtures for ngboost_lightning tests."""

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from tests._constants import SEED


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(seed=SEED)


@pytest.fixture()
def california_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """California housing train/test split with fixed seed."""
    cal = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        cal.data, cal.target, test_size=0.2, random_state=SEED
    )
    return X_train, X_test, y_train, y_test
