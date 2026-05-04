"""UCI regression benchmark dataset loaders.

Reproduces the dataset loading from the NGBoost paper (Duan et al. 2019,
Table 1) for head-to-head comparison of lightning-boost vs NGBoost.

All datasets are downloaded on first use and cached under
``benchmarks/data/``.  That directory is git-ignored.
"""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_DATA_DIR = Path(__file__).resolve().parent / "data"

DatasetInfo = tuple[NDArray[np.floating], NDArray[np.floating], int]
"""(X, y, N) tuple."""

# ── Download URLs ────────────────────────────────────────────────────

_URLS: dict[str, str] = {
    "housing": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
    "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
    "energy": "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
    "kin8nm": "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.csv",
    "naval": "https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
    "power": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
    "protein": "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
    "wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "yacht": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
    "msd": "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip",
}

# Local filenames after download/extraction
_FILENAMES: dict[str, str] = {
    "housing": "housing.data",
    "concrete": "Concrete_Data.xls",
    "energy": "ENB2012_data.xlsx",
    "kin8nm": "kin8nm.csv",
    "naval": "naval-propulsion.txt",
    "power": "power-plant.xlsx",
    "protein": "CASP.csv",
    "wine": "winequality-red.csv",
    "yacht": "yacht_hydrodynamics.data",
    "msd": "YearPredictionMSD.txt",
}


# ── Helpers ──────────────────────────────────────────────────────────


def _ensure_dir() -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR


def _cached_path(name: str) -> Path:
    return _DATA_DIR / _FILENAMES[name]


def _download(name: str) -> Path:
    """Download a dataset if not already cached.  Returns the local path."""
    dest = _cached_path(name)
    if dest.exists():
        return dest

    _ensure_dir()
    url = _URLS[name]

    # Datasets that come as zip archives need extraction
    if url.endswith(".zip"):
        _download_and_extract(name, url, dest)
    else:
        print(f"Downloading {name} from {url} ...")
        urlretrieve(url, dest)

    return dest


def _download_and_extract(name: str, url: str, dest: Path) -> None:
    """Download a zip archive and extract the needed file."""
    import io
    import zipfile

    from urllib.request import urlopen

    print(f"Downloading {name} from {url} ...")
    with urlopen(url) as resp:
        zdata = io.BytesIO(resp.read())

    with zipfile.ZipFile(zdata) as zf:
        members = zf.namelist()
        target_name = dest.name

        if name == "naval":
            # UCI CBM Dataset.zip has the data inside a subdirectory
            match = [
                m for m in members if m.endswith("data/data/UCI CBM Dataset/data.txt")
            ]
            if not match:
                match = [m for m in members if "data.txt" in m]
            if not match:
                msg = f"Cannot find data.txt in naval zip: {members}"
                raise FileNotFoundError(msg)
            with zf.open(match[0]) as src, open(dest, "wb") as dst:
                dst.write(src.read())

        elif name == "power":
            # CCPP.zip contains an xlsx inside a subfolder
            match = [m for m in members if m.endswith(".xlsx")]
            if not match:
                msg = f"Cannot find .xlsx in power zip: {members}"
                raise FileNotFoundError(msg)
            with zf.open(match[0]) as src, open(dest, "wb") as dst:
                dst.write(src.read())

        elif name == "msd":
            match = [m for m in members if target_name in m]
            if not match:
                msg = f"Cannot find {target_name} in msd zip: {members}"
                raise FileNotFoundError(msg)
            with zf.open(match[0]) as src, open(dest, "wb") as dst:
                dst.write(src.read())

        else:
            msg = f"No zip extraction logic for dataset {name!r}"
            raise NotImplementedError(msg)


# ── Loaders ──────────────────────────────────────────────────────────


def _load_housing() -> pd.DataFrame:
    return pd.read_csv(_download("housing"), header=None, sep=r"\s+")


def _load_concrete() -> pd.DataFrame:
    return pd.read_excel(_download("concrete"))


def _load_energy() -> pd.DataFrame:
    return pd.read_excel(_download("energy")).iloc[:, :-1]


def _load_kin8nm() -> pd.DataFrame:
    return pd.read_csv(_download("kin8nm"))


def _load_naval() -> pd.DataFrame:
    return pd.read_csv(
        _download("naval"),
        sep=r"\s+",
        header=None,
    ).iloc[:, :-1]


def _load_power() -> pd.DataFrame:
    return pd.read_excel(_download("power"))


def _load_protein() -> pd.DataFrame:
    return pd.read_csv(_download("protein"))[
        ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "RMSD"]
    ]


def _load_wine() -> pd.DataFrame:
    return pd.read_csv(_download("wine"), delimiter=";")


def _load_yacht() -> pd.DataFrame:
    return pd.read_csv(_download("yacht"), header=None, sep=r"\s+")


def _load_msd() -> pd.DataFrame:
    # Columns are reversed so the target (year) is last, matching the
    # ngboost convention: ``pd.read_csv(...).iloc[:, ::-1]``
    return pd.read_csv(_download("msd")).iloc[:, ::-1]


# ── Registry ─────────────────────────────────────────────────────────

DATASET_REGISTRY: dict[str, tuple[object, int]] = {
    "housing": (_load_housing, 506),
    "concrete": (_load_concrete, 1030),
    "energy": (_load_energy, 768),
    "kin8nm": (_load_kin8nm, 8192),
    "naval": (_load_naval, 11934),
    "power": (_load_power, 9568),
    "protein": (_load_protein, 45730),
    "wine": (_load_wine, 1588),
    "yacht": (_load_yacht, 308),
    "msd": (_load_msd, 515345),
}

# Per-dataset hyperparameter overrides from ngboost's run_empirical_regression.sh
# and the paper text.  Datasets not listed here use the defaults
# (lr=0.01, n_est=2000, n_splits=20, minibatch_frac=1.0).
#
# MSD uses a fixed train/test split (first 463715 / rest), 1 split, and
# 10% minibatch.  The fixed split indices are stored here so the CLI can
# detect and apply them.
DATASET_HPARAMS: dict[str, dict[str, object]] = {
    "housing": {"lr": 0.0007, "n_est": 5000},
    "concrete": {"lr": 0.002, "n_est": 5000},
    "energy": {"lr": 0.002, "n_est": 5000},
    "protein": {"n_splits": 5},
    "msd": {
        "n_splits": 1,
        "minibatch_frac": 0.1,
        "fixed_split": (463715, None),  # (train_end, test_end=None means rest)
    },
}


def list_datasets() -> list[str]:
    """Return sorted list of available dataset names."""
    return sorted(DATASET_REGISTRY.keys())


def load_dataset(name: str) -> DatasetInfo:
    """Load a UCI benchmark dataset by name.

    Args:
        name: One of the keys in :data:`DATASET_REGISTRY`.

    Returns:
        ``(X, y, N)`` where *X* is the feature matrix, *y* is the target
        vector, and *N* is the sample count.

    Raises:
        KeyError: If *name* is not a recognised dataset.
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(list_datasets())
        msg = f"Unknown dataset {name!r}. Available: {available}"
        raise KeyError(msg)

    loader, _expected_n = DATASET_REGISTRY[name]
    df = loader()
    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.astype(np.float64)
    return X, y, len(y)
