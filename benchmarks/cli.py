"""Benchmark CLI for lightning-boost vs NGBoost.

Usage::

    uv run python -m benchmarks training
    uv run python -m benchmarks inference
    uv run python -m benchmarks scaling
    uv run python -m benchmarks comparison
    uv run python -m benchmarks uci
    uv run python -m benchmarks all
"""

import time as time_mod

import numpy as np
import typer
from ngboost import NGBClassifier
from ngboost import NGBRegressor
from ngboost.distns import Bernoulli as NGBBernoulli
from ngboost.distns import Normal as NGBNormal
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from benchmarks._datasets import DATASET_HPARAMS
from benchmarks._datasets import list_datasets
from benchmarks._datasets import load_dataset
from benchmarks._utils import LEARNING_RATE
from benchmarks._utils import N_ESTIMATORS
from benchmarks._utils import N_FEATURES
from benchmarks._utils import N_TRIALS
from benchmarks._utils import NUM_LEAVES
from benchmarks._utils import SEED
from benchmarks._utils import make_binary_data
from benchmarks._utils import make_regression_data
from benchmarks._utils import print_table
from benchmarks._utils import save_results
from benchmarks._utils import time_it
from ngboost_lightning import LightningBoostClassifier
from ngboost_lightning import LightningBoostRegressor

app = typer.Typer(help="Benchmark suite for lightning-boost vs NGBoost.")

# ── Training speed ───────────────────────────────────────────────────

TRAINING_SIZES = [10_000, 50_000, 100_000]


def _bench_regression_training() -> None:
    rows: list[list[str]] = []

    for n in TRAINING_SIZES:
        X_train, _X_test, y_train, _y_test = make_regression_data(n)

        def fit_ngb(X=X_train, y=y_train) -> None:
            NGBRegressor(
                Dist=NGBNormal,
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                verbose=False,
                random_state=SEED,
            ).fit(X, y)

        def fit_lb(X=X_train, y=y_train) -> None:
            LightningBoostRegressor(
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                num_leaves=NUM_LEAVES,
                verbose=False,
                random_state=SEED,
            ).fit(X, y)

        ngb_mean, ngb_std = time_it(fit_ngb)
        lb_mean, lb_std = time_it(fit_lb)
        speedup = ngb_mean / lb_mean if lb_mean > 0 else float("inf")

        rows.append(
            [
                f"{n:,}",
                f"{ngb_mean:.2f} \u00b1 {ngb_std:.2f}",
                f"{lb_mean:.2f} \u00b1 {lb_std:.2f}",
                f"{speedup:.1f}x",
            ]
        )

    print_table(
        "Regression Training Speed (Normal, n_estimators=200)",
        ["n_samples", "NGBoost (s)", "LB (s)", "Speedup"],
        rows,
    )


def _bench_binary_training() -> None:
    rows: list[list[str]] = []

    for n in TRAINING_SIZES:
        X_train, _X_test, y_train, _y_test = make_binary_data(n)

        def fit_ngb(X=X_train, y=y_train) -> None:
            NGBClassifier(
                Dist=NGBBernoulli,
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                verbose=False,
                random_state=SEED,
            ).fit(X, y)

        def fit_lb(X=X_train, y=y_train) -> None:
            LightningBoostClassifier(
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                num_leaves=NUM_LEAVES,
                verbose=False,
                random_state=SEED,
            ).fit(X, y)

        ngb_mean, ngb_std = time_it(fit_ngb)
        lb_mean, lb_std = time_it(fit_lb)
        speedup = ngb_mean / lb_mean if lb_mean > 0 else float("inf")

        rows.append(
            [
                f"{n:,}",
                f"{ngb_mean:.2f} \u00b1 {ngb_std:.2f}",
                f"{lb_mean:.2f} \u00b1 {lb_std:.2f}",
                f"{speedup:.1f}x",
            ]
        )

    print_table(
        "Binary Classification Training Speed (Bernoulli, n_estimators=200)",
        ["n_samples", "NGBoost (s)", "LB (s)", "Speedup"],
        rows,
    )


@app.command()
def training() -> None:
    """Benchmark training speed for regression and binary classification."""
    _bench_regression_training()
    _bench_binary_training()


# ── Inference speed ──────────────────────────────────────────────────

INFERENCE_TRAIN_SIZE = 50_000
INFERENCE_TEST_SIZES = [1_000, 10_000, 50_000]


def _prepare_test_data(X_test_full: np.ndarray, n_test: int) -> np.ndarray:
    if n_test <= len(X_test_full):
        return X_test_full[:n_test]
    reps = (n_test // len(X_test_full)) + 1
    return np.tile(X_test_full, (reps, 1))[:n_test]


def _bench_regression_inference() -> None:
    X_train, X_test_full, y_train, _y_test = make_regression_data(INFERENCE_TRAIN_SIZE)

    print("Fitting regression models on 50k samples...")
    ngb = NGBRegressor(
        Dist=NGBNormal,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        verbose=False,
        random_state=SEED,
    )
    ngb.fit(X_train, y_train)

    lb = LightningBoostRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        verbose=False,
        random_state=SEED,
    )
    lb.fit(X_train, y_train)

    for method_name, ngb_fn, lb_fn in [
        ("predict", ngb.predict, lb.predict),
        ("pred_dist", ngb.pred_dist, lb.pred_dist),
    ]:
        rows: list[list[str]] = []
        for n_test in INFERENCE_TEST_SIZES:
            X_test = _prepare_test_data(X_test_full, n_test)

            ngb_mean, _ = time_it(lambda _X=X_test, _f=ngb_fn: _f(_X))
            lb_mean, _ = time_it(lambda _X=X_test, _f=lb_fn: _f(_X))

            ngb_tp = n_test / ngb_mean if ngb_mean > 0 else float("inf")
            lb_tp = n_test / lb_mean if lb_mean > 0 else float("inf")
            speedup = lb_tp / ngb_tp if ngb_tp > 0 else float("inf")

            rows.append(
                [
                    f"{n_test:,}",
                    f"{ngb_tp:,.0f}",
                    f"{lb_tp:,.0f}",
                    f"{speedup:.1f}x",
                ]
            )

        print_table(
            f"Regression Inference \u2014 {method_name}() (samples/sec)",
            ["n_test", "NGBoost", "LB", "Speedup"],
            rows,
        )


def _bench_binary_inference() -> None:
    X_train, X_test_full, y_train, _y_test = make_binary_data(INFERENCE_TRAIN_SIZE)

    print("Fitting binary classification models on 50k samples...")
    ngb = NGBClassifier(
        Dist=NGBBernoulli,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        verbose=False,
        random_state=SEED,
    )
    ngb.fit(X_train, y_train)

    lb = LightningBoostClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        verbose=False,
        random_state=SEED,
    )
    lb.fit(X_train, y_train)

    for method_name, ngb_fn, lb_fn in [
        ("predict", ngb.predict, lb.predict),
        ("predict_proba", ngb.predict_proba, lb.predict_proba),
        ("pred_dist", ngb.pred_dist, lb.pred_dist),
    ]:
        rows: list[list[str]] = []
        for n_test in INFERENCE_TEST_SIZES:
            X_test = _prepare_test_data(X_test_full, n_test)

            ngb_mean, _ = time_it(lambda _X=X_test, _f=ngb_fn: _f(_X))
            lb_mean, _ = time_it(lambda _X=X_test, _f=lb_fn: _f(_X))

            ngb_tp = n_test / ngb_mean if ngb_mean > 0 else float("inf")
            lb_tp = n_test / lb_mean if lb_mean > 0 else float("inf")
            speedup = lb_tp / ngb_tp if ngb_tp > 0 else float("inf")

            rows.append(
                [
                    f"{n_test:,}",
                    f"{ngb_tp:,.0f}",
                    f"{lb_tp:,.0f}",
                    f"{speedup:.1f}x",
                ]
            )

        print_table(
            f"Binary Inference \u2014 {method_name}() (samples/sec)",
            ["n_test", "NGBoost", "LB", "Speedup"],
            rows,
        )


@app.command()
def inference() -> None:
    """Benchmark inference throughput for regression and classification."""
    _bench_regression_inference()
    _bench_binary_inference()


# ── Scaling curves ───────────────────────────────────────────────────

SCALING_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000]


def _make_ngb(X: np.ndarray, y: np.ndarray) -> NGBRegressor:
    m = NGBRegressor(
        Dist=NGBNormal,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        verbose=False,
        random_state=SEED,
    )
    m.fit(X, y)
    return m


def _make_lb(X: np.ndarray, y: np.ndarray) -> LightningBoostRegressor:
    m = LightningBoostRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        verbose=False,
        random_state=SEED,
    )
    m.fit(X, y)
    return m


@app.command()
def scaling() -> None:
    """Benchmark training time and model quality vs dataset size."""
    rows: list[list[str]] = []

    for n in SCALING_SIZES:
        X_train, X_test, y_train, y_test = make_regression_data(n)

        ngb_mean, _ = time_it(lambda _X=X_train, _y=y_train: _make_ngb(_X, _y))
        lb_mean, _ = time_it(lambda _X=X_train, _y=y_train: _make_lb(_X, _y))
        speedup = ngb_mean / lb_mean if lb_mean > 0 else float("inf")

        # Quality metrics (one final fit)
        ngb_model = _make_ngb(X_train, y_train)
        lb_model = _make_lb(X_train, y_train)

        ngb_dist = ngb_model.pred_dist(X_test)
        lb_dist = lb_model.pred_dist(X_test)

        ngb_nll = float(-ngb_dist.logpdf(y_test).mean())
        lb_nll = float(lb_dist.total_score(y_test))

        ngb_preds = ngb_model.predict(X_test)
        lb_preds = lb_model.predict(X_test)

        ngb_mse = float(np.mean((ngb_preds - y_test) ** 2))
        lb_mse = float(np.mean((lb_preds - y_test) ** 2))

        rows.append(
            [
                f"{n:,}",
                f"{ngb_mean:.2f}",
                f"{lb_mean:.2f}",
                f"{speedup:.1f}x",
                f"{ngb_nll:.3f}",
                f"{lb_nll:.3f}",
                f"{ngb_mse:.1f}",
                f"{lb_mse:.1f}",
            ]
        )

    print_table(
        "Scaling: Training Time & Quality vs Dataset Size (Regression, Normal)",
        [
            "n_samples",
            "NGB (s)",
            "LB (s)",
            "Speedup",
            "NGB NLL",
            "LB NLL",
            "NGB MSE",
            "LB MSE",
        ],
        rows,
    )


# ── Full comparison (with JSON output) ──────────────────────────────

COMPARISON_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000]


def _coverage_90(dist_obj: object, y: np.ndarray) -> float:
    if hasattr(dist_obj, "ppf"):
        lo, hi = dist_obj.ppf(0.05), dist_obj.ppf(0.95)
    else:
        lo = dist_obj.dist.ppf(0.05)  # type: ignore[attr-defined]
        hi = dist_obj.dist.ppf(0.95)  # type: ignore[attr-defined]
    return float(np.mean((y >= lo) & (y <= hi)))


@app.command()
def comparison() -> None:
    """Full comparison with JSON output: time, NLL, MSE, coverage."""
    config = {
        "n_estimators": N_ESTIMATORS,
        "learning_rate": LEARNING_RATE,
        "num_leaves": NUM_LEAVES,
        "n_features": N_FEATURES,
        "n_trials": N_TRIALS,
        "seed": SEED,
    }

    all_results = []
    table_rows: list[list[str]] = []

    for n in COMPARISON_SIZES:
        print(f"Running n_samples={n:,} ...")
        X_train, X_test, y_train, y_test = make_regression_data(n)

        def fit_ngb(X=X_train, y=y_train) -> NGBRegressor:
            m = NGBRegressor(
                Dist=NGBNormal,
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                verbose=False,
                random_state=SEED,
            )
            m.fit(X, y)
            return m

        def fit_lb(X=X_train, y=y_train) -> LightningBoostRegressor:
            m = LightningBoostRegressor(
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                num_leaves=NUM_LEAVES,
                verbose=False,
                random_state=SEED,
            )
            m.fit(X, y)
            return m

        ngb_time_mean, ngb_time_std = time_it(fit_ngb)
        lb_time_mean, lb_time_std = time_it(fit_lb)
        speedup = ngb_time_mean / lb_time_mean if lb_time_mean > 0 else float("inf")

        ngb_model = fit_ngb()
        lb_model = fit_lb()

        ngb_dist = ngb_model.pred_dist(X_test)
        lb_dist = lb_model.pred_dist(X_test)

        ngb_nll = float(-ngb_dist.logpdf(y_test).mean())
        lb_nll = float(lb_dist.total_score(y_test))

        ngb_preds = ngb_model.predict(X_test)
        lb_preds = lb_model.predict(X_test)
        ngb_mse = float(np.mean((ngb_preds - y_test) ** 2))
        lb_mse = float(np.mean((lb_preds - y_test) ** 2))

        ngb_cov = _coverage_90(ngb_dist, y_test)
        lb_cov = _coverage_90(lb_dist, y_test)

        all_results.append(
            {
                "n_samples": n,
                "ngboost": {
                    "train_time_mean": round(ngb_time_mean, 3),
                    "train_time_std": round(ngb_time_std, 3),
                    "nll": round(ngb_nll, 4),
                    "mse": round(ngb_mse, 4),
                    "coverage_90": round(ngb_cov, 4),
                },
                "ngboost_lightning": {
                    "train_time_mean": round(lb_time_mean, 3),
                    "train_time_std": round(lb_time_std, 3),
                    "nll": round(lb_nll, 4),
                    "mse": round(lb_mse, 4),
                    "coverage_90": round(lb_cov, 4),
                },
            }
        )

        table_rows.append(
            [
                f"{n:,}",
                f"{ngb_time_mean:.2f}",
                f"{lb_time_mean:.2f}",
                f"{speedup:.1f}x",
                f"{ngb_nll:.3f}",
                f"{lb_nll:.3f}",
                f"{ngb_mse:.1f}",
                f"{lb_mse:.1f}",
                f"{ngb_cov:.1%}",
                f"{lb_cov:.1%}",
            ]
        )

    print_table(
        "Comparison: Training Time, NLL, MSE, 90% Coverage (Regression, Normal)",
        [
            "n_samples",
            "NGB (s)",
            "LB (s)",
            "Speedup",
            "NGB NLL",
            "LB NLL",
            "NGB MSE",
            "LB MSE",
            "NGB Cov",
            "LB Cov",
        ],
        table_rows,
    )

    save_results({"config": config, "results": all_results}, "comparison")


# ── UCI paper benchmark (Table 1 reproduction) ─────────────────────

UCI_DEFAULT_N_EST = 2000
UCI_DEFAULT_LR = 0.01
UCI_DEFAULT_SPLITS = 20
UCI_EARLY_STOPPING = 50


def _run_uci_fold(
    X_trainall: np.ndarray,
    X_test: np.ndarray,
    y_trainall: np.ndarray,
    y_test: np.ndarray,
    n_est: int,
    lr: float,
    minibatch_frac: float = 1.0,
) -> dict:
    """Run a single CV fold for both NGBoost and lightning-boost.

    Returns a dict with per-model NLL, RMSE, and training time.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainall, y_trainall, test_size=0.2
    )

    # --- NGBoost ---
    ngb = NGBRegressor(
        Dist=NGBNormal,
        n_estimators=n_est,
        learning_rate=lr,
        natural_gradient=True,
        minibatch_frac=minibatch_frac,
        verbose=False,
    )
    t0 = time_mod.perf_counter()
    ngb.fit(
        X_train,
        y_train,
        X_val=X_val,
        Y_val=y_val,
        early_stopping_rounds=UCI_EARLY_STOPPING,
    )
    ngb_time = time_mod.perf_counter() - t0

    ngb_dist = ngb.pred_dist(X_test)
    ngb_nll = float(-ngb_dist.logpdf(y_test).mean())
    ngb_rmse = float(np.sqrt(mean_squared_error(ngb.predict(X_test), y_test)))

    # --- lightning-boost ---
    lb = LightningBoostRegressor(
        n_estimators=n_est,
        learning_rate=lr,
        natural_gradient=True,
        minibatch_frac=minibatch_frac,
        verbose=False,
    )
    t0 = time_mod.perf_counter()
    lb.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=UCI_EARLY_STOPPING,
    )
    lb_time = time_mod.perf_counter() - t0

    lb_dist = lb.pred_dist(X_test)
    lb_nll = float(lb_dist.total_score(y_test))
    lb_rmse = float(np.sqrt(mean_squared_error(lb.predict(X_test), y_test)))

    return {
        "ngb_nll": ngb_nll,
        "ngb_rmse": ngb_rmse,
        "ngb_time": ngb_time,
        "lb_nll": lb_nll,
        "lb_rmse": lb_rmse,
        "lb_time": lb_time,
    }


@app.command()
def uci(
    dataset: str | None = typer.Option(
        None, help="Single dataset name, or omit for all 10 UCI datasets."
    ),
    n_splits: int = typer.Option(
        UCI_DEFAULT_SPLITS, help="Number of random train/test splits."
    ),
    n_est: int = typer.Option(UCI_DEFAULT_N_EST, help="Max boosting iterations."),
    lr: float = typer.Option(
        UCI_DEFAULT_LR,
        help="Learning rate (overridden per-dataset when defaults used).",
    ),
    seed: int = typer.Option(1, help="Random seed (matches ngboost paper)."),
) -> None:
    """Reproduce Table 1 from the NGBoost paper: NLL on UCI datasets.

    Compares NGBoost vs lightning-boost on quality (NLL, RMSE) and speed.
    Uses the Gal & Ghahramani (2016) cross-validation protocol: n_splits
    random 90/10 train/test splits with a further 80/20 train/val split
    for early stopping.
    """
    datasets = [dataset] if dataset else list_datasets()

    summary_rows: list[list[str]] = []
    all_results: list[dict] = []

    for ds_name in datasets:
        # Per-dataset hyperparameter overrides
        hparams = DATASET_HPARAMS.get(ds_name, {})
        ds_lr = hparams.get("lr", lr)
        ds_n_est = hparams.get("n_est", n_est)
        ds_n_splits = hparams.get("n_splits", n_splits)
        ds_minibatch_frac = hparams.get("minibatch_frac", 1.0)

        X, y, N = load_dataset(ds_name)
        print(f"\n{'─' * 60}")
        print(
            f"  {ds_name} (N={N}, {ds_n_splits} splits, lr={ds_lr}, n_est={ds_n_est})"
        )
        print(f"{'─' * 60}")

        # Build folds — MSD uses a fixed split; others use random CV
        fixed_split = hparams.get("fixed_split")
        if fixed_split is not None:
            train_end = fixed_split[0]
            folds = [(np.arange(train_end), np.arange(train_end, len(X)))]
        else:
            np.random.seed(seed)
            folds = []
            n = X.shape[0]
            for _ in range(ds_n_splits):
                perm = np.random.choice(range(n), n, replace=False)
                end_train = round(n * 9.0 / 10)
                train_idx = perm[:end_train]
                test_idx = perm[end_train:]
                folds.append((train_idx, test_idx))

        ngb_nlls, ngb_rmses, ngb_times = [], [], []
        lb_nlls, lb_rmses, lb_times = [], [], []

        for i, (train_idx, test_idx) in enumerate(folds):
            X_trainall, X_test = X[train_idx], X[test_idx]
            y_trainall, y_test = y[train_idx], y[test_idx]

            result = _run_uci_fold(
                X_trainall,
                X_test,
                y_trainall,
                y_test,
                n_est=ds_n_est,
                lr=ds_lr,
                minibatch_frac=ds_minibatch_frac,
            )

            ngb_nlls.append(result["ngb_nll"])
            ngb_rmses.append(result["ngb_rmse"])
            ngb_times.append(result["ngb_time"])
            lb_nlls.append(result["lb_nll"])
            lb_rmses.append(result["lb_rmse"])
            lb_times.append(result["lb_time"])

            print(
                f"  [{i + 1}/{ds_n_splits}] "
                f"NGB NLL={result['ngb_nll']:.3f}  LB NLL={result['lb_nll']:.3f}  "
                f"NGB {result['ngb_time']:.1f}s  LB {result['lb_time']:.1f}s"
            )

        # Per-dataset summary table
        ngb_nll_arr = np.array(ngb_nlls)
        lb_nll_arr = np.array(lb_nlls)
        ngb_rmse_arr = np.array(ngb_rmses)
        lb_rmse_arr = np.array(lb_rmses)
        ngb_time_arr = np.array(ngb_times)
        lb_time_arr = np.array(lb_times)

        speedup = (
            ngb_time_arr.mean() / lb_time_arr.mean()
            if lb_time_arr.mean() > 0
            else float("inf")
        )

        print_table(
            f"{ds_name} (N={N})",
            ["Metric", "NGBoost", "LB"],
            [
                [
                    "NLL",
                    f"{ngb_nll_arr.mean():.2f} \u00b1 {ngb_nll_arr.std():.2f}",
                    f"{lb_nll_arr.mean():.2f} \u00b1 {lb_nll_arr.std():.2f}",
                ],
                [
                    "RMSE",
                    f"{ngb_rmse_arr.mean():.2f} \u00b1 {ngb_rmse_arr.std():.2f}",
                    f"{lb_rmse_arr.mean():.2f} \u00b1 {lb_rmse_arr.std():.2f}",
                ],
                [
                    "Time (s)",
                    f"{ngb_time_arr.mean():.2f} \u00b1 {ngb_time_arr.std():.2f}",
                    f"{lb_time_arr.mean():.2f} \u00b1 {lb_time_arr.std():.2f}"
                    f" ({speedup:.1f}x)",
                ],
            ],
        )

        summary_rows.append(
            [
                ds_name,
                str(N),
                f"{ngb_nll_arr.mean():.2f} \u00b1 {ngb_nll_arr.std():.2f}",
                f"{lb_nll_arr.mean():.2f} \u00b1 {lb_nll_arr.std():.2f}",
                f"{ngb_time_arr.mean():.1f}",
                f"{lb_time_arr.mean():.1f}",
                f"{speedup:.1f}x",
            ]
        )

        all_results.append(
            {
                "dataset": ds_name,
                "N": N,
                "n_splits": ds_n_splits,
                "lr": ds_lr,
                "n_est": ds_n_est,
                "ngboost": {
                    "nll_mean": round(float(ngb_nll_arr.mean()), 4),
                    "nll_std": round(float(ngb_nll_arr.std()), 4),
                    "rmse_mean": round(float(ngb_rmse_arr.mean()), 4),
                    "rmse_std": round(float(ngb_rmse_arr.std()), 4),
                    "time_mean": round(float(ngb_time_arr.mean()), 3),
                    "time_std": round(float(ngb_time_arr.std()), 3),
                    "nll_per_fold": [round(v, 4) for v in ngb_nlls],
                    "rmse_per_fold": [round(v, 4) for v in ngb_rmses],
                    "time_per_fold": [round(v, 3) for v in ngb_times],
                },
                "ngboost_lightning": {
                    "nll_mean": round(float(lb_nll_arr.mean()), 4),
                    "nll_std": round(float(lb_nll_arr.std()), 4),
                    "rmse_mean": round(float(lb_rmse_arr.mean()), 4),
                    "rmse_std": round(float(lb_rmse_arr.std()), 4),
                    "time_mean": round(float(lb_time_arr.mean()), 3),
                    "time_std": round(float(lb_time_arr.std()), 3),
                    "nll_per_fold": [round(v, 4) for v in lb_nlls],
                    "rmse_per_fold": [round(v, 4) for v in lb_rmses],
                    "time_per_fold": [round(v, 3) for v in lb_times],
                },
            }
        )

    # Final summary table
    if len(datasets) > 1:
        print_table(
            "UCI Benchmark Summary: NGBoost vs lightning-boost",
            ["Dataset", "N", "NGB NLL", "LB NLL", "NGB (s)", "LB (s)", "Speedup"],
            summary_rows,
        )

    save_results(
        {
            "config": {"n_est": n_est, "lr": lr, "n_splits": n_splits, "seed": seed},
            "results": all_results,
        },
        "uci",
    )


# ── Run all ──────────────────────────────────────────────────────────


@app.command(name="all")
def run_all() -> None:
    """Run all benchmarks sequentially."""
    total_start = time_mod.perf_counter()

    print("=" * 70)
    print("  lightning-boost benchmark suite")
    print("=" * 70)

    for title, func in [
        ("Training Speed", training),
        ("Inference Speed", inference),
        ("Scaling Curves", scaling),
        ("Full Comparison", comparison),
    ]:
        rule = "\u2500" * 70
        print(f"\n{rule}")
        print(f"  {title}")
        print(rule)
        func()

    total_elapsed = time_mod.perf_counter() - total_start
    print(f"\n{'=' * 70}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print("  All benchmarks passed.")
    print("=" * 70)
