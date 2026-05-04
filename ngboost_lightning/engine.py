"""Core natural gradient boosting engine for ngboost-lightning."""

import logging
from collections.abc import Callable
from collections.abc import Generator
from typing import Any
from typing import Self

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from ngboost_lightning.distributions.base import Distribution
from ngboost_lightning.distributions.normal import Normal
from ngboost_lightning.scoring import LogScore
from ngboost_lightning.scoring import ScoringRule
from ngboost_lightning.survival import _is_censored_y

logger = logging.getLogger(__name__)

_DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "num_leaves": 31,
    "verbose": -1,
}

SURFACED_LGBM_KEYS: frozenset[str] = frozenset(
    {
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    }
)


def build_lgbm_params(
    estimator: object,
    lgbm_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge surfaced LightGBM kwargs from an estimator with extra params.

    Args:
        estimator: An sklearn-style estimator whose attributes include the
            keys in :data:`SURFACED_LGBM_KEYS`.
        lgbm_params: Additional LightGBM parameters. If a key conflicts
            with a surfaced key, a ``ValueError`` is raised.

    Returns:
        Merged parameter dictionary for LightGBM.
    """
    merged: dict[str, Any] = {
        key: getattr(estimator, key) for key in SURFACED_LGBM_KEYS
    }

    if lgbm_params:
        conflicts = set(lgbm_params) & SURFACED_LGBM_KEYS
        if conflicts:
            msg = (
                f"Parameters {sorted(conflicts)} appear in both "
                f"constructor kwargs and lgbm_params. "
                f"Use one or the other, not both."
            )
            raise ValueError(msg)
        merged.update(lgbm_params)

    return merged


class NGBEngine:
    """Core natural gradient boosting engine.

    Implements the boosting loop with K independent LightGBM boosters (one
    per distribution parameter), faithfully replicating NGBoost's algorithm.

    This is an internal class. Users should use
    :class:`~ngboost_lightning.LightningBoostRegressor` instead.

    Args:
        dist: Distribution class to use. Must be a subclass of Distribution.
        n_estimators: Number of boosting iterations.
        learning_rate: Outer learning rate applied to each boosting step.
        minibatch_frac: Fraction of rows to subsample each iteration.
            1.0 means no subsampling.
        col_sample: Fraction of columns to subsample each iteration.
            1.0 means no column subsampling. All K parameter-boosters
            see the same feature subset each iteration.
        natural_gradient: Whether to use the natural gradient (True) or
            the ordinary gradient (False).
        tol: Numerical tolerance for the line search. When the mean norm of
            the scaled residuals falls below this, the line search stops.
        random_state: Seed for reproducibility (minibatch sampling).
        verbose: Whether to print training progress.
        verbose_eval: Print progress every this many iterations.
        lgbm_params: Additional parameters passed to each LightGBM Booster.
            ``objective`` is always overridden to ``"none"``.
        scoring_rule: The scoring rule used for training. Defaults to
            ``LogScore()`` (negative log-likelihood). Pass ``CRPScore()``
            for CRPS-based training.
    """

    def __init__(
        self,
        dist: type[Distribution] = Normal,
        n_estimators: int = 500,
        learning_rate: float = 0.01,
        minibatch_frac: float = 1.0,
        col_sample: float = 1.0,
        natural_gradient: bool = True,
        tol: float = 1e-4,
        random_state: int | None = None,
        verbose: bool = True,
        verbose_eval: int = 100,
        lgbm_params: dict[str, Any] | None = None,
        scoring_rule: ScoringRule | None = None,
    ) -> None:
        """Initialize the engine.

        See class docstring for parameter descriptions.
        """
        self.dist = dist
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.natural_gradient = natural_gradient
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.lgbm_params = lgbm_params or {}
        self.scoring_rule: ScoringRule = scoring_rule or LogScore()

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
        early_stopping_rounds: int | None = None,
        sample_weight: NDArray[np.floating] | None = None,
        val_sample_weight: NDArray[np.floating] | None = None,
        train_loss_monitor: Callable[[Distribution, NDArray[np.floating]], float]
        | None = None,
        val_loss_monitor: Callable[[Distribution, NDArray[np.floating]], float]
        | None = None,
    ) -> Self:
        """Fit the boosting model.

        Args:
            X: Training features, shape ``[n_samples, n_features]``.
            y: Training targets, shape ``[n_samples]``.
            X_val: Validation features for early stopping.
            y_val: Validation targets for early stopping.
            early_stopping_rounds: Stop if validation loss hasn't improved
                for this many consecutive iterations.
            sample_weight: Per-sample training weights, shape ``[n_samples]``.
                If ``None``, all samples are weighted equally.
            val_sample_weight: Per-sample validation weights,
                shape ``[n_val_samples]``. Required when both
                ``sample_weight`` and validation data are provided.
            train_loss_monitor: Custom callable for computing training loss.
                Signature: ``(pred_dist, y) -> float``. When provided,
                replaces the default scoring-rule-based training loss for
                recording purposes only (gradients still use the scoring
                rule).
            val_loss_monitor: Custom callable for computing validation loss.
                Signature: ``(pred_dist, y) -> float``. When provided,
                replaces the default scoring-rule-based validation loss for
                both recording and early stopping decisions.

        Returns:
            self

        Raises:
            ValueError: If ``sample_weight`` is provided with validation data
                but ``val_sample_weight`` is not, or vice versa.
        """
        X = np.asarray(X, dtype=np.float64)
        if not _is_censored_y(y):
            y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X.shape
        n_params = self.dist.n_params
        rng = np.random.default_rng(self.random_state)

        # Validate and normalize sample weights
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        # Initial parameters from marginal distribution
        # For censored data, fit on observed times only
        if _is_censored_y(y):
            fit_y = y["Time"][y["Event"]]  # type: ignore[call-overload]
            fit_w = sample_weight[y["Event"]] if sample_weight is not None else None  # type: ignore[call-overload]
        else:
            fit_y = y
            fit_w = sample_weight
        self.init_params_ = self.dist.fit(fit_y, sample_weight=fit_w)
        params = np.tile(self.init_params_, (n_samples, 1))

        # Prepare LightGBM parameters
        merged_params = {**_DEFAULT_LGBM_PARAMS, **self.lgbm_params}
        merged_params["objective"] = "none"
        # Force LightGBM learning_rate=1 so trees aren't internally shrunk.
        # Step size is controlled by our own learning_rate * line_search_scale.
        merged_params["learning_rate"] = 1.0

        # Create K datasets sharing feature binning
        datasets = self._create_datasets(X, n_params, merged_params)

        # Create K boosters
        boosters: list[lgb.Booster] = [
            lgb.Booster(merged_params, ds) for ds in datasets
        ]

        # Training state
        self.scalings_: list[float] = []
        self.train_loss_: list[float] = []
        self.col_indices_: list[NDArray[np.integer]] = []
        self.n_features_in_ = n_features

        # Column subsampling setup
        if not 0.0 < self.col_sample <= 1.0:
            msg = f"col_sample must be in (0, 1], got {self.col_sample}"
            raise ValueError(msg)
        use_col_sample = self.col_sample < 1.0
        n_cols_select = max(1, int(np.ceil(self.col_sample * n_features)))

        # Validation state
        has_val = X_val is not None and y_val is not None
        if has_val and sample_weight is not None and val_sample_weight is None:
            msg = (
                "sample_weight is provided but val_sample_weight is not. "
                "Pass val_sample_weight when using sample_weight with "
                "validation data."
            )
            raise ValueError(msg)
        if has_val and sample_weight is None and val_sample_weight is not None:
            msg = (
                "val_sample_weight is provided but sample_weight is not. "
                "Pass sample_weight when using val_sample_weight."
            )
            raise ValueError(msg)

        val_params: NDArray[np.floating] | None = None
        if has_val:
            assert X_val is not None
            assert y_val is not None
            X_val = np.asarray(X_val, dtype=np.float64)
            if not _is_censored_y(y_val):
                y_val = np.asarray(y_val, dtype=np.float64)
            if val_sample_weight is not None:
                val_sample_weight = np.asarray(val_sample_weight, dtype=np.float64)
            val_params = np.tile(self.init_params_, (len(X_val), 1))
            self.val_loss_: list[float] = []
            best_val_loss = np.inf
            self.best_val_loss_itr_: int | None = None

        for itr in range(self.n_estimators):
            # --- Minibatch sampling ---
            if self.minibatch_frac < 1.0:
                batch_size = max(1, int(self.minibatch_frac * n_samples))
                batch_idx = rng.choice(n_samples, size=batch_size, replace=False)
                batch_params = params[batch_idx]
                batch_y = y[batch_idx]
                batch_w = (
                    sample_weight[batch_idx] if sample_weight is not None else None
                )
            else:
                batch_idx = None
                batch_params = params
                batch_y = y
                batch_w = sample_weight

            # --- Compute gradients ---
            dist_obj = self.dist(batch_params)
            if train_loss_monitor is not None:
                train_loss = train_loss_monitor(dist_obj, batch_y)
            else:
                train_loss = self.scoring_rule.total_score(
                    dist_obj, batch_y, sample_weight=batch_w
                )
            self.train_loss_.append(train_loss)

            if self.natural_gradient:
                grads = self.scoring_rule.natural_gradient(dist_obj, batch_y)
            else:
                grads = self.scoring_rule.d_score(dist_obj, batch_y)

            # --- Column subsampling ---
            if use_col_sample:
                col_idx = np.sort(
                    rng.choice(n_features, size=n_cols_select, replace=False)
                )
                self.col_indices_.append(col_idx)
                col_mask = np.ones(n_features, dtype=bool)
                col_mask[col_idx] = False
                # Zero out non-selected columns in-place so LightGBM
                # trees cannot split on them during update().
                saved_cols = X[:, col_mask].copy()
                X[:, col_mask] = 0.0
            else:
                self.col_indices_.append(np.arange(n_features))

            # --- Update each booster ---
            # When using minibatch, scatter batch gradients into full-size
            # arrays (zeros for non-batch rows) so we can keep a single
            # booster accumulating trees on the full-data Dataset.
            if batch_idx is not None:
                full_grads = np.zeros((n_samples, n_params))
                full_grads[batch_idx] = grads
                full_w: NDArray[np.floating] | None = None
                if sample_weight is not None:
                    # Zero weight for non-batch rows so they don't
                    # influence the tree fit.
                    full_w = np.zeros(n_samples)
                    full_w[batch_idx] = sample_weight[batch_idx]
                for k in range(n_params):
                    fobj_k = self._make_fobj(full_grads[:, k], full_w)
                    boosters[k].update(fobj=fobj_k)
            else:
                for k in range(n_params):
                    fobj_k = self._make_fobj(grads[:, k], sample_weight)
                    boosters[k].update(fobj=fobj_k)

            # --- Get this iteration's tree predictions on ALL data ---
            resids = np.column_stack(
                [
                    np.asarray(
                        b.predict(
                            X, raw_score=True, start_iteration=itr, num_iteration=1
                        )
                    )
                    for b in boosters
                ]
            )

            # --- Restore masked columns ---
            if use_col_sample:
                X[:, col_mask] = saved_cols

            # --- Line search ---
            scale = self._line_search(resids, params, y, sample_weight=sample_weight)
            self.scalings_.append(scale)

            # --- Update parameters ---
            params -= self.learning_rate * scale * resids

            # --- Validation ---
            if has_val:
                assert X_val is not None
                assert y_val is not None
                assert val_params is not None
                # Apply same column mask to validation data
                if use_col_sample:
                    saved_val_cols = X_val[:, col_mask].copy()
                    X_val[:, col_mask] = 0.0
                val_resids = np.column_stack(
                    [
                        np.asarray(
                            b.predict(
                                X_val,
                                raw_score=True,
                                start_iteration=itr,
                                num_iteration=1,
                            )
                        )
                        for b in boosters
                    ]
                )
                if use_col_sample:
                    X_val[:, col_mask] = saved_val_cols
                val_params -= self.learning_rate * scale * val_resids
                val_dist = self.dist(val_params)
                if val_loss_monitor is not None:
                    val_loss = val_loss_monitor(val_dist, y_val)
                else:
                    val_loss = self.scoring_rule.total_score(
                        val_dist,
                        y_val,
                        sample_weight=val_sample_weight,
                    )
                self.val_loss_.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_val_loss_itr_ = itr

                if (
                    early_stopping_rounds is not None
                    and len(self.val_loss_) > early_stopping_rounds
                    and best_val_loss < min(self.val_loss_[-early_stopping_rounds:])
                ):
                    if self.verbose:
                        logger.info(
                            "Early stopping at iteration %d "
                            "(best val_loss=%.4f at iteration %d)",
                            itr,
                            best_val_loss,
                            self.best_val_loss_itr_,
                        )
                    break

            # --- Verbose ---
            if self.verbose and self.verbose_eval > 0 and itr % self.verbose_eval == 0:
                grad_norm = np.linalg.norm(grads, axis=1).mean() * scale
                msg = (
                    f"[iter {itr}] loss={train_loss:.4f} "
                    f"scale={scale:.4f} norm={grad_norm:.4f}"
                )
                if has_val:
                    msg += f" val_loss={self.val_loss_[-1]:.4f}"
                logger.info(msg)

            # --- Gradient norm convergence check ---
            if np.linalg.norm(grads, axis=1).mean() < self.tol:
                if self.verbose:
                    logger.info(
                        "Converged at iteration %d (gradient norm below tol)",
                        itr,
                    )
                break

        self.boosters_ = boosters
        self.n_estimators_ = len(self.scalings_)
        return self

    @property
    def feature_importances_(self) -> NDArray[np.floating]:
        """Feature importances per distribution parameter.

        Returns:
            Importance array, shape ``[n_params, n_features]``. Each row
            sums to 1.0 and corresponds to one distribution parameter.
        """
        rows = []
        for k in range(self.dist.n_params):
            imp = self.boosters_[k].feature_importance(importance_type="gain")
            imp = imp.astype(np.float64)
            total = imp.sum()
            if total > 0:
                imp /= total
            rows.append(imp)
        return np.stack(rows)

    def predict_params(
        self,
        X: NDArray[np.floating],
        n_iterations: int | None = None,
    ) -> NDArray[np.floating]:
        """Predict internal distribution parameters.

        Args:
            X: Features, shape ``[n_samples, n_features]``.
            n_iterations: Number of boosting iterations to use.
                If ``None``, uses all fitted iterations.

        Returns:
            Parameters, shape ``[n_samples, n_params]``.
        """
        X = np.asarray(X, dtype=np.float64)
        params = np.tile(self.init_params_, (len(X), 1))
        use_col_sample = self.col_sample < 1.0

        n_itr = n_iterations if n_iterations is not None else self.n_estimators_
        if use_col_sample:
            X_pred = X.copy()
        for itr in range(n_itr):
            scale = self.scalings_[itr]
            if use_col_sample:
                col_idx = self.col_indices_[itr]
                X_pred[:] = 0.0
                X_pred[:, col_idx] = X[:, col_idx]
                X_iter = X_pred
            else:
                X_iter = X
            resids = np.column_stack(
                [
                    np.asarray(
                        b.predict(
                            X_iter,
                            raw_score=True,
                            start_iteration=itr,
                            num_iteration=1,
                        )
                    )
                    for b in self.boosters_
                ]
            )
            params -= self.learning_rate * scale * resids

        return params

    def pred_dist(
        self,
        X: NDArray[np.floating],
        n_iterations: int | None = None,
    ) -> Distribution:
        """Predict the full conditional distribution.

        Args:
            X: Features, shape ``[n_samples, n_features]``.
            n_iterations: Number of boosting iterations to use.
                If ``None``, uses all fitted iterations.

        Returns:
            A Distribution instance for all samples.
        """
        return self.dist(self.predict_params(X, n_iterations=n_iterations))

    def predict(
        self,
        X: NDArray[np.floating],
        n_iterations: int | None = None,
    ) -> NDArray[np.floating]:
        """Point prediction (conditional mean).

        Args:
            X: Features, shape ``[n_samples, n_features]``.
            n_iterations: Number of boosting iterations to use.
                If ``None``, uses all fitted iterations.

        Returns:
            Predictions, shape ``[n_samples]``.
        """
        return self.pred_dist(X, n_iterations=n_iterations).mean()

    def staged_predict_params(
        self, X: NDArray[np.floating]
    ) -> Generator[NDArray[np.floating]]:
        """Yield distribution parameters after each boosting iteration.

        Args:
            X: Features, shape ``[n_samples, n_features]``.

        Yields:
            Parameters at iteration *i*, shape ``[n_samples, n_params]``.
        """
        X = np.asarray(X, dtype=np.float64)
        params = np.tile(self.init_params_, (len(X), 1))
        use_col_sample = self.col_sample < 1.0

        if use_col_sample:
            X_pred = X.copy()
        for itr in range(self.n_estimators_):
            scale = self.scalings_[itr]
            if use_col_sample:
                col_idx = self.col_indices_[itr]
                X_pred[:] = 0.0
                X_pred[:, col_idx] = X[:, col_idx]
                X_iter = X_pred
            else:
                X_iter = X
            resids = np.column_stack(
                [
                    np.asarray(
                        b.predict(
                            X_iter,
                            raw_score=True,
                            start_iteration=itr,
                            num_iteration=1,
                        )
                    )
                    for b in self.boosters_
                ]
            )
            params = params - self.learning_rate * scale * resids
            yield params.copy()

    def staged_pred_dist(self, X: NDArray[np.floating]) -> Generator[Distribution]:
        """Yield the full conditional distribution after each iteration.

        Args:
            X: Features, shape ``[n_samples, n_features]``.

        Yields:
            Distribution at iteration *i*.
        """
        for params in self.staged_predict_params(X):
            yield self.dist(params)

    def staged_predict(
        self, X: NDArray[np.floating]
    ) -> Generator[NDArray[np.floating]]:
        """Yield point predictions (conditional mean) after each iteration.

        Args:
            X: Features, shape ``[n_samples, n_features]``.

        Yields:
            Predictions at iteration *i*, shape ``[n_samples]``.
        """
        for dist in self.staged_pred_dist(X):
            yield dist.mean()

    def _line_search(
        self,
        resids: NDArray[np.floating],
        params: NDArray[np.floating],
        y: NDArray[np.floating],
        scale_init: float = 1.0,
        sample_weight: NDArray[np.floating] | None = None,
    ) -> float:
        """Find optimal scaling for the boosting step.

        Replicates NGBoost's line search: scale up by 2x until loss worsens,
        then scale down by 0.5x until loss improves.

        Args:
            resids: Tree predictions for current iteration,
                shape ``[n_samples, n_params]``.
            params: Current parameter values, shape ``[n_samples, n_params]``.
            y: Target values, shape ``[n_samples]``.
            scale_init: Starting scale factor.
            sample_weight: Per-sample weights for weighted loss evaluation.

        Returns:
            Optimal scale factor.
        """
        loss_init = self.scoring_rule.total_score(
            self.dist(params), y, sample_weight=sample_weight
        )
        scale = scale_init

        # Scale up
        while True:
            scaled = resids * scale
            candidate = params - scaled
            loss = self.scoring_rule.total_score(
                self.dist(candidate), y, sample_weight=sample_weight
            )
            if not np.isfinite(loss) or loss > loss_init or scale > 256:
                break
            scale *= 2

        # Scale down
        while True:
            scaled = resids * scale
            candidate = params - scaled
            loss = self.scoring_rule.total_score(
                self.dist(candidate), y, sample_weight=sample_weight
            )
            norm = float(np.mean(np.linalg.norm(scaled, axis=1)))
            if norm < self.tol:
                break
            if np.isfinite(loss) and loss < loss_init:
                break
            scale *= 0.5

        return scale

    @staticmethod
    def _make_fobj(
        grad_column: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> Callable[
        [NDArray[np.floating], lgb.Dataset],
        tuple[NDArray[np.floating], NDArray[np.floating]],
    ]:
        """Create a custom objective function for a single booster.

        The returned function ignores LightGBM's internal predictions and
        returns the **negated** gradient component with hessian set to
        sample weights (or ones if unweighted).

        LightGBM convention: ``fobj`` returns ``(grad, hess)`` of the loss
        w.r.t. the prediction.  The tree then predicts ``-grad / hess``, i.e.
        the descent direction.  Since we want the tree to approximate
        ``+grad_column`` (the natural-gradient ascent direction of the NLL),
        we pass ``-grad_column * hess`` so the tree output is
        ``+grad_column``.  When ``sample_weight`` is provided, setting
        ``hess = weight`` makes the tree fit a weighted MSE on the gradient
        targets.

        Args:
            grad_column: Gradient for one parameter, shape ``[n_samples]``.
            sample_weight: Per-sample weights, shape ``[n_samples]``.
                If ``None``, all samples have weight 1.

        Returns:
            A callable with signature ``(preds, dataset) -> (grad, hess)``.
        """
        if sample_weight is not None:
            hess = sample_weight.copy()
            neg_grad = -grad_column * sample_weight
        else:
            hess = np.ones_like(grad_column)
            neg_grad = -grad_column

        def fobj(
            _preds: NDArray[np.floating], _dataset: lgb.Dataset
        ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
            return neg_grad, hess

        return fobj

    @staticmethod
    def _create_datasets(
        X: NDArray[np.floating],
        n_params: int,
        lgbm_params: dict[str, Any],
    ) -> list[lgb.Dataset]:
        """Create K LightGBM Datasets sharing feature binning.

        Args:
            X: Feature matrix, shape ``[n_samples, n_features]``.
            n_params: Number of distribution parameters (K).
            lgbm_params: Parameters for Dataset construction.

        Returns:
            List of K Datasets.
        """
        n = len(X)
        ds0 = lgb.Dataset(X, label=np.zeros(n), params=lgbm_params, free_raw_data=False)
        ds0.construct()
        datasets = [ds0]
        for _ in range(1, n_params):
            ds_k = lgb.Dataset(
                X,
                label=np.zeros(n),
                reference=ds0,
                params=lgbm_params,
                free_raw_data=False,
            )
            datasets.append(ds_k)
        return datasets
