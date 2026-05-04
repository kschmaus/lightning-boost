"""sklearn-compatible regressor for ngboost-lightning."""

from collections.abc import Callable
from collections.abc import Generator
from typing import Any
from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data

from ngboost_lightning.distributions.base import Distribution
from ngboost_lightning.distributions.normal import Normal
from ngboost_lightning.engine import NGBEngine
from ngboost_lightning.engine import build_lgbm_params
from ngboost_lightning.scoring import ScoringRule


class LightningBoostRegressor(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Natural gradient boosting regressor powered by LightGBM.

    Outputs full probability distributions (not just point predictions)
    by boosting the parameters of a conditional distribution using the
    natural gradient of the log-likelihood.

    Internally trains K independent LightGBM boosters (one per distribution
    parameter), faithfully replicating the NGBoost algorithm with LightGBM's
    histogram-based splitting for speed.

    Args:
        dist: Distribution class to use. Must be a subclass of
            ``Distribution``. Defaults to ``Normal``.
        n_estimators: Number of boosting iterations.
        learning_rate: Outer learning rate applied to each boosting step.
        minibatch_frac: Fraction of training rows to subsample each iteration
            for gradient computation (NGBoost-style minibatch). 1.0 means no
            subsampling. Distinct from ``subsample``, which controls LightGBM's
            own per-tree row subsampling.
        col_sample: Fraction of columns to subsample each boosting iteration.
            1.0 means no column subsampling. All K parameter-boosters see the
            same feature subset each iteration.
        natural_gradient: Whether to use the natural gradient (True) or
            the ordinary gradient (False).
        tol: Convergence tolerance. Training stops when the mean gradient
            norm falls below this value.
        random_state: Seed for reproducibility (minibatch sampling).
        verbose: Whether to log training progress.
        verbose_eval: Log progress every this many iterations.
        num_leaves: Maximum number of leaves per tree. Primary complexity
            control for LightGBM.
        max_depth: Maximum tree depth. -1 means no limit.
        min_child_samples: Minimum number of samples in a leaf.
        subsample: LightGBM-level row subsampling ratio per tree. Distinct
            from ``minibatch_frac`` which controls gradient subsampling.
        colsample_bytree: Column subsampling ratio per tree.
        reg_alpha: L1 regularization on leaf weights.
        reg_lambda: L2 regularization on leaf weights.
        lgbm_params: Additional parameters passed to each LightGBM Booster.
            Use this for less common LightGBM options (e.g. ``max_bin``,
            ``min_gain_to_split``). If a key conflicts with a surfaced
            constructor kwarg, a ``ValueError`` is raised.
        scoring_rule: The scoring rule used for training. Defaults to
            ``None`` (LogScore / negative log-likelihood). Pass
            ``CRPScore()`` for CRPS-based training.
        validation_fraction: Fraction of training data to hold out as
            validation for early stopping. If set and ``X_val``/``y_val``
            are not provided to ``fit()``, the training data is
            automatically split. Defaults to ``None`` (no auto-split).

    Attributes:
        engine_: The fitted ``NGBEngine`` instance.
        n_features_in_: Number of features seen during ``fit``.
        n_estimators_: Actual number of boosting iterations (may be less
            than ``n_estimators`` due to early stopping or convergence).
        init_params_: Initial distribution parameters from ``dist.fit(y)``.
        scalings_: Line search scale factor per iteration.
        train_loss_: Training NLL per iteration.
        val_loss_: Validation NLL per iteration (only if validation data
            was provided).
        best_val_loss_itr_: Iteration with best validation loss (only if
            validation data was provided).

    Examples:
        >>> from ngboost_lightning import LightningBoostRegressor
        >>> reg = LightningBoostRegressor(n_estimators=100, learning_rate=0.05)
        >>> reg.fit(X_train, y_train)
        >>> preds = reg.predict(X_test)
        >>> dist = reg.pred_dist(X_test)  # full distribution
        >>> dist.scale  # predicted uncertainty
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
        # Surfaced LightGBM params
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        # Escape hatch
        lgbm_params: dict[str, Any] | None = None,
        # Scoring rule
        scoring_rule: ScoringRule | None = None,
        # Auto validation split
        validation_fraction: float | None = None,
    ) -> None:
        """Initialize the regressor. See class docstring for parameters."""
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
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.lgbm_params = lgbm_params
        self.scoring_rule = scoring_rule
        self.validation_fraction = validation_fraction

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
        """Fit the natural gradient boosting model.

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
                Signature: ``(pred_dist, y) -> float``. Replaces the
                default scoring-rule-based training loss for recording
                only (gradients still use the scoring rule).
            val_loss_monitor: Custom callable for computing validation loss.
                Signature: ``(pred_dist, y) -> float``. Replaces the
                default scoring-rule-based validation loss for both
                recording and early stopping decisions.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If a LightGBM parameter appears in both a surfaced
                constructor kwarg and ``lgbm_params``, if weight/validation
                arguments are inconsistent, or if both ``validation_fraction``
                and explicit ``X_val``/``y_val`` are provided.
        """
        X_checked, y_checked = validate_data(self, X, y)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if val_sample_weight is not None:
            val_sample_weight = np.asarray(val_sample_weight, dtype=np.float64)

        # Auto validation split
        has_explicit_val = X_val is not None and y_val is not None
        if self.validation_fraction is not None and has_explicit_val:
            msg = (
                "validation_fraction and explicit X_val/y_val cannot both "
                "be provided. Use one or the other."
            )
            raise ValueError(msg)

        if self.validation_fraction is not None and not has_explicit_val:
            split_arrays = [X_checked, y_checked]
            if sample_weight is not None:
                split_arrays.append(sample_weight)

            splits = train_test_split(
                *split_arrays,
                test_size=self.validation_fraction,
                random_state=self.random_state,
            )
            if sample_weight is not None:
                X_checked, X_val, y_checked, y_val, sample_weight, val_sample_weight = (
                    splits
                )
            else:
                X_checked, X_val, y_checked, y_val = splits

            if early_stopping_rounds is None:
                early_stopping_rounds = 20

        if X_val is not None and y_val is not None:
            X_val = check_array(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.float64)

        merged_lgbm = build_lgbm_params(self, self.lgbm_params)

        self.engine_ = NGBEngine(
            dist=self.dist,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            col_sample=self.col_sample,
            natural_gradient=self.natural_gradient,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
            verbose_eval=self.verbose_eval,
            lgbm_params=merged_lgbm,
            scoring_rule=self.scoring_rule,
        )
        self.engine_.fit(
            X_checked,
            y_checked,
            X_val=X_val,
            y_val=y_val,
            early_stopping_rounds=early_stopping_rounds,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
            train_loss_monitor=train_loss_monitor,
            val_loss_monitor=val_loss_monitor,
        )

        # Copy fitted attributes from engine
        self.init_params_: NDArray[np.floating] = self.engine_.init_params_
        self.scalings_: list[float] = self.engine_.scalings_
        self.train_loss_: list[float] = self.engine_.train_loss_
        self.n_estimators_: int = self.engine_.n_estimators_
        self.boosters_ = self.engine_.boosters_

        if hasattr(self.engine_, "val_loss_"):
            self.val_loss_: list[float] = self.engine_.val_loss_
            self.best_val_loss_itr_: int | None = self.engine_.best_val_loss_itr_

        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Point prediction (conditional mean).

        Args:
            X: Features, shape ``[n_samples, n_features]``.

        Returns:
            Predictions, shape ``[n_samples]``.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=np.float64)
        return self.engine_.predict(X_checked)

    def pred_dist(self, X: NDArray[np.floating]) -> Distribution:
        """Predict the full conditional distribution.

        This is the primary probabilistic output. The returned distribution
        object provides ``mean``, ``scale``, ``cdf``, ``ppf``, ``sample``,
        and other methods.

        Args:
            X: Features, shape ``[n_samples, n_features]``.

        Returns:
            A Distribution instance for all samples.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=np.float64)
        return self.engine_.pred_dist(X_checked)

    def staged_predict(
        self, X: NDArray[np.floating]
    ) -> Generator[NDArray[np.floating]]:
        """Yield point predictions after each boosting iteration.

        Args:
            X: Features, shape ``[n_samples, n_features]``.

        Yields:
            Predictions at iteration *i*, shape ``[n_samples]``.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=np.float64)
        yield from self.engine_.staged_predict(X_checked)

    def staged_pred_dist(self, X: NDArray[np.floating]) -> Generator[Distribution]:
        """Yield the full conditional distribution after each iteration.

        Args:
            X: Features, shape ``[n_samples, n_features]``.

        Yields:
            Distribution at iteration *i*.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=np.float64)
        yield from self.engine_.staged_pred_dist(X_checked)

    def score(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> float:
        """Negative mean score (higher is better).

        Uses the scoring rule from training (LogScore or CRPScore).
        Follows sklearn's convention that ``score()`` returns a value where
        higher is better, making it compatible with ``cross_val_score`` and
        other sklearn utilities.

        Args:
            X: Features, shape ``[n_samples, n_features]``.
            y: Target values, shape ``[n_samples]``.

        Returns:
            ``-mean(score)`` as a float. Higher indicates a better fit.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=np.float64)
        y_checked = np.asarray(y, dtype=np.float64)
        return -self.engine_.scoring_rule.total_score(
            self.engine_.pred_dist(X_checked), y_checked
        )

    @property
    def feature_importances_(self) -> NDArray[np.floating]:
        """Feature importances per distribution parameter.

        Returns:
            Importance array, shape ``[n_params, n_features]``. Each row
            sums to 1.0 and corresponds to one distribution parameter
            (e.g. row 0 = mean, row 1 = log_scale for Normal).
        """
        check_is_fitted(self)
        return self.engine_.feature_importances_
