"""sklearn-compatible survival estimator for ngboost-lightning."""

from collections.abc import Callable
from collections.abc import Generator
from typing import TYPE_CHECKING
from typing import Any
from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data

from ngboost_lightning.distributions.base import Distribution
from ngboost_lightning.distributions.lognormal import LogNormal
from ngboost_lightning.engine import NGBEngine
from ngboost_lightning.engine import build_lgbm_params
from ngboost_lightning.survival import CensoredLogScore
from ngboost_lightning.survival import Y_from_censored

if TYPE_CHECKING:
    from ngboost_lightning.scoring import ScoringRule


class LightningBoostSurvival(BaseEstimator):  # type: ignore[misc]
    """Natural gradient boosting for survival analysis with right censoring.

    Outputs full probability distributions over survival times by boosting
    the parameters of a conditional distribution using the natural gradient
    of the censored log-likelihood.

    Supports right-censored observations: uncensored samples contribute
    ``-logpdf(T)`` to the loss while censored samples contribute
    ``-logsf(T) = -log(1 - CDF(T))``.

    The distribution must implement ``logsf(y)`` (Exponential, LogNormal,
    and Weibull support this).

    Args:
        dist: Distribution class to use. Must support ``logsf``.
            Defaults to ``LogNormal``.
        n_estimators: Number of boosting iterations.
        learning_rate: Outer learning rate applied to each boosting step.
        minibatch_frac: Fraction of training rows to subsample each iteration
            for gradient computation. 1.0 means no subsampling.
        col_sample: Fraction of columns to subsample each boosting iteration.
            1.0 means no column subsampling. All K parameter-boosters see the
            same feature subset each iteration.
        natural_gradient: Whether to use the natural gradient.
        tol: Convergence tolerance.
        random_state: Seed for reproducibility.
        verbose: Whether to log training progress.
        verbose_eval: Log progress every this many iterations.
        num_leaves: Maximum number of leaves per tree.
        max_depth: Maximum tree depth. -1 means no limit.
        min_child_samples: Minimum number of samples in a leaf.
        subsample: LightGBM-level row subsampling ratio per tree.
        colsample_bytree: Column subsampling ratio per tree.
        reg_alpha: L1 regularization on leaf weights.
        reg_lambda: L2 regularization on leaf weights.
        lgbm_params: Additional parameters passed to each LightGBM Booster.
        validation_fraction: Fraction of training data to hold out for
            early stopping. Defaults to ``None`` (no auto-split).

    Attributes:
        engine_: The fitted ``NGBEngine`` instance.
        n_features_in_: Number of features seen during ``fit``.
        n_estimators_: Actual number of boosting iterations.

    Examples:
        >>> from ngboost_lightning import LightningBoostSurvival
        >>> surv = LightningBoostSurvival(n_estimators=100, learning_rate=0.05)
        >>> surv.fit(X_train, T_train, E_train)
        >>> median_times = surv.predict(X_test)
        >>> dist = surv.pred_dist(X_test)
    """

    def __init__(
        self,
        dist: type[Distribution] = LogNormal,
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
        # Auto validation split
        validation_fraction: float | None = None,
    ) -> None:
        """Initialize the survival estimator. See class docstring for params."""
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
        self.validation_fraction = validation_fraction

    def fit(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.floating],
        E: NDArray[np.integer] | NDArray[np.bool_],
        X_val: NDArray[np.floating] | None = None,
        T_val: NDArray[np.floating] | None = None,
        E_val: NDArray[np.integer] | NDArray[np.bool_] | None = None,
        early_stopping_rounds: int | None = None,
        sample_weight: NDArray[np.floating] | None = None,
        val_sample_weight: NDArray[np.floating] | None = None,
        train_loss_monitor: Callable[[Distribution, NDArray[np.floating]], float]
        | None = None,
        val_loss_monitor: Callable[[Distribution, NDArray[np.floating]], float]
        | None = None,
    ) -> Self:
        """Fit the survival model on right-censored data.

        Args:
            X: Training features, shape ``[n_samples, n_features]``.
            T: Times to event or censoring, shape ``[n_samples]``.
            E: Event indicators, shape ``[n_samples]``.
                ``E[i] = 1`` means observed, ``E[i] = 0`` means censored.
            X_val: Validation features for early stopping.
            T_val: Validation times for early stopping.
            E_val: Validation event indicators for early stopping.
            early_stopping_rounds: Stop if validation loss hasn't improved
                for this many consecutive iterations.
            sample_weight: Per-sample training weights, shape ``[n_samples]``.
            val_sample_weight: Per-sample validation weights.
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
            ValueError: If both ``validation_fraction`` and explicit
                validation data are provided.
        """
        X_checked, T_checked = validate_data(self, X, T)
        E_checked = np.asarray(E, dtype=bool)
        Y = Y_from_censored(T_checked, E_checked)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if val_sample_weight is not None:
            val_sample_weight = np.asarray(val_sample_weight, dtype=np.float64)

        # Auto validation split
        has_explicit_val = X_val is not None and T_val is not None and E_val is not None
        if self.validation_fraction is not None and has_explicit_val:
            msg = (
                "validation_fraction and explicit X_val/T_val/E_val cannot "
                "both be provided. Use one or the other."
            )
            raise ValueError(msg)

        Y_val = None
        if self.validation_fraction is not None and not has_explicit_val:
            split_arrays = [X_checked, Y]
            if sample_weight is not None:
                split_arrays.append(sample_weight)

            splits = train_test_split(
                *split_arrays,
                test_size=self.validation_fraction,
                random_state=self.random_state,
            )
            if sample_weight is not None:
                X_checked, X_val, Y, Y_val, sample_weight, val_sample_weight = splits
            else:
                X_checked, X_val, Y, Y_val = splits

            if early_stopping_rounds is None:
                early_stopping_rounds = 20

        if has_explicit_val:
            assert T_val is not None
            assert E_val is not None
            X_val = check_array(X_val, dtype=np.float64)
            Y_val = Y_from_censored(
                np.asarray(T_val, dtype=np.float64),
                np.asarray(E_val, dtype=bool),
            )

        merged_lgbm = build_lgbm_params(self, self.lgbm_params)

        scoring_rule: ScoringRule = CensoredLogScore()  # type: ignore[assignment]

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
            scoring_rule=scoring_rule,
        )
        self.engine_.fit(
            X_checked,
            Y,  # type: ignore[arg-type]
            X_val=X_val,
            y_val=Y_val,
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
        """Predict median survival time.

        Args:
            X: Features, shape ``[n_samples, n_features]``.

        Returns:
            Median survival times, shape ``[n_samples]``.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=np.float64)
        dist = self.engine_.pred_dist(X_checked)
        return dist.ppf(np.full(len(X_checked), 0.5))

    def pred_dist(self, X: NDArray[np.floating]) -> Distribution:
        """Predict the full conditional survival distribution.

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
        """Yield median survival times after each boosting iteration.

        Args:
            X: Features, shape ``[n_samples, n_features]``.

        Yields:
            Median survival times at iteration *i*, shape ``[n_samples]``.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=np.float64)
        for dist in self.engine_.staged_pred_dist(X_checked):
            yield dist.ppf(np.full(len(X_checked), 0.5))

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

    def score(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.floating],
        E: NDArray[np.integer] | NDArray[np.bool_],
    ) -> float:
        """Negative mean censored NLL (higher is better).

        Args:
            X: Features, shape ``[n_samples, n_features]``.
            T: Times to event or censoring, shape ``[n_samples]``.
            E: Event indicators, shape ``[n_samples]``.

        Returns:
            ``-mean(censored_NLL)`` as a float.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=np.float64)
        Y = Y_from_censored(
            np.asarray(T, dtype=np.float64),
            np.asarray(E, dtype=bool),
        )
        return -self.engine_.scoring_rule.total_score(
            self.engine_.pred_dist(X_checked),
            Y,  # type: ignore[arg-type]
        )

    @property
    def feature_importances_(self) -> NDArray[np.floating]:
        """Feature importances per distribution parameter.

        Returns:
            Importance array, shape ``[n_params, n_features]``.
        """
        check_is_fitted(self)
        return self.engine_.feature_importances_
