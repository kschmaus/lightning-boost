"""sklearn-compatible classifier for ngboost-lightning."""

from collections.abc import Callable
from collections.abc import Generator
from typing import Any
from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data

from ngboost_lightning.distributions.base import Distribution
from ngboost_lightning.distributions.categorical import Bernoulli
from ngboost_lightning.distributions.categorical import Categorical
from ngboost_lightning.engine import NGBEngine
from ngboost_lightning.engine import build_lgbm_params
from ngboost_lightning.scoring import CRPScore
from ngboost_lightning.scoring import ScoringRule


class LightningBoostClassifier(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Natural gradient boosting classifier powered by LightGBM.

    Outputs full probability distributions over classes by boosting the
    parameters of a categorical distribution using the natural gradient
    of the log-likelihood.

    Internally trains K-1 independent LightGBM boosters (one per logit
    parameter), faithfully replicating the NGBoost algorithm with
    LightGBM's histogram-based splitting for speed.

    Args:
        dist: Distribution class to use. Must be a subclass of
            ``Categorical`` (created via ``k_categorical``). Defaults to
            ``Bernoulli`` (binary classification, K=2). For multiclass,
            use ``k_categorical(K)`` with the appropriate K.
        n_estimators: Number of boosting iterations.
        learning_rate: Outer learning rate applied to each boosting step.
        minibatch_frac: Fraction of training rows to subsample each iteration
            for gradient computation (NGBoost-style minibatch). 1.0 means no
            subsampling.
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
        num_leaves: Maximum number of leaves per tree.
        max_depth: Maximum tree depth. -1 means no limit.
        min_child_samples: Minimum number of samples in a leaf.
        subsample: LightGBM-level row subsampling ratio per tree.
        colsample_bytree: Column subsampling ratio per tree.
        reg_alpha: L1 regularization on leaf weights.
        reg_lambda: L2 regularization on leaf weights.
        lgbm_params: Additional parameters passed to each LightGBM Booster.
        validation_fraction: Fraction of training data to hold out as
            validation for early stopping. If set and ``X_val``/``y_val``
            are not provided to ``fit()``, the training data is
            automatically split. Defaults to ``None`` (no auto-split).

    Attributes:
        engine_: The fitted ``NGBEngine`` instance.
        classes_: Array of unique class labels seen during fit.
        n_classes_: Number of classes.
        n_features_in_: Number of features seen during ``fit``.
        n_estimators_: Actual number of boosting iterations.
        init_params_: Initial distribution parameters from ``dist.fit(y)``.
        scalings_: Line search scale factor per iteration.
        train_loss_: Training NLL per iteration.

    Examples:
        >>> from ngboost_lightning import LightningBoostClassifier
        >>> clf = LightningBoostClassifier(n_estimators=100, learning_rate=0.05)
        >>> clf.fit(X_train, y_train)
        >>> probs = clf.predict_proba(X_test)
        >>> labels = clf.predict(X_test)
    """

    def __init__(
        self,
        dist: type[Distribution] = Bernoulli,
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
        """Initialize the classifier. See class docstring for parameters."""
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
        """Fit the natural gradient boosting classifier.

        Args:
            X: Training features, shape ``[n_samples, n_features]``.
            y: Training class labels, shape ``[n_samples]``.
            X_val: Validation features for early stopping.
            y_val: Validation class labels for early stopping.
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
            ValueError: If the number of classes in y does not match the
                distribution's K, or if a LightGBM parameter appears in
                both a surfaced kwarg and ``lgbm_params``, or if
                weight/validation arguments are inconsistent, or if both
                ``validation_fraction`` and explicit ``X_val``/``y_val``
                are provided.
        """
        X_checked, y_checked = validate_data(self, X, y)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if val_sample_weight is not None:
            val_sample_weight = np.asarray(val_sample_weight, dtype=np.float64)

        # Auto validation split (before label encoding so encoder sees all
        # classes from the full y)
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
                stratify=y_checked,
            )
            if sample_weight is not None:
                X_checked, X_val, y_checked, y_val, sample_weight, val_sample_weight = (
                    splits
                )
            else:
                X_checked, X_val, y_checked, y_val = splits

            if early_stopping_rounds is None:
                early_stopping_rounds = 20

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y_checked).astype(np.float64)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_: int = len(self.classes_)

        # Validate distribution K matches n_classes
        if not issubclass(self.dist, Categorical):
            msg = (
                f"dist must be a Categorical subclass (from k_categorical), "
                f"got {self.dist}"
            )
            raise TypeError(msg)

        expected_k = getattr(self.dist, "K", None)
        if expected_k is not None and expected_k != self.n_classes_:
            msg = (
                f"Distribution expects K={expected_k} classes but y contains "
                f"{self.n_classes_} classes. Use dist=k_categorical({self.n_classes_})."
            )
            raise ValueError(msg)

        if X_val is not None and y_val is not None:
            X_val = check_array(X_val, dtype=np.float64)
            y_val_encoded = self._label_encoder.transform(np.asarray(y_val)).astype(
                np.float64
            )
        else:
            y_val_encoded = None

        merged_lgbm = build_lgbm_params(self, self.lgbm_params)

        if isinstance(self.scoring_rule, CRPScore):
            msg = (
                "CRPScore is not supported for classification. "
                "CRPS is only defined for continuous distributions."
            )
            raise ValueError(msg)

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
            y_encoded,
            X_val=X_val,
            y_val=y_val_encoded,
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

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.integer]:
        """Predict class labels.

        Args:
            X: Features, shape ``[n_samples, n_features]``.

        Returns:
            Predicted class labels, shape ``[n_samples]``.
        """
        check_is_fitted(self)
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        result: NDArray[np.integer] = self.classes_[indices]
        return result

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict class probabilities.

        Args:
            X: Features, shape ``[n_samples, n_features]``.

        Returns:
            Probability matrix, shape ``[n_samples, n_classes]``.
            Each row sums to 1.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=np.float64)
        dist = self.engine_.pred_dist(X_checked)
        assert isinstance(dist, Categorical)
        result: NDArray[np.floating] = dist.probs
        return result

    def pred_dist(self, X: NDArray[np.floating]) -> Categorical:
        """Predict the full conditional distribution.

        Args:
            X: Features, shape ``[n_samples, n_features]``.

        Returns:
            A Categorical distribution instance for all samples.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=np.float64)
        dist = self.engine_.pred_dist(X_checked)
        if not isinstance(dist, Categorical):
            msg = f"Expected Categorical distribution, got {type(dist)}"
            raise TypeError(msg)
        return dist

    def staged_predict(self, X: NDArray[np.floating]) -> Generator[NDArray[np.integer]]:
        """Yield class label predictions after each boosting iteration.

        Args:
            X: Features, shape ``[n_samples, n_features]``.

        Yields:
            Predicted class labels at iteration *i*, shape ``[n_samples]``.
        """
        for probs in self.staged_predict_proba(X):
            indices = np.argmax(probs, axis=1)
            result: NDArray[np.integer] = self.classes_[indices]
            yield result

    def staged_predict_proba(
        self, X: NDArray[np.floating]
    ) -> Generator[NDArray[np.floating]]:
        """Yield class probabilities after each boosting iteration.

        Args:
            X: Features, shape ``[n_samples, n_features]``.

        Yields:
            Probability matrix at iteration *i*,
            shape ``[n_samples, n_classes]``.
        """
        for dist in self.staged_pred_dist(X):
            result: NDArray[np.floating] = dist.probs
            yield result

    def staged_pred_dist(self, X: NDArray[np.floating]) -> Generator[Categorical]:
        """Yield the full conditional distribution after each iteration.

        Args:
            X: Features, shape ``[n_samples, n_features]``.

        Yields:
            Categorical distribution at iteration *i*.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=np.float64)
        for dist in self.engine_.staged_pred_dist(X_checked):
            assert isinstance(dist, Categorical)
            yield dist

    def score(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> float:
        """Negative mean NLL (higher is better).

        Follows the same convention as ``LightningBoostRegressor``: returns
        ``-mean(NLL)`` so that higher values indicate better fit. This is
        consistent with probabilistic scoring but differs from sklearn's
        ``ClassifierMixin.score()`` which returns accuracy.

        Args:
            X: Features, shape ``[n_samples, n_features]``.
            y: True class labels, shape ``[n_samples]``.

        Returns:
            ``-mean(NLL)`` as a float. Higher indicates a better fit.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=np.float64)
        y_encoded = self._label_encoder.transform(np.asarray(y)).astype(np.float64)
        return -self.engine_.scoring_rule.total_score(
            self.engine_.pred_dist(X_checked), y_encoded
        )

    @property
    def feature_importances_(self) -> NDArray[np.floating]:
        """Feature importances per distribution parameter.

        Returns:
            Importance array, shape ``[n_params, n_features]``. Each row
            sums to 1.0 and corresponds to one logit parameter.
        """
        check_is_fitted(self)
        return self.engine_.feature_importances_
