"""Tests for the sklearn-compatible LightningBoostClassifier."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ngboost_lightning.classifier import LightningBoostClassifier
from ngboost_lightning.distributions.categorical import Bernoulli
from ngboost_lightning.distributions.categorical import Categorical
from ngboost_lightning.distributions.categorical import k_categorical
from tests._constants import SEED


@pytest.fixture()
def binary_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Breast cancer train/test split (small subset for speed)."""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=SEED
    )
    return X_train[:300], X_test[:100], y_train[:300], y_test[:100]


@pytest.fixture()
def multiclass_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Iris train/test split (small subset for speed)."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=SEED
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture()
def fitted_binary(
    binary_split: tuple,
) -> tuple[LightningBoostClassifier, np.ndarray, np.ndarray]:
    """Fitted binary classifier with test data."""
    X_train, X_test, y_train, y_test = binary_split
    clf = LightningBoostClassifier(
        n_estimators=15, learning_rate=0.05, verbose=False, random_state=SEED
    )
    clf.fit(X_train, y_train)
    return clf, X_test, y_test


@pytest.fixture()
def fitted_multiclass(
    multiclass_split: tuple,
) -> tuple[LightningBoostClassifier, np.ndarray, np.ndarray]:
    """Fitted multiclass classifier with test data."""
    X_train, X_test, y_train, y_test = multiclass_split
    clf = LightningBoostClassifier(
        dist=k_categorical(3),
        n_estimators=15,
        learning_rate=0.05,
        verbose=False,
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    return clf, X_test, y_test


# ---------- Constructor ----------


class TestConstructor:
    """Tests for constructor behavior and param merging."""

    def test_default_params(self) -> None:
        """Default constructor creates a valid object with expected defaults."""
        clf = LightningBoostClassifier()
        assert clf.dist is Bernoulli
        assert clf.n_estimators == 500
        assert clf.learning_rate == 0.01

    def test_custom_params(self) -> None:
        """All constructor args are stored correctly."""
        clf = LightningBoostClassifier(
            dist=k_categorical(3),
            n_estimators=100,
            learning_rate=0.05,
        )
        assert clf.n_estimators == 100
        assert clf.learning_rate == 0.05
        assert issubclass(clf.dist, Categorical)
        assert clf.dist.K == 3

    def test_lgbm_params_conflict(self) -> None:
        """Raises ValueError when same key in surfaced kwargs and lgbm_params."""
        clf = LightningBoostClassifier(
            num_leaves=64,
            lgbm_params={"num_leaves": 32},
        )
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 5))
        y = rng.integers(0, 2, size=50).astype(float)
        with pytest.raises(ValueError, match="num_leaves"):
            clf.fit(X, y)


# ---------- Binary Fit ----------


class TestBinaryFit:
    """Tests for fitting on binary classification data."""

    def test_fit_returns_self(self, binary_split: tuple) -> None:
        """fit() returns the estimator instance."""
        X_train, _, y_train, _ = binary_split
        clf = LightningBoostClassifier(
            n_estimators=10, learning_rate=0.05, verbose=False
        )
        result = clf.fit(X_train, y_train)
        assert result is clf

    def test_fitted_attributes(self, binary_split: tuple) -> None:
        """After fit, all expected attributes exist."""
        X_train, _, y_train, _ = binary_split
        clf = LightningBoostClassifier(
            n_estimators=10, learning_rate=0.05, verbose=False
        )
        clf.fit(X_train, y_train)
        assert hasattr(clf, "classes_")
        assert clf.n_classes_ == 2
        assert hasattr(clf, "n_estimators_")
        assert hasattr(clf, "init_params_")
        assert hasattr(clf, "boosters_")
        assert clf.n_estimators_ == 10

    def test_fit_with_validation(self, binary_split: tuple) -> None:
        """Fit with X_val/y_val sets val_loss_ and best_val_loss_itr_."""
        X_train, X_test, y_train, y_test = binary_split
        clf = LightningBoostClassifier(
            n_estimators=50, learning_rate=0.05, verbose=False
        )
        clf.fit(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            early_stopping_rounds=10,
        )
        assert hasattr(clf, "val_loss_")
        assert hasattr(clf, "best_val_loss_itr_")
        assert clf.best_val_loss_itr_ is not None


# ---------- Multiclass Fit ----------


class TestMulticlassFit:
    """Tests for fitting on multiclass classification data."""

    def test_fit_multiclass(self, multiclass_split: tuple) -> None:
        """Fitting with k_categorical(3) on 3-class data succeeds."""
        X_train, _, y_train, _ = multiclass_split
        clf = LightningBoostClassifier(
            dist=k_categorical(3),
            n_estimators=10,
            learning_rate=0.05,
            verbose=False,
        )
        clf.fit(X_train, y_train)
        assert len(clf.classes_) == 3
        assert clf.n_classes_ == 3

    def test_dist_mismatch_raises(self, multiclass_split: tuple) -> None:
        """Bernoulli on 3-class data raises ValueError."""
        X_train, _, y_train, _ = multiclass_split
        clf = LightningBoostClassifier(
            dist=Bernoulli,
            n_estimators=10,
            learning_rate=0.05,
            verbose=False,
        )
        with pytest.raises(ValueError, match=r"K=2.*3 classes"):
            clf.fit(X_train, y_train)


# ---------- Predict ----------


class TestPredict:
    """Tests for predict()."""

    def test_predict_shape(self, fitted_binary: tuple) -> None:
        """Predict returns shape [n_test]."""
        clf, X_test, _ = fitted_binary
        preds = clf.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_predict_valid_labels(self, fitted_binary: tuple) -> None:
        """All predictions are valid class labels."""
        clf, X_test, _ = fitted_binary
        preds = clf.predict(X_test)
        assert set(preds).issubset(set(clf.classes_))

    def test_predict_dtype(self, fitted_binary: tuple) -> None:
        """Predictions have integer dtype."""
        clf, X_test, _ = fitted_binary
        preds = clf.predict(X_test)
        assert np.issubdtype(preds.dtype, np.integer)

    def test_predict_before_fit_raises(self) -> None:
        """Predict before fit raises NotFittedError."""
        from sklearn.exceptions import NotFittedError

        clf = LightningBoostClassifier()
        with pytest.raises(NotFittedError):
            clf.predict(np.zeros((5, 3)))


# ---------- Predict Proba ----------


class TestPredictProba:
    """Tests for predict_proba()."""

    def test_proba_shape_binary(self, fitted_binary: tuple) -> None:
        """Binary predict_proba returns shape [n_test, 2]."""
        clf, X_test, _ = fitted_binary
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)

    def test_proba_shape_multiclass(self, fitted_multiclass: tuple) -> None:
        """Multiclass predict_proba returns shape [n_test, 3]."""
        clf, X_test, _ = fitted_multiclass
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)

    def test_proba_sums_to_one(self, fitted_binary: tuple) -> None:
        """Each row of predict_proba sums to 1."""
        clf, X_test, _ = fitted_binary
        proba = clf.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_proba_in_range(self, fitted_binary: tuple) -> None:
        """All probability values are in [0, 1]."""
        clf, X_test, _ = fitted_binary
        proba = clf.predict_proba(X_test)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)


# ---------- Score ----------


class TestScore:
    """Tests for the score method."""

    def test_score_is_finite(self, fitted_binary: tuple) -> None:
        """score() returns a finite float."""
        clf, X_test, y_test = fitted_binary
        s = clf.score(X_test, y_test)
        assert np.isfinite(s)

    def test_score_sign_convention(self, fitted_binary: tuple) -> None:
        """Score is negative since it is -mean(NLL) and NLL > 0."""
        clf, X_test, y_test = fitted_binary
        s = clf.score(X_test, y_test)
        assert s < 0


# ---------- Feature importances ----------


class TestFeatureImportances:
    """Tests for feature_importances_ property."""

    def test_shape_binary(self, fitted_binary: tuple) -> None:
        """Binary classifier has shape [1, n_features]."""
        clf, _, _ = fitted_binary
        fi = clf.feature_importances_
        assert fi.shape == (1, clf.n_features_in_)

    def test_shape_multiclass(self, fitted_multiclass: tuple) -> None:
        """Multiclass classifier has shape [2, n_features] for 3 classes."""
        clf, _, _ = fitted_multiclass
        fi = clf.feature_importances_
        assert fi.shape == (2, clf.n_features_in_)

    def test_nonnegative(self, fitted_binary: tuple) -> None:
        """All importance values are >= 0."""
        clf, _, _ = fitted_binary
        assert np.all(clf.feature_importances_ >= 0)

    def test_normalized(self, fitted_binary: tuple) -> None:
        """Each row sums to 1.0."""
        clf, _, _ = fitted_binary
        row_sums = clf.feature_importances_.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


# ---------- sklearn compatibility ----------


class TestSklearnCompat:
    """Tests for sklearn integration (clone, Pipeline, cross_val_score)."""

    def test_clone_preserves_params(self) -> None:
        """clone() produces an equivalent unfitted estimator."""
        clf = LightningBoostClassifier(
            dist=k_categorical(3),
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=64,
        )
        cloned = clone(clf)
        assert cloned.get_params() == clf.get_params()

    def test_pipeline(self, binary_split: tuple) -> None:
        """Works inside a sklearn Pipeline with StandardScaler."""
        X_train, X_test, y_train, _ = binary_split
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LightningBoostClassifier(
                        n_estimators=10, learning_rate=0.05, verbose=False
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_cross_val_score(self) -> None:
        """cross_val_score runs without error and returns finite scores."""
        data = load_breast_cancer()
        X, y = data.data, data.target
        clf = LightningBoostClassifier(
            n_estimators=15, learning_rate=0.05, verbose=False
        )
        scores = cross_val_score(clf, X, y, cv=3)
        assert len(scores) == 3
        assert np.all(np.isfinite(scores))

    def test_get_set_params_roundtrip(self) -> None:
        """set_params(**get_params()) is idempotent."""
        clf = LightningBoostClassifier(
            n_estimators=100,
            num_leaves=64,
            lgbm_params={"max_bin": 127},
        )
        params = clf.get_params()
        clf2 = LightningBoostClassifier()
        clf2.set_params(**params)
        assert clf2.get_params() == params
