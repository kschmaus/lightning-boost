"""Tests for staged prediction generators."""

import numpy as np
import pytest

from ngboost_lightning import LightningBoostClassifier
from ngboost_lightning import LightningBoostRegressor
from ngboost_lightning.engine import NGBEngine
from ngboost_lightning.scoring import LogScore
from tests._constants import SEED

N_ESTIMATORS = 20


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Simple regression dataset."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(200, 5))
    y = 3 * X[:, 0] + rng.normal(scale=0.5, size=200)
    return X, y


@pytest.fixture()
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Simple binary classification dataset."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + X[:, 1] + rng.normal(scale=0.3, size=200) > 0).astype(float)
    return X, y


@pytest.fixture()
def fitted_engine(regression_data: tuple) -> tuple[NGBEngine, np.ndarray]:
    """Fitted engine with default Normal distribution."""
    X, y = regression_data
    eng = NGBEngine(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        verbose=False,
        random_state=SEED,
    )
    eng.fit(X, y)
    return eng, X


# ===================================================================
# Engine n_iterations
# ===================================================================


class TestNIterations:
    """Test n_iterations parameter on engine prediction methods."""

    def test_n_iterations_none_uses_all(self, fitted_engine: tuple) -> None:
        """None uses all fitted iterations."""
        eng, X = fitted_engine
        np.testing.assert_allclose(
            eng.predict_params(X[:5]),
            eng.predict_params(X[:5], n_iterations=N_ESTIMATORS),
        )

    def test_n_iterations_partial(self, fitted_engine: tuple) -> None:
        """Partial iterations differ from full."""
        eng, X = fitted_engine
        params_5 = eng.predict_params(X[:5], n_iterations=5)
        params_all = eng.predict_params(X[:5])
        assert not np.allclose(params_5, params_all)

    def test_n_iterations_1(self, fitted_engine: tuple) -> None:
        """Single iteration produces something different from init."""
        eng, X = fitted_engine
        params_1 = eng.predict_params(X[:5], n_iterations=1)
        init = np.tile(eng.init_params_, (5, 1))
        assert not np.allclose(params_1, init)

    def test_pred_dist_n_iterations(self, fitted_engine: tuple) -> None:
        """pred_dist passes n_iterations through."""
        eng, X = fitted_engine
        dist_5 = eng.pred_dist(X[:5], n_iterations=5)
        dist_all = eng.pred_dist(X[:5])
        assert not np.allclose(dist_5.mean(), dist_all.mean())

    def test_predict_n_iterations(self, fitted_engine: tuple) -> None:
        """Predict passes n_iterations through."""
        eng, X = fitted_engine
        pred_5 = eng.predict(X[:5], n_iterations=5)
        pred_all = eng.predict(X[:5])
        assert not np.allclose(pred_5, pred_all)


# ===================================================================
# Engine staged generators
# ===================================================================


class TestEngineStagedPredict:
    """Tests for staged generators on NGBEngine."""

    def test_staged_predict_length(self, fitted_engine: tuple) -> None:
        """Length equals n_estimators_."""
        eng, X = fitted_engine
        stages = list(eng.staged_predict(X[:5]))
        assert len(stages) == eng.n_estimators_

    def test_staged_predict_params_length(self, fitted_engine: tuple) -> None:
        """Length equals n_estimators_."""
        eng, X = fitted_engine
        stages = list(eng.staged_predict_params(X[:5]))
        assert len(stages) == eng.n_estimators_

    def test_staged_pred_dist_length(self, fitted_engine: tuple) -> None:
        """Length equals n_estimators_."""
        eng, X = fitted_engine
        stages = list(eng.staged_pred_dist(X[:5]))
        assert len(stages) == eng.n_estimators_

    def test_final_matches_predict(self, fitted_engine: tuple) -> None:
        """Last staged prediction equals predict()."""
        eng, X = fitted_engine
        stages = list(eng.staged_predict(X[:10]))
        np.testing.assert_allclose(stages[-1], eng.predict(X[:10]))

    def test_final_params_match(self, fitted_engine: tuple) -> None:
        """Last staged params equals predict_params()."""
        eng, X = fitted_engine
        stages = list(eng.staged_predict_params(X[:10]))
        np.testing.assert_allclose(stages[-1], eng.predict_params(X[:10]))

    def test_staged_matches_n_iterations(self, fitted_engine: tuple) -> None:
        """Staged prediction at iteration i equals n_iterations=i+1."""
        eng, X = fitted_engine
        stages = list(eng.staged_predict_params(X[:5]))
        for i in [0, 4, 9, N_ESTIMATORS - 1]:
            np.testing.assert_allclose(
                stages[i],
                eng.predict_params(X[:5], n_iterations=i + 1),
            )

    def test_loss_decreases_over_stages(
        self, regression_data: tuple, fitted_engine: tuple
    ) -> None:
        """Training loss generally decreases over staged predictions."""
        _, y = regression_data
        eng, X = fitted_engine
        scoring = LogScore()
        losses = [scoring.total_score(d, y) for d in eng.staged_pred_dist(X)]
        # First loss should be higher than last
        assert losses[0] > losses[-1]
        # Check monotone decrease for first 5 iterations
        for i in range(min(5, len(losses) - 1)):
            assert losses[i] >= losses[i + 1] - 1e-6


# ===================================================================
# Regressor staged generators
# ===================================================================


class TestRegressorStaged:
    """Tests for staged generators on LightningBoostRegressor."""

    def test_staged_predict_length(self, regression_data: tuple) -> None:
        """Length equals n_estimators_."""
        X, y = regression_data
        reg = LightningBoostRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
        )
        reg.fit(X, y)
        stages = list(reg.staged_predict(X[:5]))
        assert len(stages) == reg.n_estimators_

    def test_staged_predict_final_matches(self, regression_data: tuple) -> None:
        """Last staged prediction equals predict()."""
        X, y = regression_data
        reg = LightningBoostRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
        )
        reg.fit(X, y)
        stages = list(reg.staged_predict(X[:10]))
        np.testing.assert_allclose(stages[-1], reg.predict(X[:10]))

    def test_staged_pred_dist_final_matches(self, regression_data: tuple) -> None:
        """Last staged dist matches pred_dist()."""
        X, y = regression_data
        reg = LightningBoostRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
        )
        reg.fit(X, y)
        staged_dists = list(reg.staged_pred_dist(X[:10]))
        final_dist = reg.pred_dist(X[:10])
        np.testing.assert_allclose(staged_dists[-1].mean(), final_dist.mean())

    def test_staged_predict_before_fit_raises(self) -> None:
        """Calling staged_predict before fit raises."""
        reg = LightningBoostRegressor(n_estimators=5)
        with pytest.raises(Exception):  # noqa: B017
            list(reg.staged_predict(np.zeros((5, 3))))


# ===================================================================
# Classifier staged generators
# ===================================================================


class TestClassifierStaged:
    """Tests for staged generators on LightningBoostClassifier."""

    def test_staged_predict_proba_length(self, classification_data: tuple) -> None:
        """Length equals n_estimators_."""
        X, y = classification_data
        clf = LightningBoostClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
        )
        clf.fit(X, y)
        stages = list(clf.staged_predict_proba(X[:5]))
        assert len(stages) == clf.n_estimators_

    def test_staged_predict_proba_final_matches(
        self, classification_data: tuple
    ) -> None:
        """Last staged proba equals predict_proba()."""
        X, y = classification_data
        clf = LightningBoostClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
        )
        clf.fit(X, y)
        stages = list(clf.staged_predict_proba(X[:10]))
        np.testing.assert_allclose(stages[-1], clf.predict_proba(X[:10]))

    def test_staged_predict_labels_final_matches(
        self, classification_data: tuple
    ) -> None:
        """Last staged labels equals predict()."""
        X, y = classification_data
        clf = LightningBoostClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
        )
        clf.fit(X, y)
        stages = list(clf.staged_predict(X[:10]))
        np.testing.assert_array_equal(stages[-1], clf.predict(X[:10]))

    def test_staged_pred_dist_final_matches(self, classification_data: tuple) -> None:
        """Last staged dist matches pred_dist()."""
        X, y = classification_data
        clf = LightningBoostClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
        )
        clf.fit(X, y)
        staged_dists = list(clf.staged_pred_dist(X[:10]))
        final_dist = clf.pred_dist(X[:10])
        np.testing.assert_allclose(staged_dists[-1].probs, final_dist.probs)

    def test_staged_predict_proba_sums_to_one(self, classification_data: tuple) -> None:
        """All staged probabilities sum to 1."""
        X, y = classification_data
        clf = LightningBoostClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
        )
        clf.fit(X, y)
        for probs in clf.staged_predict_proba(X[:10]):
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)


# ===================================================================
# Early-stopped models
# ===================================================================


class TestEarlyStoppedStaged:
    """Staged predictions work with early-stopped models."""

    def test_regressor_early_stopped(self, regression_data: tuple) -> None:
        """Stages match n_estimators_ after early stopping."""
        X, y = regression_data
        n = len(y)
        reg = LightningBoostRegressor(
            n_estimators=200,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
        )
        reg.fit(
            X[: n - 40],
            y[: n - 40],
            X_val=X[n - 40 :],
            y_val=y[n - 40 :],
            early_stopping_rounds=5,
        )
        # Should have stopped early
        assert reg.n_estimators_ < 200
        stages = list(reg.staged_predict(X[:5]))
        assert len(stages) == reg.n_estimators_
        np.testing.assert_allclose(stages[-1], reg.predict(X[:5]))

    def test_classifier_early_stopped(self, classification_data: tuple) -> None:
        """Stages match n_estimators_ after early stopping."""
        X, y = classification_data
        n = len(y)
        clf = LightningBoostClassifier(
            n_estimators=200,
            learning_rate=0.05,
            verbose=False,
            random_state=SEED,
        )
        clf.fit(
            X[: n - 40],
            y[: n - 40],
            X_val=X[n - 40 :],
            y_val=y[n - 40 :],
            early_stopping_rounds=5,
        )
        assert clf.n_estimators_ < 200
        stages = list(clf.staged_predict_proba(X[:5]))
        assert len(stages) == clf.n_estimators_
        np.testing.assert_allclose(stages[-1], clf.predict_proba(X[:5]))
