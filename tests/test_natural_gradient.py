"""Direct numerical comparison of gradient computation between libraries."""

import numpy as np
import pytest
from ngboost.distns.normal import Normal as NGBNormal
from ngboost.manifold import manifold
from ngboost.scores import LogScore

from ngboost_lightning import LightningBoostRegressor
from ngboost_lightning.distributions import Normal
from tests._constants import SEED


@pytest.fixture()
def shared_params() -> tuple[np.ndarray, np.ndarray]:
    """Shared parameters and targets for gradient comparison.

    Returns:
        (params_row_major [n_samples, 2], y [n_samples])
    """
    rng = np.random.default_rng(SEED)
    n = 100
    loc = rng.normal(2.0, 1.0, size=n)
    log_scale = rng.normal(0.0, 0.3, size=n)
    params = np.column_stack([loc, log_scale])
    y = rng.normal(loc, np.exp(log_scale))
    return params, y


def _ngb_manifold(params_row: np.ndarray) -> object:
    """Construct NGBoost manifold object from row-major params."""
    # NGBoost wants [n_params, n_samples]
    ngb_params = params_row.T
    ngb_manifold_cls = manifold(LogScore, NGBNormal)
    return ngb_manifold_cls(ngb_params)


# ---------- Gradient parity ----------


class TestGradientParity:
    """Compare d_score, metric, and natural_gradient between libraries."""

    def test_d_score_matches_ngboost(self, shared_params: tuple) -> None:
        """Ordinary gradient should match NGBoost element-wise."""
        params, y = shared_params
        lb_dist = Normal(params)
        lb_grad = lb_dist.d_score(y)

        ngb_m = _ngb_manifold(params)
        ngb_grad = ngb_m.d_score(y)

        np.testing.assert_allclose(lb_grad, ngb_grad, atol=1e-10)

    def test_metric_matches_ngboost(self, shared_params: tuple) -> None:
        """Fisher Information should match NGBoost element-wise."""
        params, _ = shared_params
        lb_dist = Normal(params)
        lb_fi = lb_dist.metric()

        ngb_m = _ngb_manifold(params)
        ngb_fi = ngb_m.metric()

        np.testing.assert_allclose(lb_fi, ngb_fi, atol=1e-10)

    def test_natural_gradient_matches_ngboost(self, shared_params: tuple) -> None:
        """Natural gradient should match NGBoost element-wise."""
        params, y = shared_params
        lb_dist = Normal(params)
        lb_ng = lb_dist.natural_gradient(y)

        ngb_m = _ngb_manifold(params)
        ngb_ng = ngb_m.grad(y, natural=True)

        np.testing.assert_allclose(lb_ng, ngb_ng, atol=1e-10)


# ---------- Natural gradient effect ----------


class TestNaturalGradientEffect:
    """Test that natural vs ordinary gradient both converge for Normal."""

    def test_natural_vs_ordinary_same_for_normal(self) -> None:
        """Both gradient types should converge to similar losses.

        For Normal(mean, log_scale), the Fisher Information is diagonal,
        so the natural gradient is just a rescaled ordinary gradient. The
        tree fits differ in magnitude but both should converge.
        """
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split

        cal = fetch_california_housing()
        X_tr, _, y_tr, _ = train_test_split(
            cal.data, cal.target, test_size=0.8, random_state=SEED
        )
        X_tr, y_tr = X_tr[:500], y_tr[:500]

        reg_nat = LightningBoostRegressor(
            n_estimators=100,
            learning_rate=0.05,
            natural_gradient=True,
            random_state=SEED,
            verbose=False,
        )
        reg_nat.fit(X_tr, y_tr)

        reg_ord = LightningBoostRegressor(
            n_estimators=100,
            learning_rate=0.05,
            natural_gradient=False,
            random_state=SEED,
            verbose=False,
        )
        reg_ord.fit(X_tr, y_tr)

        # Both should converge (final loss < initial loss)
        assert reg_nat.train_loss_[-1] < reg_nat.train_loss_[0]
        assert reg_ord.train_loss_[-1] < reg_ord.train_loss_[0]

        # Natural gradient should be at least as good as ordinary
        assert reg_nat.train_loss_[-1] <= reg_ord.train_loss_[-1] * 1.05
