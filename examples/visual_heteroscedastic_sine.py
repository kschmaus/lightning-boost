"""Visual comparison on a 1D heteroscedastic sine wave.

Generates synthetic data where variance grows with x, fits both
NGBoost and lightning-boost, and produces a Plotly figure showing
predicted means and 90% prediction intervals side-by-side.
"""

import numpy as np
import plotly.graph_objects as go
from ngboost import NGBRegressor
from ngboost.distns import Normal as NGBNormal
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split

from ngboost_lightning import LightningBoostRegressor

# ── Synthetic data: y = sin(x) + heteroscedastic noise ──────────────
rng = np.random.default_rng(24601)
n = 1000
x = rng.uniform(0, 2 * np.pi, size=n)
noise_std = 0.1 + 0.4 * x / (2 * np.pi)
y = np.sin(x) + rng.normal(0, noise_std)

X = x.reshape(-1, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# ── Common hyperparameters ───────────────────────────────────────────
PARAMS = dict(n_estimators=500, learning_rate=0.05, random_state=42)
EARLY_STOPPING_ROUNDS = 20

# ── Fit NGBoost ──────────────────────────────────────────────────────
print("Training NGBoost...")
ngb = NGBRegressor(Dist=NGBNormal, verbose=False, **PARAMS)
ngb.fit(
    X_train,
    y_train,
    X_val=X_val,
    Y_val=y_val,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
)

# ── Fit lightning-boost (matched tree capacity) ──────────────────────
print("Training lightning-boost...")
lb = LightningBoostRegressor(
    verbose=False, num_leaves=8, max_depth=3, min_child_samples=1, **PARAMS
)
lb.fit(
    X_train,
    y_train,
    X_val=X_val,
    y_val=y_val,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
)

# ── Dense prediction grid ────────────────────────────────────────────
x_grid = np.linspace(0, 2 * np.pi, 300)
X_grid = x_grid.reshape(-1, 1)

ngb_dist = ngb.pred_dist(X_grid)
lb_dist = lb.pred_dist(X_grid)

ngb_mean = ngb.predict(X_grid)
lb_mean = lb.predict(X_grid)

ngb_lo, ngb_hi = ngb_dist.ppf(0.05), ngb_dist.ppf(0.95)
lb_lo, lb_hi = lb_dist.ppf(0.05), lb_dist.ppf(0.95)

# True envelope
true_mean = np.sin(x_grid)
true_std = 0.1 + 0.4 * x_grid / (2 * np.pi)

# ── Goodness-of-fit on validation set ────────────────────────────────
ngb_val_dist = ngb.pred_dist(X_val)
lb_val_dist = lb.pred_dist(X_val)

ngb_nll = float(-ngb_val_dist.logpdf(y_val).mean())
lb_nll = float(lb_val_dist.total_score(y_val))

ngb_val_lo, ngb_val_hi = ngb_val_dist.ppf(0.05), ngb_val_dist.ppf(0.95)
lb_val_lo, lb_val_hi = lb_val_dist.ppf(0.05), lb_val_dist.ppf(0.95)

ngb_cov = float(np.mean((y_val >= ngb_val_lo) & (y_val <= ngb_val_hi)))
lb_cov = float(np.mean((y_val >= lb_val_lo) & (y_val <= lb_val_hi)))

# ── Plotly figure ────────────────────────────────────────────────────
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        f"NGBoost  (NLL={ngb_nll:.3f}, 90% cov={ngb_cov:.1%})",
        f"lightning-boost  (NLL={lb_nll:.3f}, 90% cov={lb_cov:.1%})",
    ),
    shared_yaxes=True,
    horizontal_spacing=0.05,
)

for col, (_name, mean, lo, hi) in enumerate(
    [
        ("NGBoost", ngb_mean, ngb_lo, ngb_hi),
        ("lightning-boost", lb_mean, lb_lo, lb_hi),
    ],
    start=1,
):
    show_legend = col == 1

    # 90% prediction interval
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_grid, x_grid[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself",
            fillcolor="rgba(99,110,250,0.2)",
            line=dict(color="rgba(99,110,250,0)"),
            name="90% PI",
            legendgroup="pi",
            showlegend=show_legend,
        ),
        row=1,
        col=col,
    )

    # True 90% interval
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_grid, x_grid[::-1]]),
            y=np.concatenate(
                [true_mean + 1.645 * true_std, (true_mean - 1.645 * true_std)[::-1]]
            ),
            fill="toself",
            fillcolor="rgba(0,0,0,0)",
            line=dict(color="gray", dash="dot", width=1),
            name="True 90% interval",
            legendgroup="true_pi",
            showlegend=show_legend,
        ),
        row=1,
        col=col,
    )

    # Predicted mean
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=mean,
            mode="lines",
            line=dict(color="rgb(99,110,250)", width=2),
            name="Predicted mean",
            legendgroup="mean",
            showlegend=show_legend,
        ),
        row=1,
        col=col,
    )

    # True mean
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=true_mean,
            mode="lines",
            line=dict(color="gray", width=1.5, dash="dash"),
            name="True mean",
            legendgroup="true",
            showlegend=show_legend,
        ),
        row=1,
        col=col,
    )

    # Training data
    fig.add_trace(
        go.Scatter(
            x=X_train.ravel(),
            y=y_train,
            mode="markers",
            marker=dict(size=3, color="black", opacity=0.3),
            name="Training data",
            legendgroup="data",
            showlegend=show_legend,
        ),
        row=1,
        col=col,
    )

fig.update_xaxes(title_text="x", row=1, col=1)
fig.update_xaxes(title_text="x", matches="x", row=1, col=2)
fig.update_yaxes(title_text="y", row=1, col=1)
fig.update_yaxes(matches="y", row=1, col=2)

fig.update_layout(
    title="Heteroscedastic Sine Wave: NGBoost vs lightning-boost",
    height=450,
    width=950,
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
)

fig.write_html("heteroscedastic_sine.html")
print("Saved heteroscedastic_sine.html")
fig.show()
