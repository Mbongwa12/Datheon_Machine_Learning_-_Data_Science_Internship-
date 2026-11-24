"""Figures.

One fixed colour per model across every figure, so a series never changes
identity between charts. The correlation matrix uses a diverging blue-grey-red
ramp: zero correlation must read as "nothing", which a rainbow cannot do.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

C_OLS, C_RIDGE, C_LASSO = "#2a78d6", "#eb6834", "#1baf7a"
MODEL_COLOR = {"OLS": C_OLS, "Ridge": C_RIDGE, "Lasso": C_LASSO}

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#7d1b1a"]
)
NEG, POS = "#2a78d6", "#e34948"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": INK,
            "font.size": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "figure.dpi": 110,
        }
    )


def thousands(x, _=None) -> str:
    return f"{x/1000:,.0f}k"


def plot_validation_curves(searches: dict, path) -> None:
    """CV RMSE against alpha, one panel per penalised model.

    Separate panels rather than a shared axis: the two alpha scales are not
    comparable (scikit-learn normalises the Lasso loss by 1/2n and Ridge not),
    and a dual-axis chart would invite exactly that false comparison.
    """
    fig, axes = plt.subplots(1, len(searches), figsize=(6 * len(searches), 4.1))
    axes = np.atleast_1d(axes)
    for ax, (name, search) in zip(axes, searches.items()):
        res = search.cv_results_
        alphas = res["param_model__regressor__alpha"].data.astype(float)
        order = np.argsort(alphas)
        alphas = alphas[order]
        val = -res["mean_test_score"][order]
        std = res["std_test_score"][order]
        train = -res["mean_train_score"][order]
        color = MODEL_COLOR[name]

        ax.fill_between(alphas, val - std, val + std, color=color, alpha=0.15, lw=0)
        ax.plot(alphas, val, color=color, label=f"{name} — 5-fold CV")
        ax.plot(alphas, train, color=MUTED, lw=1.5, ls="--", label="training")
        best = search.best_params_["model__regressor__alpha"]
        ax.axvline(best, color=INK, lw=1, ls=":")
        ax.annotate(
            f"best α = {best:,.4g}\nCV RMSE ${-search.best_score_:,.0f}",
            xy=(best, -search.best_score_),
            xytext=(10, 22),
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
            fontweight="bold",
        )
        ax.set_xscale("log")
        ax.set_title(f"{name} validation curve")
        ax.set_xlabel("α (log scale)")
        ax.set_ylabel("RMSE (USD)")
        ax.yaxis.set_major_formatter(thousands)
        ax.legend(loc="upper left")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comparison, models, path) -> None:
    """Dollar errors and R^2 in separate panels -- different units, never one axis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.55, 1]})

    ax = axes[0]
    metrics = ["Test RMSE", "Test MAE"]
    xpos = np.arange(len(metrics))
    width = 0.26
    for i, m in enumerate(models):
        vals = [comparison.loc[m, k] for k in metrics]
        bars = ax.bar(
            xpos + (i - 1) * width, vals, width * 0.92,
            color=MODEL_COLOR[m], label=m, edgecolor=SURFACE, linewidth=2,
        )
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 900, f"${v:,.0f}",
                    ha="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(xpos, ["RMSE", "MAE"])
    ax.set_ylabel("error (USD)")
    ax.yaxis.set_major_formatter(thousands)
    ax.set_title("Test-set error — lower is better")
    ax.set_ylim(0, comparison["Test RMSE"].max() * 1.16)
    ax.legend(loc="upper right", ncols=3)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    vals = [comparison.loc[m, "Test R2"] for m in models]
    bars = ax.bar(models, vals, 0.5, color=[MODEL_COLOR[m] for m in models],
                  edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.4f}",
                ha="center", fontsize=9, color=INK_2, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_ylabel("R²")
    ax.set_title("Test-set R² — higher is better")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_vs_actual(y_test, predictions, results, models, path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(4.7 * len(models), 4.6),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lims = (0, 520_000)
    for ax, name in zip(axes, models):
        ax.scatter(y_test, predictions[name], s=7, alpha=0.22,
                   color=MODEL_COLOR[name], edgecolors="none", rasterized=True)
        ax.plot(lims, lims, color=INK, lw=1.2, ls="--")
        ax.axvline(500_001, color=MUTED, lw=1, ls=":")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_xlabel("actual value (USD)")
        ax.xaxis.set_major_formatter(thousands)
        ax.yaxis.set_major_formatter(thousands)
        m = results[name]
        ax.text(0.04, 0.955,
                f"R² = {m['Test R2']:.4f}\nRMSE = ${m['Test RMSE']:,.0f}\n"
                f"MAE = ${m['Test MAE']:,.0f}",
                transform=ax.transAxes, va="top", fontsize=8.5,
                color=INK, linespacing=1.5)
    axes[0].set_ylabel("predicted value (USD)")
    fig.suptitle("Predicted vs actual — dashed line is perfect prediction",
                 fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_residual_diagnostics(y_test, y_pred, categories, model_name, path) -> None:
    resid = np.asarray(y_test) - np.asarray(y_pred)
    y_pred = np.asarray(y_pred)
    color = MODEL_COLOR[model_name]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))

    ax = axes[0, 0]
    ax.scatter(y_pred, resid, s=7, alpha=0.22, color=color,
               edgecolors="none", rasterized=True)
    ax.axhline(0, color=INK, lw=1.2, ls="--")
    bins = np.linspace(y_pred.min(), y_pred.max(), 22)
    idx = np.digitize(y_pred, bins)
    centres = [(bins[i - 1] + bins[i]) / 2 for i in range(1, len(bins))]
    means = [resid[idx == i].mean() if (idx == i).sum() > 15 else np.nan
             for i in range(1, len(bins))]
    ax.plot(centres, means, color=POS, lw=2, label="binned mean residual")
    ax.set_title(f"(a) Residuals vs fitted — {model_name}")
    ax.set_xlabel("predicted value (USD)")
    ax.set_ylabel("residual (USD)")
    ax.xaxis.set_major_formatter(thousands)
    ax.yaxis.set_major_formatter(thousands)
    ax.legend(loc="lower left")

    ax = axes[0, 1]
    ax.hist(resid, bins=70, density=True, color=color, alpha=0.45,
            edgecolor=SURFACE, linewidth=0.4)
    xs = np.linspace(resid.min(), resid.max(), 400)
    ax.plot(xs, stats.gaussian_kde(resid)(xs), color=color, lw=2.2,
            label="empirical KDE")
    ax.plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std()), color=INK,
            lw=1.6, ls="--", label="normal reference")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_title("(b) Residual distribution curve")
    ax.set_xlabel("residual (USD)")
    ax.set_ylabel("density")
    ax.xaxis.set_major_formatter(thousands)
    ax.set_yticks([])
    ax.legend(loc="upper left")

    ax = axes[1, 0]
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax.scatter(osm, osr, s=7, alpha=0.4, color=color, edgecolors="none",
               rasterized=True)
    ax.plot(osm, slope * osm + intercept, color=INK, lw=1.4, ls="--")
    ax.set_title("(c) Normal Q–Q plot of residuals")
    ax.set_xlabel("theoretical quantiles")
    ax.set_ylabel("observed residual (USD)")
    ax.yaxis.set_major_formatter(thousands)

    ax = axes[1, 1]
    cats = np.asarray(categories)
    order = [c for c in ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
             if (cats == c).sum() > 0]
    box = ax.boxplot(
        [resid[cats == c] for c in order], patch_artist=True, widths=0.55,
        flierprops=dict(marker="o", markersize=2.5, alpha=0.3,
                        markerfacecolor=MUTED, markeredgecolor="none"),
        medianprops=dict(color=INK, linewidth=1.6),
        whiskerprops=dict(color=AXIS), capprops=dict(color=AXIS),
    )
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(SURFACE)
    ax.axhline(0, color=INK, lw=1.2, ls="--")
    ax.set_xticks(range(1, len(order) + 1),
                  [f"{c}\n(n={int((cats == c).sum())})" for c in order], fontsize=8)
    ax.set_title("(d) Residuals by ocean proximity")
    ax.set_ylabel("residual (USD)")
    ax.yaxis.set_major_formatter(thousands)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
