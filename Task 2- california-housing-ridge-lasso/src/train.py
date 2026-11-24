"""Train and compare OLS, Ridge and Lasso on the California housing dataset.

Usage
-----
    python -m src.train                 # full run (~2-3 minutes)
    python -m src.train --quick         # coarse alpha grids, for a smoke test
    python -m src.train --no-plots      # metrics only
    python -m src.train --data path.csv # explicit dataset path
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from .config import (
    CV_FOLDS,
    FIGURE_DIR,
    METRICS_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from .data import load_housing
from .evaluate import (
    bias_by_decile,
    coefficient_summary,
    regression_metrics,
    residual_diagnostics,
)
from .pipeline import build_model, feature_names
from .plots import (
    plot_model_comparison,
    plot_predicted_vs_actual,
    plot_residual_diagnostics,
    plot_validation_curves,
    set_style,
)

warnings.filterwarnings("ignore")

MODELS = ["OLS", "Ridge", "Lasso"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=None, help="path to housing.csv")
    p.add_argument("--quick", action="store_true",
                   help="coarse alpha grids for a fast smoke test")
    p.add_argument("--no-plots", action="store_true", help="skip figure generation")
    return p.parse_args()


def main() -> pd.DataFrame:
    args = parse_args()
    n_alphas = 8 if args.quick else 40
    n_alphas_lasso = 5 if args.quick else 20

    df = load_housing(args.data)
    print(f"Loaded {len(df):,} districts x {df.shape[1]} columns")
    print(f"Missing values: {int(df.isna().sum().sum()):,} "
          f"(all in total_bedrooms, median-imputed inside the pipeline)")
    capped = int((df[TARGET] >= 500_001).sum())
    print(f"Target censored at $500,001 for {capped:,} districts "
          f"({capped/len(df):.2%}) -- a hard floor on achievable RMSE\n")

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"Train {len(X_train):,} / test {len(X_test):,}\n")

    results, predictions, fitted, searches = {}, {}, {}, {}

    def record(name: str, estimator) -> None:
        train_m = regression_metrics(y_train, estimator.predict(X_train))
        test_m = regression_metrics(y_test, estimator.predict(X_test))
        results[name] = {
            "Train RMSE": train_m["RMSE"], "Test RMSE": test_m["RMSE"],
            "Train MAE": train_m["MAE"], "Test MAE": test_m["MAE"],
            "Train R2": train_m["R2"], "Test R2": test_m["R2"],
        }
        predictions[name] = estimator.predict(X_test)
        fitted[name] = estimator
        print(f"{name:<6} test RMSE ${test_m['RMSE']:>10,.0f} | "
              f"MAE ${test_m['MAE']:>9,.0f} | R2 {test_m['R2']:.4f}"
              f"   (train R2 {train_m['R2']:.4f})")

    # --- OLS baseline -------------------------------------------------------
    record("OLS", build_model(LinearRegression()).fit(X_train, y_train))
    names = feature_names(fitted["OLS"])
    print(f"       design matrix: {len(names)} features after degree-2 expansion")

    # --- Ridge (L2) ---------------------------------------------------------
    searches["Ridge"] = GridSearchCV(
        build_model(Ridge(random_state=RANDOM_STATE)),
        {"model__regressor__alpha": np.logspace(-4, 4, n_alphas)},
        cv=cv, scoring="neg_root_mean_squared_error",
        return_train_score=True, n_jobs=1,
    ).fit(X_train, y_train)
    record("Ridge", searches["Ridge"].best_estimator_)

    # --- Lasso (L1) ---------------------------------------------------------
    searches["Lasso"] = GridSearchCV(
        build_model(Lasso(max_iter=5000, tol=1e-3, random_state=RANDOM_STATE)),
        {"model__regressor__alpha": np.logspace(-4, 0.5, n_alphas_lasso)},
        cv=cv, scoring="neg_root_mean_squared_error",
        return_train_score=True, n_jobs=1,
    ).fit(X_train, y_train)
    record("Lasso", searches["Lasso"].best_estimator_)

    # --- selection on cross-validated RMSE, not on the test set -------------
    print("\nSelected on cross-validated RMSE "
          "(the test set is used once, for reporting only):")
    for name, search in searches.items():
        print(f"  {name:<6} best α {search.best_params_['model__regressor__alpha']:<12,.6g}"
              f" CV RMSE ${-search.best_score_:,.0f}")
    best_name = max(searches, key=lambda n: searches[n].best_score_)
    print(f"  -> best model: {best_name}")

    comparison = pd.DataFrame(results).T
    comparison["R2 gap"] = comparison["Train R2"] - comparison["Test R2"]

    print("\nCoefficient shrinkage (standardised units):")
    for name in MODELS:
        s = coefficient_summary(fitted[name].named_steps["model"].regressor_.coef_)
        print(f"  {name:<6} largest |β| {s['largest_abs_beta']:>8.3f} | "
              f"mean {s['mean_abs_beta']:>6.4f} | "
              f"features used {s['n_features_used']:>3}/{len(names)}")

    print(f"\nResidual diagnostics ({best_name}):")
    diag = residual_diagnostics(y_test, predictions[best_name])
    print(f"  skew {diag['skewness']:+.3f} | excess kurtosis "
          f"{diag['excess_kurtosis']:+.3f}")
    print(f"  Breusch-Pagan LM {diag['breusch_pagan_lm']:,.1f} "
          f"(p = {diag['breusch_pagan_p']:.3e}) -> "
          f"{'heteroscedastic' if diag['breusch_pagan_p'] < 0.05 else 'constant variance'}")
    print(f"  p90 |error| ${diag['p90_abs_error']:,.0f} | "
          f"worst miss ${diag['worst_abs_error']:,.0f}")
    print(f"  OLS worst miss for comparison: "
          f"${residual_diagnostics(y_test, predictions['OLS'])['worst_abs_error']:,.0f}")

    bias = bias_by_decile(y_test, predictions[best_name])
    print(f"\nRegression toward the mean (residual by decile of actual value):")
    print(f"  cheapest decile mean residual ${bias.iloc[0]['mean_residual']:>9,.0f} "
          f"(over-predicted)")
    print(f"  dearest  decile mean residual ${bias.iloc[-1]['mean_residual']:>9,.0f} "
          f"(under-predicted)")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(METRICS_PATH)
    print(f"\nMetrics written to {METRICS_PATH.relative_to(PROJECT_ROOT)}")

    if not args.no_plots:
        set_style()
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        plot_validation_curves(searches, FIGURE_DIR / "validation_curves.png")
        plot_model_comparison(comparison, MODELS, FIGURE_DIR / "model_comparison.png")
        plot_predicted_vs_actual(y_test, predictions, results, MODELS,
                                 FIGURE_DIR / "predicted_vs_actual.png")
        plot_residual_diagnostics(y_test, predictions[best_name],
                                  X_test["ocean_proximity"].values, best_name,
                                  FIGURE_DIR / "residual_diagnostics.png")
        print(f"Figures written to {FIGURE_DIR.relative_to(PROJECT_ROOT)}/")

    return comparison


if __name__ == "__main__":
    main()
