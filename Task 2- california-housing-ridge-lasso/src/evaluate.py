"""Metrics and residual diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE and MAE in dollars, plus R^2."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def coefficient_summary(coefs: np.ndarray) -> dict[str, float]:
    """Shrinkage diagnostics in standardised units."""
    abs_c = np.abs(coefs)
    return {
        "largest_abs_beta": float(abs_c.max()),
        "mean_abs_beta": float(abs_c.mean()),
        "n_above_one": int((abs_c > 1).sum()),
        "n_exactly_zero": int((abs_c < 1e-10).sum()),
        "n_features_used": int((abs_c >= 1e-10).sum()),
    }


def residual_diagnostics(y_true, y_pred) -> dict[str, float]:
    """Normality, skew, tails and the Breusch-Pagan heteroscedasticity test."""
    resid = np.asarray(y_true) - np.asarray(y_pred)
    lm_stat, lm_pvalue, _, _ = het_breuschpagan(
        resid, sm.add_constant(np.asarray(y_pred))
    )
    return {
        "mean": float(resid.mean()),
        "std": float(resid.std()),
        "skewness": float(stats.skew(resid)),
        "excess_kurtosis": float(stats.kurtosis(resid)),
        "p90_abs_error": float(np.percentile(np.abs(resid), 90)),
        "p99_abs_error": float(np.percentile(np.abs(resid), 99)),
        "worst_abs_error": float(np.abs(resid).max()),
        "breusch_pagan_lm": float(lm_stat),
        "breusch_pagan_p": float(lm_pvalue),
    }


def bias_by_decile(y_true, y_pred) -> pd.DataFrame:
    """Mean residual within deciles of the *actual* target.

    A monotonic trend here is regression toward the mean: the model
    over-predicts cheap districts and under-predicts expensive ones.
    """
    y_true = np.asarray(y_true)
    resid = y_true - np.asarray(y_pred)
    edges = np.quantile(y_true, np.linspace(0, 1, 11))
    rows = []
    for i in range(10):
        upper = y_true <= edges[i + 1] if i == 9 else y_true < edges[i + 1]
        mask = (y_true >= edges[i]) & upper
        rows.append(
            {
                "decile": i + 1,
                "actual_from": edges[i],
                "actual_to": edges[i + 1],
                "n": int(mask.sum()),
                "mean_residual": float(resid[mask].mean()),
            }
        )
    return pd.DataFrame(rows)
