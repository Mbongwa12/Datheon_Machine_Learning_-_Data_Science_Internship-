"""Feature engineering.

Layer 1 (this module): domain ratio features.
Layer 2 (pipeline.py): a degree-2 polynomial expansion over all numeric columns.

The ratios are built *before* imputation so that the missing values inherited
from ``total_bedrooms`` are filled by the pipeline's imputer, which is fitted on
training folds only.
"""
from __future__ import annotations

import pandas as pd


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add per-household / per-room ratios to the raw district counts.

    The raw counts (``total_rooms``, ``population``, ...) are district-size
    proxies and correlate with each other at r = 0.86-0.98 while carrying almost
    no signal about price (|r| <= 0.14). Normalising them turns them into
    quantities that describe the housing stock itself.
    """
    X = X.copy()
    X["rooms_per_household"] = X["total_rooms"] / X["households"]
    X["bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"]
    X["population_per_household"] = X["population"] / X["households"]
    return X
