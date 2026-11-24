"""Model construction.

Every data-derived parameter -- the imputation median, the standardisation mean
and scale, the one-hot vocabulary -- lives inside the Pipeline, so that
cross-validation refits the whole chain per fold and nothing leaks from the
validation portion into training.

    engineer ratios
        |-- numeric:     median impute -> PolynomialFeatures(2) -> StandardScaler
        |-- categorical: OneHotEncoder(drop="first")
                |-- TransformedTargetRegressor(StandardScaler on y)
                        |-- OLS / Ridge / Lasso
"""
from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .features import engineer_features


def build_preprocessor(degree: int = 2) -> ColumnTransformer:
    """Impute, expand to degree-2 interactions, scale; one-hot the category."""
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("interactions", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", drop="first", sparse_output=False
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_model(estimator: BaseEstimator, degree: int = 2) -> Pipeline:
    """Wrap an estimator in the full preprocessing chain.

    The target is standardised inside a ``TransformedTargetRegressor``. Penalties
    act on coefficient magnitude, so the useful range of ``alpha`` depends on the
    scale of ``y``; with the target in raw dollars (sigma ~ $115,000) the penalty
    is inert unless ``alpha`` is enormous and the CV curve flattens out.
    Standardising ``y`` puts ``alpha`` on a conventional scale. The transformer
    inverts automatically, so predictions and metrics come back in dollars.
    """
    return Pipeline(
        [
            ("engineer", FunctionTransformer(engineer_features)),
            ("pre", build_preprocessor(degree=degree)),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=estimator, transformer=StandardScaler()
                ),
            ),
        ]
    )


def feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Readable names for the expanded design matrix of a fitted pipeline."""
    return [
        name.split("__", 1)[1]
        for name in fitted_pipeline.named_steps["pre"].get_feature_names_out()
    ]
