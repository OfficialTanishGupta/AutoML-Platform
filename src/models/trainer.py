import pandas as pd

from pandas.api.types import is_numeric_dtype

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import build_preprocessor

CLASSIFICATION_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    ),
}

REGRESSION_MODELS = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    ),
}


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
    model_name: str = "random_forest",
) -> Pipeline:
    """Build and train a complete preprocessing and ML pipeline."""

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    if X_train.empty:
        raise ValueError("Training features are empty.")

    if y_train.empty:
        raise ValueError("Training target is empty.")

    if len(X_train) != len(y_train):
        raise ValueError("Training features and target must have equal row counts.")

    if y_train.isna().any():
        raise ValueError("Training target contains missing values.")

    if task_type == "classification":
        if y_train.nunique() < 2:
            raise ValueError("Classification requires at least two target classes.")

        available_models = CLASSIFICATION_MODELS

    elif task_type == "regression":
        if not is_numeric_dtype(y_train.dtype):
            raise ValueError("Regression target must contain numeric values.")

        available_models = REGRESSION_MODELS

    else:
        raise ValueError("Unsupported task type. Use 'classification' or 'regression'.")

    if model_name not in available_models:
        raise ValueError(f"Unsupported model '{model_name}' for {task_type}.")

    estimator: BaseEstimator = available_models[model_name]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline
