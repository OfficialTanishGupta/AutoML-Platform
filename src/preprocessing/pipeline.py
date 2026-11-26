import pandas as pd

from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build an automated preprocessing pipeline.

    Numerical features:
        - Median imputation
        - Standard scaling

    Categorical features:
        - Most frequent imputation
        - One-hot encoding

    Pass only feature columns, excluding the target column.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("The input dataset is empty.")

    if X.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate column names.")

    if any(str(column).strip() == "" for column in X.columns):
        raise ValueError("The dataset contains empty column names.")

    numeric_columns = [
        column
        for column in X.columns
        if is_numeric_dtype(X[column].dtype) and not is_bool_dtype(X[column].dtype)
    ]

    datetime_columns = [
        column for column in X.columns if is_datetime64_any_dtype(X[column].dtype)
    ]

    if datetime_columns:
        raise ValueError(
            "Datetime columns require feature engineering before "
            f"preprocessing: {datetime_columns}"
        )

    categorical_columns = [
        column for column in X.columns if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )

        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                    ),
                ),
            ]
        )

        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
