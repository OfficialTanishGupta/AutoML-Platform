from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from src.data.loader import validate_dataset


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Generate an automatic profile of a dataset."""

    validate_dataset(df)

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    datetime_columns = [
        column for column in df.columns if is_datetime64_any_dtype(df[column])
    ]

    categorical_columns = [
        column
        for column in df.columns
        if column not in numeric_columns and column not in datetime_columns
    ]

    missing_values = {
        str(column): int(count) for column, count in df.isnull().sum().items()
    }

    numeric_summary = {}

    if numeric_columns:
        numeric_summary = df[numeric_columns].describe().round(2).to_dict()

    categorical_summary = {}

    for column in categorical_columns:
        value_counts = df[column].value_counts(dropna=True)

        categorical_summary[str(column)] = {
            "unique_values": int(df[column].nunique()),
            "most_frequent_value": (
                str(value_counts.index[0]) if not value_counts.empty else None
            ),
            "most_frequent_count": (
                int(value_counts.iloc[0]) if not value_counts.empty else 0
            ),
        }

    return {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "numeric_columns": [str(col) for col in numeric_columns],
        "categorical_columns": [str(col) for col in categorical_columns],
        "datetime_columns": [str(col) for col in datetime_columns],
        "missing_values": missing_values,
        "total_missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
    }
