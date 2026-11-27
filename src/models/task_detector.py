import pandas as pd

from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


def detect_task_type(
    target: pd.Series,
    max_unique_values: int = 20,
    max_unique_ratio: float = 0.05,
) -> str:
    """Automatically detect whether a target requires classification or regression."""

    if not isinstance(target, pd.Series):
        raise TypeError("Target must be a pandas Series.")

    if target.empty:
        raise ValueError("Target column is empty.")

    if is_datetime64_any_dtype(target.dtype):
        raise ValueError(
            "Datetime targets are not supported. "
            "Convert the target into a suitable feature first."
        )

    if max_unique_values < 2:
        raise ValueError("max_unique_values must be at least 2.")

    if not 0 < max_unique_ratio <= 1:
        raise ValueError("max_unique_ratio must be between 0 and 1.")

    clean_target = target.dropna()

    if clean_target.empty:
        raise ValueError("Target column contains only missing values.")

    unique_count = clean_target.nunique()

    if unique_count < 2:
        raise ValueError("Target column must contain at least two unique values.")

    if is_bool_dtype(clean_target.dtype) or not is_numeric_dtype(clean_target.dtype):
        return "classification"

    if unique_count == 2:
        return "classification"

    unique_ratio = unique_count / len(clean_target)

    if unique_count <= max_unique_values and unique_ratio <= max_unique_ratio:
        return "classification"

    return "regression"
