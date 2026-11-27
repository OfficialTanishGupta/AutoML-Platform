import pandas as pd
import pytest

from src.models.task_detector import detect_task_type


def test_detect_categorical_target():
    target = pd.Series(["Yes", "No", "Yes", "No"])
    assert detect_task_type(target) == "classification"


def test_detect_numeric_binary_target():
    target = pd.Series([0, 1, 0, 1, 0])
    assert detect_task_type(target) == "classification"


def test_detect_boolean_target():
    target = pd.Series([True, False, True, False])
    assert detect_task_type(target) == "classification"


def test_detect_numeric_low_cardinality_target():
    target = pd.Series([1, 2, 1, 2, 3] * 20)
    assert detect_task_type(target) == "classification"


def test_detect_continuous_numeric_target():
    target = pd.Series(range(100))
    assert detect_task_type(target) == "regression"


def test_empty_target_raises_error():
    with pytest.raises(ValueError, match="empty"):
        detect_task_type(pd.Series(dtype=float))


def test_target_with_only_missing_values_raises_error():
    target = pd.Series([None, None, None])

    with pytest.raises(ValueError, match="only missing"):
        detect_task_type(target)


def test_target_with_one_unique_value_raises_error():
    target = pd.Series([10, 10, 10])

    with pytest.raises(ValueError, match="two unique values"):
        detect_task_type(target)


def test_datetime_target_raises_error():
    target = pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"]))

    with pytest.raises(ValueError, match="Datetime targets"):
        detect_task_type(target)


def test_non_series_target_raises_error():
    with pytest.raises(TypeError, match="pandas Series"):
        detect_task_type([1, 2, 3])  # type: ignore[arg-type]detect_task_type([1, 2, 3])
