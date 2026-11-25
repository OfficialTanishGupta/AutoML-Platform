import pandas as pd
import pytest

from src.data.profiler import profile_dataset


@pytest.fixture
def sample_dataset():
    return pd.DataFrame(
        {
            "Age": [25, 30, 35, 40],
            "Salary": [30000, 40000, 50000, 60000],
            "Department": ["IT", "HR", "IT", "Finance"],
        }
    )


def test_profile_dataset_shape(sample_dataset):
    profile = profile_dataset(sample_dataset)

    assert profile["total_rows"] == 4
    assert profile["total_columns"] == 3


def test_detect_column_types(sample_dataset):
    profile = profile_dataset(sample_dataset)

    assert profile["numeric_columns"] == ["Age", "Salary"]
    assert profile["categorical_columns"] == ["Department"]


def test_detect_missing_values():
    df = pd.DataFrame(
        {
            "Age": [25, None, 35],
            "Department": ["IT", "HR", None],
        }
    )

    profile = profile_dataset(df)

    assert profile["total_missing_values"] == 2
    assert profile["missing_values"]["Age"] == 1
    assert profile["missing_values"]["Department"] == 1


def test_detect_duplicate_rows():
    df = pd.DataFrame(
        {
            "Age": [25, 30, 25],
            "Salary": [30000, 40000, 30000],
        }
    )

    profile = profile_dataset(df)

    assert profile["duplicate_rows"] == 1


def test_numeric_summary(sample_dataset):
    profile = profile_dataset(sample_dataset)

    assert "Age" in profile["numeric_summary"]
    assert profile["numeric_summary"]["Age"]["mean"] == 32.5


def test_categorical_summary(sample_dataset):
    profile = profile_dataset(sample_dataset)

    department = profile["categorical_summary"]["Department"]

    assert department["unique_values"] == 3
    assert department["most_frequent_value"] == "IT"
    assert department["most_frequent_count"] == 2


def test_empty_dataset():
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        profile_dataset(df)
