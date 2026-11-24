import pandas as pd
import pytest

from src.data.loader import load_dataset, validate_dataset


def test_load_csv_success(tmp_path):
    file_path = tmp_path / "sample.csv"

    pd.DataFrame(
        {
            "Age": [25, 30, 35],
            "Salary": [30000, 40000, 50000],
        }
    ).to_csv(file_path, index=False)

    df = load_dataset(file_path)

    assert df.shape == (3, 2)
    assert list(df.columns) == ["Age", "Salary"]


def test_load_excel_success(tmp_path):
    file_path = tmp_path / "sample.xlsx"

    pd.DataFrame(
        {
            "Age": [25, 30],
            "Salary": [30000, 40000],
        }
    ).to_excel(file_path, index=False)

    df = load_dataset(file_path)

    assert df.shape == (2, 2)


def test_unsupported_file_format():
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_dataset("sample.json")


def test_empty_dataset_validation():
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        validate_dataset(df)


def test_duplicate_columns_validation():
    df = pd.DataFrame(
        [[1, 2]],
        columns=["Age", "Age"],
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_dataset(df)
