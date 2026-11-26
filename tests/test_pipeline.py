import numpy as np
import pandas as pd
import pytest

from scipy.sparse import issparse

from src.preprocessing.pipeline import build_preprocessor


def to_dense_array(transformed) -> np.ndarray:
    """Convert preprocessing output into a NumPy array."""

    if issparse(transformed):
        return transformed.toarray()

    return np.asarray(transformed)


def test_build_preprocessor_with_mixed_columns():
    X = pd.DataFrame(
        {
            "Age": [25, 30, 35],
            "Salary": [30000, 40000, 50000],
            "Department": ["IT", "HR", "Finance"],
        }
    )

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape == (3, 5)


def test_numeric_preprocessing():
    X = pd.DataFrame(
        {
            "Age": [20, 30, 40],
            "Salary": [20000, 30000, 40000],
        }
    )

    preprocessor = build_preprocessor(X)
    transformed = to_dense_array(preprocessor.fit_transform(X))

    assert transformed.shape == (3, 2)
    assert np.allclose(np.mean(transformed, axis=0), 0)


def test_categorical_preprocessing():
    X = pd.DataFrame(
        {
            "Department": ["IT", "HR", "Finance"],
        }
    )

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape == (3, 3)


def test_missing_values_are_imputed():
    X = pd.DataFrame(
        {
            "Age": [20, None, 40],
            "Department": ["IT", "HR", None],
        }
    )

    preprocessor = build_preprocessor(X)

    transformed = to_dense_array(preprocessor.fit_transform(X))

    assert not np.isnan(transformed).any()


def test_unknown_categories_are_handled():
    X_train = pd.DataFrame(
        {
            "Department": ["IT", "HR", "Finance"],
        }
    )

    X_test = pd.DataFrame(
        {
            "Department": ["Marketing"],
        }
    )

    preprocessor = build_preprocessor(X_train)

    transformed = preprocessor.fit_transform(X_train)
    new_transformed = preprocessor.transform(X_test)

    assert transformed.shape == (3, 3)
    assert new_transformed.shape == (1, 3)


def test_empty_dataset_raises_error():
    X = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        build_preprocessor(X)


def test_datetime_columns_raise_error():
    X = pd.DataFrame(
        {
            "JoiningDate": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                ]
            ),
        }
    )

    with pytest.raises(ValueError, match="Datetime columns"):
        build_preprocessor(X)


def test_duplicate_columns_raise_error():
    X = pd.DataFrame(
        [[25, 30000]],
        columns=["Age", "Age"],
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_preprocessor(X)
