import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from src.models.trainer import train_model


@pytest.fixture
def classification_data():
    X = pd.DataFrame(
        {
            "Age": [22, 25, 35, 45, 52, 60],
            "Department": ["IT", "HR", "IT", "Finance", "HR", "Finance"],
        }
    )

    y = pd.Series(["No", "No", "Yes", "Yes", "No", "Yes"])

    return X, y


@pytest.fixture
def regression_data():
    X = pd.DataFrame(
        {
            "Experience": [1, 2, 3, 4, 5, 6],
            "Department": ["IT", "HR", "IT", "Finance", "HR", "Finance"],
        }
    )

    y = pd.Series([20000, 25000, 35000, 40000, 50000, 60000])

    return X, y


def test_train_logistic_regression(classification_data):
    X, y = classification_data

    model = train_model(X, y, "classification", "logistic_regression")

    assert isinstance(model, Pipeline)
    assert len(model.predict(X)) == len(y)


def test_train_random_forest_classifier(classification_data):
    X, y = classification_data

    model = train_model(X, y, "classification", "random_forest")

    assert isinstance(model, Pipeline)
    assert len(model.predict(X)) == len(y)


def test_train_linear_regression(regression_data):
    X, y = regression_data

    model = train_model(X, y, "regression", "linear_regression")

    assert isinstance(model, Pipeline)
    assert len(model.predict(X)) == len(y)


def test_train_random_forest_regressor(regression_data):
    X, y = regression_data

    model = train_model(X, y, "regression", "random_forest")

    assert isinstance(model, Pipeline)
    assert len(model.predict(X)) == len(y)


def test_invalid_task_type(classification_data):
    X, y = classification_data

    with pytest.raises(ValueError, match="Unsupported task type"):
        train_model(X, y, "clustering")


def test_invalid_model_for_task(classification_data):
    X, y = classification_data

    with pytest.raises(ValueError, match="Unsupported model"):
        train_model(X, y, "classification", "linear_regression")


def test_mismatched_row_counts(classification_data):
    X, y = classification_data

    X = X.iloc[:-1]

    with pytest.raises(ValueError, match="equal row counts"):
        train_model(X, y, "classification")


def test_missing_target_values(classification_data):
    X, y = classification_data
    y.iloc[0] = None

    with pytest.raises(ValueError, match="missing values"):
        train_model(X, y, "classification")


def test_classification_requires_two_classes(classification_data):
    X, _ = classification_data
    y = pd.Series(["Yes"] * len(X))

    with pytest.raises(ValueError, match="at least two"):
        train_model(X, y, "classification")


def test_regression_requires_numeric_target(regression_data):
    X, _ = regression_data
    y = pd.Series(["Low", "Medium", "High", "Low", "Medium", "High"])

    with pytest.raises(ValueError, match="numeric values"):
        train_model(X, y, "regression")
