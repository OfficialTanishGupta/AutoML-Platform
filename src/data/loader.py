from pathlib import Path
from typing import BinaryIO

import pandas as pd

SUPPORTED_FORMATS = {".csv", ".xlsx"}


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate a loaded dataset."""

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    if df.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate column names.")

    if any(str(column).strip() == "" for column in df.columns):
        raise ValueError("The dataset contains empty column names.")


def load_dataset(
    source: str | Path | BinaryIO,
    filename: str | None = None,
) -> pd.DataFrame:
    """Load and validate a CSV or Excel dataset."""

    if filename is None:
        if isinstance(source, (str, Path)):
            filename = Path(source).name
        else:
            source_name = getattr(source, "name", None)

            if source_name is None:
                raise ValueError("A filename must be provided.")

            filename = str(source_name)

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            "Please upload a CSV or XLSX file."
        )

    try:
        if isinstance(source, (str, Path)):
            file_source = source
        else:
            source.seek(0)
            file_source = source

        if extension == ".csv":
            df = pd.read_csv(file_source)
        else:
            df = pd.read_excel(file_source, engine="openpyxl")

    except Exception as error:
        raise ValueError(f"Unable to read the uploaded dataset: {error}") from error

    validate_dataset(df)

    return df
