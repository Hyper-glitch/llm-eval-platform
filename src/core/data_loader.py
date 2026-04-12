"""Evaluation data loading and filtering utilities."""

import json
import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd

logger = logging.getLogger(__name__)


def load_eval_df(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read eval DataFrame from CSV and deserialize JSON columns."""
    logger.info(f"Loading data from {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, nrows=nrows)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if nrows is not None and len(df) > nrows:
        df = df.head(nrows)

    for col in ("messages", "expected_tools"):
        if col in df.columns:
            df[col] = df[col].map(_from_json_str)

    return df


def _from_json_str(value: Any) -> list[Any] | dict[str, Any]:
    """Deserialize a JSON string; return an empty list for null or unparseable values."""
    if pd.isna(value):
        return []
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return cast(list[Any] | dict[str, Any], json.loads(value))
        except (json.JSONDecodeError, TypeError):
            pass

    return []
