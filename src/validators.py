from __future__ import annotations

from typing import Iterable

import pandas as pd


class ValidationError(Exception):
    """Raised when a required dataset fails integrity checks."""


def require_columns(df: pd.DataFrame, columns: Iterable[str], df_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValidationError(f"{df_name} missing columns: {missing}")


def ensure_non_empty(df: pd.DataFrame, df_name: str) -> None:
    if df.empty:
        raise ValidationError(f"{df_name} is empty")


def coerce_percent_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
