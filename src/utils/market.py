from __future__ import annotations

import pandas as pd


def compute_weekly_metrics(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    group_cols = group_cols or []

    if group_cols:
        out["wow_pct"] = out.groupby(group_cols)[value_col].pct_change() * 100
        out["cumulative_pct"] = (
            out.groupby(group_cols)[value_col].transform(lambda s: (s / s.iloc[0] - 1) * 100)
        )
        out["indexed_100"] = out.groupby(group_cols)[value_col].transform(lambda s: (s / s.iloc[0]) * 100)
    else:
        out["wow_pct"] = out[value_col].pct_change() * 100
        out["cumulative_pct"] = (out[value_col] / out[value_col].iloc[0] - 1) * 100
        out["indexed_100"] = (out[value_col] / out[value_col].iloc[0]) * 100

    return out


def align_to_weekly_close(df: pd.DataFrame, date_col: str, value_cols: list[str], frequency: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col)
    out = out.set_index(date_col)[value_cols].resample(frequency).last().ffill().reset_index()
    return out
