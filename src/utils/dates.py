from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta


def derive_date_window(
    start_date: date | None,
    end_date: date | None,
    lookback_months: int,
) -> tuple[date, date]:
    today = datetime.utcnow().date()
    resolved_end = end_date or today
    resolved_start = start_date or (resolved_end - relativedelta(months=lookback_months))
    if resolved_start >= resolved_end:
        raise ValueError("start_date must be earlier than end_date")
    return resolved_start, resolved_end


def weekly_index(start_date: date, end_date: date, frequency: str = "W-FRI") -> pd.DatetimeIndex:
    return pd.date_range(start=start_date, end=end_date, freq=frequency)


def nearest_week_label(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def next_estimated_earnings_date(base: datetime | None) -> datetime | None:
    if base is None:
        return None
    return base + timedelta(days=90)
