from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from src.assumptions_registry import MissingDataLogger
from src.fetch_diagnostics import FetchDiagnostics
from src.market_data_providers import fetch_crude_series
from src.source_logger import SourceLogger
from src.utils.yfinance_utils import first_valid

logger = logging.getLogger(__name__)


def _resample_weekly(price_df: pd.DataFrame, frequency: str, column_name: str) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame(columns=["date", column_name])

    out = price_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.rename(columns={"price": column_name})

    weekly = (
        out.set_index("date")[[column_name]]
        .resample(frequency)
        .last()
        .reset_index()
    )
    weekly[column_name] = pd.to_numeric(weekly[column_name], errors="coerce")
    return weekly


def build_crude_tracker(
    start_date: date,
    end_date: date,
    frequency: str,
    source_logger: SourceLogger,
    missing_logger: MissingDataLogger,
    fetch_diagnostics: FetchDiagnostics | None = None,
) -> pd.DataFrame:
    brent_raw, brent_provider, brent_source, brent_fallback = fetch_crude_series(
        benchmark="brent",
        start_date=start_date,
        end_date=end_date,
        diagnostics=fetch_diagnostics,
    )
    wti_raw, wti_provider, wti_source, wti_fallback = fetch_crude_series(
        benchmark="wti",
        start_date=start_date,
        end_date=end_date,
        diagnostics=fetch_diagnostics,
    )

    brent = _resample_weekly(brent_raw, frequency, "brent_price")
    wti = _resample_weekly(wti_raw, frequency, "wti_price")

    if brent.empty or pd.to_numeric(brent.get("brent_price"), errors="coerce").notna().sum() == 0:
        missing_logger.add(
            company="GLOBAL",
            field_name="Brent weekly prices",
            reason="All configured providers failed for Brent",
            attempted_sources=[
                "https://finance.yahoo.com/quote/BZ%3DF",
                "https://stooq.com/q/d/l/?s=bz.f&i=d",
                "https://fred.stlouisfed.org/series/DCOILBRENTEU",
            ],
            severity="high",
        )
    else:
        source_logger.add(
            company="GLOBAL",
            field="Brent weekly prices",
            source_url=brent_source or "https://finance.yahoo.com/quote/BZ%3DF",
            source_tier="Tier 2" if brent_provider == "fred" else "Tier 3",
            evidence_flag="exact" if brent_provider in {"yfinance", "stooq", "fred"} else "estimated",
            comments=f"Provider={brent_provider}; fallback_used={brent_fallback}",
        )

    if wti.empty or pd.to_numeric(wti.get("wti_price"), errors="coerce").notna().sum() == 0:
        missing_logger.add(
            company="GLOBAL",
            field_name="WTI weekly prices",
            reason="All configured providers failed for WTI",
            attempted_sources=[
                "https://finance.yahoo.com/quote/CL%3DF",
                "https://stooq.com/q/d/l/?s=cl.f&i=d",
                "https://fred.stlouisfed.org/series/DCOILWTICO",
            ],
            severity="high",
        )
    else:
        source_logger.add(
            company="GLOBAL",
            field="WTI weekly prices",
            source_url=wti_source or "https://finance.yahoo.com/quote/CL%3DF",
            source_tier="Tier 2" if wti_provider == "fred" else "Tier 3",
            evidence_flag="exact" if wti_provider in {"yfinance", "stooq", "fred"} else "estimated",
            comments=f"Provider={wti_provider}; fallback_used={wti_fallback}",
        )

    weekly_dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
    tracker = pd.DataFrame({"date": weekly_dates})
    tracker["date"] = pd.to_datetime(tracker["date"], errors="coerce").dt.tz_localize(None)

    tracker = tracker.merge(brent, how="left", on="date")
    tracker = tracker.merge(wti, how="left", on="date")

    tracker = tracker.sort_values("date")
    tracker["brent_price"] = pd.to_numeric(tracker["brent_price"], errors="coerce").ffill()
    tracker["wti_price"] = pd.to_numeric(tracker["wti_price"], errors="coerce").ffill()

    brent_base = first_valid(tracker["brent_price"])
    wti_base = first_valid(tracker["wti_price"])

    tracker["brent_wow_pct"] = tracker["brent_price"].pct_change() * 100
    tracker["wti_wow_pct"] = tracker["wti_price"].pct_change() * 100

    tracker["brent_cumulative_pct"] = (
        (tracker["brent_price"] / brent_base - 1) * 100 if brent_base is not None else pd.NA
    )
    tracker["wti_cumulative_pct"] = (
        (tracker["wti_price"] / wti_base - 1) * 100 if wti_base is not None else pd.NA
    )

    logger.info(
        "Built crude tracker rows=%s | brent_non_null=%s wti_non_null=%s",
        len(tracker),
        int(pd.to_numeric(tracker["brent_price"], errors="coerce").notna().sum()),
        int(pd.to_numeric(tracker["wti_price"], errors="coerce").notna().sum()),
    )
    return tracker
