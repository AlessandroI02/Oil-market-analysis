from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from src.assumptions_registry import MissingDataLogger
from src.fetch_diagnostics import FetchDiagnostics
from src.market_data_providers import fetch_equity_series
from src.source_logger import SourceLogger
from src.utils.yfinance_utils import first_valid

logger = logging.getLogger(__name__)


def _download_equity_weekly(
    ticker: str,
    start_date: date,
    end_date: date,
    frequency: str,
    fetch_diagnostics: FetchDiagnostics | None = None,
) -> tuple[pd.DataFrame, str, str, bool]:
    daily, provider, source_url, fallback_used = fetch_equity_series(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        diagnostics=fetch_diagnostics,
    )
    if daily.empty:
        return pd.DataFrame(columns=["date", "share_price"]), provider, source_url, fallback_used

    weekly = (
        daily.rename(columns={"price": "share_price"})
        .assign(date=lambda df: pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None))
        .dropna(subset=["date"])
        .set_index("date")
        .resample(frequency)
        .last()
        .reset_index()
    )
    weekly["share_price"] = pd.to_numeric(weekly["share_price"], errors="coerce")
    return weekly, provider, source_url, fallback_used


def build_equity_tracker(
    included_df: pd.DataFrame,
    crude_tracker_df: pd.DataFrame,
    fuel_tracker_df: pd.DataFrame,
    start_date: date,
    end_date: date,
    frequency: str,
    source_logger: SourceLogger,
    missing_logger: MissingDataLogger,
    fetch_diagnostics: FetchDiagnostics | None = None,
) -> pd.DataFrame:
    all_rows: list[pd.DataFrame] = []

    brent_idx = crude_tracker_df[["date", "brent_price"]].copy()
    brent_base = first_valid(brent_idx["brent_price"])
    brent_idx["brent_indexed_100"] = (
        (brent_idx["brent_price"] / brent_base) * 100 if brent_base is not None else pd.NA
    )

    for _, row in included_df.iterrows():
        company = row["company_name"]
        ticker = row["ticker"]

        weekly, provider, source_url, fallback_used = _download_equity_weekly(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            fetch_diagnostics=fetch_diagnostics,
        )
        if weekly.empty:
            missing_logger.add(
                company=company,
                field_name="weekly_share_price",
                reason=f"No equity data from configured providers for ticker {ticker}",
                attempted_sources=[
                    f"https://finance.yahoo.com/quote/{ticker}",
                    "https://stooq.com/",
                ],
                severity="high",
            )
            continue

        weekly = weekly.sort_values("date")
        weekly["company_name"] = company
        weekly["ticker"] = ticker
        weekly["equity_wow_pct"] = weekly["share_price"].pct_change() * 100
        equity_base = first_valid(weekly["share_price"])
        weekly["equity_cumulative_pct"] = (
            (weekly["share_price"] / equity_base - 1) * 100 if equity_base is not None else pd.NA
        )
        weekly["equity_indexed_100"] = (
            (weekly["share_price"] / equity_base) * 100 if equity_base is not None else pd.NA
        )

        merged = weekly.merge(brent_idx[["date", "brent_indexed_100"]], on="date", how="left")

        company_fuel = fuel_tracker_df[fuel_tracker_df["ticker"] == ticker][
            ["date", "blended_combined_fuels_price"]
        ].copy()
        if not company_fuel.empty:
            fuel_base = first_valid(company_fuel["blended_combined_fuels_price"])
            company_fuel["fuel_indexed_100"] = (
                (company_fuel["blended_combined_fuels_price"] / fuel_base) * 100 if fuel_base is not None else pd.NA
            )
            merged = merged.merge(company_fuel[["date", "fuel_indexed_100"]], on="date", how="left")
        else:
            merged["fuel_indexed_100"] = pd.NA

        merged["relative_vs_brent"] = merged["equity_indexed_100"] - merged["brent_indexed_100"]
        merged["relative_vs_fuels"] = merged["equity_indexed_100"] - merged["fuel_indexed_100"]

        all_rows.append(merged)

        source_logger.add(
            company=company,
            field="equity_weekly_prices",
            source_url=source_url or f"https://finance.yahoo.com/quote/{ticker}",
            source_tier="Tier 3",
            evidence_flag="exact",
            comments=f"Provider={provider}; fallback_used={fallback_used}",
        )

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "date",
                "company_name",
                "ticker",
                "share_price",
                "equity_wow_pct",
                "equity_cumulative_pct",
                "equity_indexed_100",
                "relative_vs_brent",
                "relative_vs_fuels",
            ]
        )

    out = pd.concat(all_rows, ignore_index=True)
    logger.info("Built equity tracker rows: %s", len(out))
    return out
