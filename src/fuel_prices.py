from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

from src.assumptions_registry import AssumptionsRegistry, MissingDataLogger
from src.fetch_diagnostics import FetchDiagnostics
from src.market_data_providers import fetch_fuel_series
from src.source_logger import SourceLogger
from src.utils.yfinance_utils import first_valid

logger = logging.getLogger(__name__)


COUNTRY_FUEL_MULTIPLIER = {
    "US": 1.00,
    "Canada": 1.08,
    "UK": 1.45,
    "France": 1.50,
    "Germany": 1.48,
    "Italy": 1.52,
    "Spain": 1.35,
    "Portugal": 1.40,
    "Austria": 1.36,
    "Romania": 1.15,
    "Hungary": 1.22,
    "Poland": 1.18,
    "Czech Republic": 1.20,
    "Lithuania": 1.17,
    "Belgium": 1.47,
    "Netherlands": 1.49,
    "Switzerland": 1.42,
    "Morocco": 1.05,
    "Egypt": 0.95,
    "South Korea": 1.25,
    "Singapore": 1.18,
    "Australia": 1.20,
    "South Africa": 1.08,
    "Brazil": 1.12,
    "Mexico": 1.06,
    "Peru": 1.04,
    "China": 1.14,
    "Saudi Arabia": 0.78,
    "Norway": 1.58,
    "Europe": 1.42,
    "Other": 1.20,
}


def _to_weekly(series_df: pd.DataFrame, frequency: str, out_col: str) -> pd.DataFrame:
    if series_df is None or series_df.empty:
        return pd.DataFrame(columns=["date", out_col])

    out = series_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.rename(columns={"price": out_col})
    out[out_col] = pd.to_numeric(out[out_col], errors="coerce")

    return (
        out.set_index("date")[[out_col]]
        .resample(frequency)
        .last()
        .reset_index()
    )


def _proxy_fuel_benchmarks(
    start_date: date,
    end_date: date,
    frequency: str,
    source_logger: SourceLogger,
    missing_logger: MissingDataLogger,
    fetch_diagnostics: FetchDiagnostics | None,
) -> tuple[pd.DataFrame, str]:
    gasoline_raw, gasoline_provider, gasoline_source, _ = fetch_fuel_series(
        fuel_kind="gasoline",
        start_date=start_date,
        end_date=end_date,
        diagnostics=fetch_diagnostics,
    )
    diesel_raw, diesel_provider, diesel_source, _ = fetch_fuel_series(
        fuel_kind="diesel",
        start_date=start_date,
        end_date=end_date,
        diagnostics=fetch_diagnostics,
    )

    gasoline = _to_weekly(gasoline_raw, frequency, "rbob_per_gal")
    diesel = _to_weekly(diesel_raw, frequency, "heating_oil_per_gal")

    dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
    out = pd.DataFrame({"date": dates})
    out = out.merge(gasoline, how="left", on="date")
    out = out.merge(diesel, how="left", on="date")

    out["rbob_per_gal"] = pd.to_numeric(out["rbob_per_gal"], errors="coerce").ffill()
    out["heating_oil_per_gal"] = pd.to_numeric(out["heating_oil_per_gal"], errors="coerce").ffill()
    out["petrol_proxy_usd_per_bbl"] = out["rbob_per_gal"] * 42
    out["diesel_proxy_usd_per_bbl"] = out["heating_oil_per_gal"] * 42

    if out[["rbob_per_gal", "heating_oil_per_gal"]].isna().all().all():
        missing_logger.add(
            company="GLOBAL",
            field_name="fuel benchmark series",
            reason="All configured fuel benchmark providers failed",
            attempted_sources=[
                "https://finance.yahoo.com/quote/RB%3DF",
                "https://finance.yahoo.com/quote/HO%3DF",
                "https://stooq.com/q/d/l/?s=rb.f&i=d",
                "https://stooq.com/q/d/l/?s=ho.f&i=d",
                "https://fred.stlouisfed.org/series/GASREGW",
                "https://fred.stlouisfed.org/series/GASDESW",
            ],
            severity="high",
        )
        return out, "missing"

    if out["rbob_per_gal"].notna().any():
        source_logger.add(
            company="GLOBAL",
            field="gasoline_benchmark",
            source_url=gasoline_source or "https://finance.yahoo.com/quote/RB%3DF",
            source_tier="Tier 2" if gasoline_provider == "fred" else "Tier 3",
            evidence_flag="exact",
            comments=f"Provider={gasoline_provider}",
        )
    if out["heating_oil_per_gal"].notna().any():
        source_logger.add(
            company="GLOBAL",
            field="diesel_benchmark",
            source_url=diesel_source or "https://finance.yahoo.com/quote/HO%3DF",
            source_tier="Tier 2" if diesel_provider == "fred" else "Tier 3",
            evidence_flag="exact",
            comments=f"Provider={diesel_provider}",
        )

    if gasoline_provider == "fred" or diesel_provider == "fred":
        data_type = "observed_country_proxy_fred"
    elif gasoline_provider in {"yfinance", "stooq"} or diesel_provider in {"yfinance", "stooq"}:
        data_type = "observed_market_benchmark"
    else:
        data_type = "unknown"

    return out, data_type


def _country_multiplier(country: str) -> float:
    if country in COUNTRY_FUEL_MULTIPLIER:
        return COUNTRY_FUEL_MULTIPLIER[country]
    return COUNTRY_FUEL_MULTIPLIER.get("Other", 1.2)


def build_fuel_trackers(
    profiles_df: pd.DataFrame,
    crude_tracker_df: pd.DataFrame,
    start_date: date,
    end_date: date,
    frequency: str,
    source_logger: SourceLogger,
    assumptions_registry: AssumptionsRegistry,
    missing_logger: MissingDataLogger,
    fetch_diagnostics: FetchDiagnostics | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_result = _proxy_fuel_benchmarks(
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        source_logger=source_logger,
        missing_logger=missing_logger,
        fetch_diagnostics=fetch_diagnostics,
    )
    if isinstance(benchmark_result, tuple):
        benchmark, benchmark_data_type = benchmark_result
    else:
        benchmark = benchmark_result
        benchmark_data_type = "observed_market_benchmark"

    if benchmark[["petrol_proxy_usd_per_bbl", "diesel_proxy_usd_per_bbl"]].isna().all().all():
        if "brent_price" in crude_tracker_df.columns and crude_tracker_df["brent_price"].notna().any():
            benchmark = benchmark.merge(
                crude_tracker_df[["date", "brent_price"]],
                on="date",
                how="left",
            )
            benchmark["petrol_proxy_usd_per_bbl"] = benchmark["brent_price"] * 1.18
            benchmark["diesel_proxy_usd_per_bbl"] = benchmark["brent_price"] * 1.22
            benchmark_data_type = "modelled_brent_proxy"
            assumptions_registry.add(
                field_name="fuel_proxy_from_brent",
                company="GLOBAL",
                estimate_value="petrol=1.18*brent, diesel=1.22*brent",
                estimate_type="proxy_estimate",
                reasoning="Observed fuel benchmarks unavailable; used Brent-scaled proxy to preserve weekly relative signal.",
                source_urls=["https://www.eia.gov/petroleum/marketing/monthly/"],
                confidence="Low",
                model_version="2.1",
            )
        else:
            missing_logger.add(
                company="GLOBAL",
                field_name="fuel proxy benchmarks",
                reason="No observed fuel series and Brent fallback unavailable",
                attempted_sources=[
                    "https://finance.yahoo.com/quote/RB%3DF",
                    "https://finance.yahoo.com/quote/HO%3DF",
                    "https://fred.stlouisfed.org/series/GASREGW",
                    "https://fred.stlouisfed.org/series/GASDESW",
                ],
                severity="high",
            )

    assumptions_registry.add(
        field_name="COUNTRY_FUEL_MULTIPLIER",
        company="GLOBAL",
        estimate_value=str(COUNTRY_FUEL_MULTIPLIER),
        estimate_type="proxy_estimate",
        reasoning="Country multipliers proxy retail tax/marketing spread over benchmark fuel series where exact weekly multi-country retail data is unavailable.",
        source_urls=[
            "https://www.iea.org/reports/oil-2025",
            "https://www.eia.gov/petroleum/marketing/monthly/",
            "https://ec.europa.eu/energy/data-analysis/weekly-oil-bulletin_en",
        ],
        confidence="Low",
        model_version="2.1",
    )

    fuel_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for _, row in profiles_df.iterrows():
        company = row["company_name"]
        ticker = row["ticker"]
        weights: dict[str, float] = row["retail_country_weights"]
        source_links = row.get("source_links") or []

        if not weights:
            missing_logger.add(
                company=company,
                field_name="retail_country_weights",
                reason="No retail geography weights available for blended fuel pricing",
                attempted_sources=source_links,
            )
            continue

        for country, weight in weights.items():
            multiplier = _country_multiplier(country)
            weight_rows.append(
                {
                    "company": company,
                    "ticker": ticker,
                    "geography": country,
                    "weight": weight,
                    "fuel_multiplier": multiplier,
                    "source": "profile_weights_and_public_footprint",
                    "notes": "Weight inferred from retail footprint; multiplier estimates retail spread/tax level.",
                }
            )

        company_series = benchmark[["date", "petrol_proxy_usd_per_bbl", "diesel_proxy_usd_per_bbl"]].copy()

        def _weighted(row_series: pd.Series, fuel_col: str) -> float:
            acc = 0.0
            for country, weight in weights.items():
                base_val = float(row_series[fuel_col]) if pd.notna(row_series[fuel_col]) else 0.0
                acc += base_val * _country_multiplier(country) * weight
            return acc

        company_series["blended_petrol_price"] = company_series.apply(
            lambda r: _weighted(r, "petrol_proxy_usd_per_bbl"), axis=1
        )
        company_series["blended_diesel_price"] = company_series.apply(
            lambda r: _weighted(r, "diesel_proxy_usd_per_bbl"), axis=1
        )
        company_series["blended_combined_fuels_price"] = (
            company_series["blended_petrol_price"] * 0.5 + company_series["blended_diesel_price"] * 0.5
        )
        company_series["company_name"] = company
        company_series["ticker"] = ticker
        company_series["fuel_wow_pct"] = company_series["blended_combined_fuels_price"].pct_change() * 100
        base_fuel = first_valid(company_series["blended_combined_fuels_price"])
        company_series["fuel_cumulative_pct"] = (
            (company_series["blended_combined_fuels_price"] / base_fuel - 1) * 100
            if base_fuel is not None
            else pd.NA
        )
        company_series["petrol_data_type"] = benchmark_data_type
        company_series["diesel_data_type"] = benchmark_data_type
        company_series["blended_data_type"] = (
            "observed_plus_proxy_weights"
            if benchmark_data_type in {"observed_market_benchmark", "observed_country_proxy_fred"}
            else "modelled_proxy"
        )

        if "brent_price" in company_series.columns:
            company_series = company_series.drop(columns=["brent_price"])
        merged = company_series.merge(crude_tracker_df[["date", "brent_price"]], on="date", how="left")
        merged["fuels_to_brent_ratio"] = merged["blended_combined_fuels_price"] / merged["brent_price"]

        fuel_rows.extend(merged.to_dict(orient="records"))

        assumptions_registry.add(
            field_name="company_fuel_geography_weights",
            company=company,
            estimate_value=str(weights),
            estimate_type="analyst_estimate",
            reasoning="Retail geography weights estimated from disclosed station footprints and regional exposure.",
            source_urls=source_links,
            confidence=row.get("profile_confidence", "Low"),
            model_version="2.1",
        )

        for src in source_links:
            source_logger.add(
                company=company,
                field="fuel_price_weighting",
                source_url=src,
                source_tier="Tier 2",
                evidence_flag="estimated",
                comments="Retail geography weighting source",
            )

    fuel_tracker = pd.DataFrame(fuel_rows)
    fuel_weights = pd.DataFrame(weight_rows)

    if fuel_tracker.empty:
        fuel_tracker = pd.DataFrame(
            columns=[
                "date",
                "company_name",
                "ticker",
                "blended_petrol_price",
                "blended_diesel_price",
                "blended_combined_fuels_price",
                "fuel_wow_pct",
                "fuel_cumulative_pct",
                "brent_price",
                "fuels_to_brent_ratio",
                "petrol_data_type",
                "diesel_data_type",
                "blended_data_type",
            ]
        )

    logger.info("Built fuel tracker rows: %s", len(fuel_tracker))
    return fuel_tracker, fuel_weights
