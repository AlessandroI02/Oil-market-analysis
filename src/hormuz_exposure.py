from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.assumptions_registry import AssumptionsRegistry
from src.models import ExposureEstimate

logger = logging.getLogger(__name__)


CRUDE_REGION_HORMUZ_FACTOR: dict[str, float] = {
    "Middle East": 0.95,
    "North America": 0.05,
    "Europe": 0.12,
    "Latin America": 0.04,
    "Africa": 0.15,
    "Asia Pacific": 0.18,
    "China": 0.02,
    "Central Asia": 0.25,
    "US": 0.05,
    "Other": 0.10,
}

REFINED_REGION_HORMUZ_FACTOR: dict[str, float] = {
    "Middle East": 0.85,
    "North America": 0.12,
    "Europe": 0.25,
    "Latin America": 0.09,
    "Africa": 0.2,
    "Asia Pacific": 0.35,
    "China": 0.3,
    "US": 0.12,
    "Other": 0.15,
}

DESTINATION_HORMUZ_IMPORT_FACTOR: dict[str, float] = {
    "Asia": 0.36,
    "Europe": 0.22,
    "North America": 0.08,
    "Africa": 0.18,
    "Latin America": 0.10,
    "Other": 0.14,
}


def _weighted_factor(weight_map: dict[str, float], factor_map: dict[str, float]) -> float:
    value = 0.0
    for region, weight in weight_map.items():
        value += weight * factor_map.get(region, factor_map.get("Other", 0.1))
    return value


def _confidence_from_profile(profile_conf: str, missing_fields: int) -> str:
    if profile_conf == "High" and missing_fields == 0:
        return "High"
    if missing_fields >= 2:
        return "Low"
    if profile_conf == "Low":
        return "Low"
    return "Medium"


def _range_width(confidence: str) -> float:
    if confidence == "High":
        return 4.0
    if confidence == "Medium":
        return 8.0
    return 14.0


def estimate_hormuz_exposure(
    profiles_df: pd.DataFrame,
    assumptions_registry: AssumptionsRegistry,
) -> pd.DataFrame:
    estimates: list[ExposureEstimate] = []

    assumptions_registry.add(
        field_name="CRUDE_REGION_HORMUZ_FACTOR",
        company="GLOBAL",
        estimate_value=str(CRUDE_REGION_HORMUZ_FACTOR),
        estimate_type="proxy_estimate",
        reasoning="Region-level crude Hormuz reliance coefficients reflect typical export route dependency by producing basin.",
        source_urls=[
            "https://www.eia.gov/international/analysis/special-topics/World_Oil_Transit_Chokepoints",
            "https://www.iea.org/reports/oil-market-report",
        ],
        confidence="Medium",
        model_version="2.0",
    )
    assumptions_registry.add(
        field_name="REFINED_REGION_HORMUZ_FACTOR",
        company="GLOBAL",
        estimate_value=str(REFINED_REGION_HORMUZ_FACTOR),
        estimate_type="proxy_estimate",
        reasoning="Refined product exposure factors proxy dependence on Middle East-linked feedstock and route requirements.",
        source_urls=[
            "https://www.eia.gov/international/analysis/special-topics/World_Oil_Transit_Chokepoints",
            "https://www.iea.org/reports/oil-market-report",
        ],
        confidence="Medium",
        model_version="2.0",
    )
    assumptions_registry.add(
        field_name="DESTINATION_HORMUZ_IMPORT_FACTOR",
        company="GLOBAL",
        estimate_value=str(DESTINATION_HORMUZ_IMPORT_FACTOR),
        estimate_type="proxy_estimate",
        reasoning="Destination import dependence factors proxy downstream economic transmission of Gulf route disruptions.",
        source_urls=[
            "https://www.eia.gov/international/analysis/special-topics/World_Oil_Transit_Chokepoints",
            "https://www.iea.org/reports/oil-market-report",
        ],
        confidence="Medium",
        model_version="2.0",
    )

    for _, row in profiles_df.iterrows():
        prod_weights: dict[str, float] = row.get("production_region_weights") or {"Other": 1.0}
        ref_weights: dict[str, float] = row.get("refinery_region_weights") or {"Other": 1.0}
        dest_weights: dict[str, float] = row.get("destination_market_weights") or {"Other": 1.0}
        intl_ratio = float(row.get("international_sales_ratio") or 0.5)
        upstream_share = float(row.get("upstream_mix_share") or 0.5)
        downstream_share = float(row.get("downstream_mix_share") or 0.5)

        crude_dependency = _weighted_factor(prod_weights, CRUDE_REGION_HORMUZ_FACTOR)
        refined_dependency = _weighted_factor(ref_weights, REFINED_REGION_HORMUZ_FACTOR)
        destination_dependency = _weighted_factor(dest_weights, DESTINATION_HORMUZ_IMPORT_FACTOR)

        crude_exposure_pct = min(max(crude_dependency * intl_ratio * 100, 0.0), 100.0)
        refined_exposure_pct = min(max(refined_dependency * intl_ratio * 100, 0.0), 100.0)

        if (upstream_share + downstream_share) == 0:
            combined = (crude_exposure_pct + refined_exposure_pct) / 2
        else:
            combined = (
                crude_exposure_pct * upstream_share + refined_exposure_pct * downstream_share
            ) / (upstream_share + downstream_share)

        missing_fields = sum(
            1
            for field in ["international_sales_ratio", "upstream_mix_share", "downstream_mix_share"]
            if pd.isna(row.get(field))
        )
        confidence = _confidence_from_profile(str(row.get("profile_confidence", "Medium")), missing_fields)
        width = _range_width(confidence)

        physical_exposure = combined
        economic_exposure = min(max((combined * 0.7) + (destination_dependency * 100 * 0.3), 0.0), 100.0)
        earnings_exposure = min(
            max(
                economic_exposure * (0.55 + upstream_share * 0.5 - downstream_share * 0.15),
                0.0,
            ),
            100.0,
        )

        methodology_note = (
            "Exposure estimated from production/refinery geography, destination dependence, international sales ratio, and route dependence coefficients."
        )

        estimates.append(
            ExposureEstimate(
                company_name=row["company_name"],
                ticker=row["ticker"],
                crude_exposure_pct=round(crude_exposure_pct, 2),
                refined_exposure_pct=round(refined_exposure_pct, 2),
                combined_exposure_pct=round(combined, 2),
                physical_exposure_pct=round(physical_exposure, 2),
                economic_exposure_pct=round(economic_exposure, 2),
                earnings_exposure_pct=round(earnings_exposure, 2),
                exposure_low_pct=round(max(combined - width, 0.0), 2),
                exposure_high_pct=round(min(combined + width, 100.0), 2),
                confidence=confidence,
                methodology_note=methodology_note,
                exact_vs_estimated="estimated",
            )
        )

        assumptions_registry.add(
            field_name="combined_hormuz_exposure_pct",
            company=row["company_name"],
            estimate_value=f"{combined:.2f}",
            estimate_type="analyst_estimate",
            reasoning="Combined weighted exposure from crude and refined route dependencies.",
            source_urls=(row.get("source_links") or []),
            confidence=confidence,
            model_version="2.0",
        )

    out = pd.DataFrame([e.model_dump() for e in estimates])
    out = out.sort_values("combined_exposure_pct", ascending=True).reset_index(drop=True)
    out["ranking"] = out.index + 1

    logger.info("Hormuz exposure estimated for %s companies", len(out))
    return out


def build_route_exposure_build(profiles_df: pd.DataFrame, exposure_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if profiles_df.empty:
        return pd.DataFrame()

    lookup = exposure_df.set_index("ticker").to_dict(orient="index") if not exposure_df.empty else {}

    for _, row in profiles_df.iterrows():
        ticker = row["ticker"]
        company = row["company_name"]
        exp = lookup.get(ticker, {})

        for region, weight in (row.get("production_region_weights") or {}).items():
            rows.append(
                {
                    "company_name": company,
                    "ticker": ticker,
                    "build_component": "production_region_weight",
                    "component_name": region,
                    "value": weight,
                    "notes": "Normalized production origin share",
                }
            )
        for region, weight in (row.get("refinery_region_weights") or {}).items():
            rows.append(
                {
                    "company_name": company,
                    "ticker": ticker,
                    "build_component": "refinery_region_weight",
                    "component_name": region,
                    "value": weight,
                    "notes": "Normalized refining footprint share",
                }
            )
        rows.append(
            {
                "company_name": company,
                "ticker": ticker,
                "build_component": "combined_exposure_pct",
                "component_name": "central_estimate",
                "value": exp.get("combined_exposure_pct"),
                "notes": f"Range {exp.get('exposure_low_pct')} to {exp.get('exposure_high_pct')}",
            }
        )

    return pd.DataFrame(rows)
