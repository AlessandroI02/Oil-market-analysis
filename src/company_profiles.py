from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.assumptions_registry import AssumptionsRegistry, MissingDataLogger
from src.models import CompanyProfile
from src.source_logger import SourceLogger
from src.utils.math_utils import normalize_weights

logger = logging.getLogger(__name__)


PROFILE_LIBRARY: dict[str, dict[str, Any]] = {
    "XOM": {
        "production_region_weights": {
            "North America": 0.42,
            "Middle East": 0.14,
            "Europe": 0.10,
            "Latin America": 0.14,
            "Asia Pacific": 0.12,
            "Africa": 0.08,
        },
        "refinery_region_weights": {
            "North America": 0.48,
            "Europe": 0.22,
            "Asia Pacific": 0.16,
            "Middle East": 0.14,
        },
        "retail_country_weights": {"US": 0.45, "UK": 0.2, "France": 0.1, "Italy": 0.1, "Singapore": 0.15},
        "destination_market_weights": {"Europe": 0.30, "Asia": 0.34, "North America": 0.24, "Other": 0.12},
        "international_sales_ratio": 0.68,
        "upstream_mix_share": 0.58,
        "downstream_mix_share": 0.42,
        "profile_confidence": "Medium",
    },
    "CVX": {
        "production_region_weights": {
            "North America": 0.5,
            "Middle East": 0.12,
            "Asia Pacific": 0.12,
            "Africa": 0.12,
            "Latin America": 0.14,
        },
        "refinery_region_weights": {
            "North America": 0.56,
            "Asia Pacific": 0.18,
            "Middle East": 0.14,
            "Europe": 0.12,
        },
        "retail_country_weights": {"US": 0.55, "South Korea": 0.12, "Singapore": 0.12, "Australia": 0.11, "South Africa": 0.1},
        "destination_market_weights": {"Asia": 0.37, "North America": 0.29, "Europe": 0.20, "Other": 0.14},
        "international_sales_ratio": 0.63,
        "upstream_mix_share": 0.60,
        "downstream_mix_share": 0.40,
        "profile_confidence": "Medium",
    },
    "SHEL": {
        "production_region_weights": {
            "North America": 0.24,
            "Middle East": 0.18,
            "Europe": 0.18,
            "Africa": 0.12,
            "Asia Pacific": 0.14,
            "Latin America": 0.14,
        },
        "refinery_region_weights": {"Europe": 0.34, "North America": 0.29, "Asia Pacific": 0.21, "Middle East": 0.16},
        "retail_country_weights": {"UK": 0.16, "Germany": 0.12, "US": 0.21, "China": 0.21, "Netherlands": 0.1, "Other": 0.2},
        "destination_market_weights": {"Europe": 0.31, "Asia": 0.35, "North America": 0.19, "Other": 0.15},
        "international_sales_ratio": 0.74,
        "upstream_mix_share": 0.52,
        "downstream_mix_share": 0.48,
        "profile_confidence": "Medium",
    },
    "BP": {
        "production_region_weights": {
            "North America": 0.32,
            "Middle East": 0.18,
            "Europe": 0.14,
            "Africa": 0.18,
            "Asia Pacific": 0.18,
        },
        "refinery_region_weights": {"Europe": 0.35, "US": 0.35, "Asia Pacific": 0.18, "Middle East": 0.12},
        "retail_country_weights": {"UK": 0.22, "Germany": 0.12, "US": 0.28, "Australia": 0.16, "Spain": 0.12, "Other": 0.1},
        "destination_market_weights": {"Europe": 0.28, "Asia": 0.34, "North America": 0.24, "Other": 0.14},
        "international_sales_ratio": 0.71,
        "upstream_mix_share": 0.55,
        "downstream_mix_share": 0.45,
        "profile_confidence": "Medium",
    },
    "TTE": {
        "production_region_weights": {
            "Middle East": 0.22,
            "Africa": 0.26,
            "Europe": 0.14,
            "North America": 0.16,
            "Asia Pacific": 0.12,
            "Latin America": 0.10,
        },
        "refinery_region_weights": {"Europe": 0.38, "Middle East": 0.17, "Africa": 0.14, "Asia Pacific": 0.17, "North America": 0.14},
        "retail_country_weights": {"France": 0.2, "Belgium": 0.08, "Germany": 0.1, "Morocco": 0.1, "Egypt": 0.1, "Other": 0.42},
        "destination_market_weights": {"Europe": 0.33, "Asia": 0.31, "Africa": 0.18, "Other": 0.18},
        "international_sales_ratio": 0.78,
        "upstream_mix_share": 0.54,
        "downstream_mix_share": 0.46,
        "profile_confidence": "Medium",
    },
    "E": {
        "production_region_weights": {
            "Africa": 0.30,
            "Middle East": 0.18,
            "Europe": 0.24,
            "North America": 0.14,
            "Asia Pacific": 0.14,
        },
        "refinery_region_weights": {"Europe": 0.47, "Africa": 0.2, "Middle East": 0.14, "Asia Pacific": 0.19},
        "retail_country_weights": {"Italy": 0.25, "Germany": 0.12, "Austria": 0.07, "France": 0.12, "Switzerland": 0.08, "Other": 0.36},
        "destination_market_weights": {"Europe": 0.46, "Asia": 0.24, "Africa": 0.16, "Other": 0.14},
        "international_sales_ratio": 0.73,
        "upstream_mix_share": 0.53,
        "downstream_mix_share": 0.47,
        "profile_confidence": "Medium",
    },
    "REP.MC": {
        "production_region_weights": {
            "North America": 0.28,
            "Latin America": 0.18,
            "Europe": 0.22,
            "Africa": 0.14,
            "Asia Pacific": 0.18,
        },
        "refinery_region_weights": {"Europe": 0.62, "Latin America": 0.2, "North America": 0.18},
        "retail_country_weights": {"Spain": 0.55, "Portugal": 0.18, "Peru": 0.08, "Mexico": 0.07, "Other": 0.12},
        "destination_market_weights": {"Europe": 0.52, "Latin America": 0.22, "Asia": 0.16, "Other": 0.1},
        "international_sales_ratio": 0.59,
        "upstream_mix_share": 0.50,
        "downstream_mix_share": 0.50,
        "profile_confidence": "Medium",
    },
    "OMV.VI": {
        "production_region_weights": {"Europe": 0.4, "Middle East": 0.26, "Africa": 0.14, "Asia Pacific": 0.2},
        "refinery_region_weights": {"Europe": 0.76, "Middle East": 0.14, "Other": 0.1},
        "retail_country_weights": {"Austria": 0.15, "Romania": 0.2, "Hungary": 0.17, "Bulgaria": 0.09, "Serbia": 0.1, "Other": 0.29},
        "destination_market_weights": {"Europe": 0.71, "Asia": 0.15, "Other": 0.14},
        "international_sales_ratio": 0.52,
        "upstream_mix_share": 0.46,
        "downstream_mix_share": 0.54,
        "profile_confidence": "Low",
    },
    "GALP.LS": {
        "production_region_weights": {"Latin America": 0.38, "Africa": 0.2, "Europe": 0.22, "Middle East": 0.2},
        "refinery_region_weights": {"Europe": 0.7, "Africa": 0.12, "Other": 0.18},
        "retail_country_weights": {"Portugal": 0.66, "Spain": 0.24, "Other": 0.1},
        "destination_market_weights": {"Europe": 0.68, "Asia": 0.2, "Other": 0.12},
        "international_sales_ratio": 0.55,
        "upstream_mix_share": 0.48,
        "downstream_mix_share": 0.52,
        "profile_confidence": "Low",
    },
    "SU": {
        "production_region_weights": {"North America": 0.88, "Other": 0.12},
        "refinery_region_weights": {"North America": 0.92, "Other": 0.08},
        "retail_country_weights": {"Canada": 0.9, "US": 0.1},
        "destination_market_weights": {"North America": 0.83, "Asia": 0.09, "Other": 0.08},
        "international_sales_ratio": 0.37,
        "upstream_mix_share": 0.57,
        "downstream_mix_share": 0.43,
        "profile_confidence": "Medium",
    },
    "PKN.WA": {
        "production_region_weights": {"Europe": 0.74, "North America": 0.14, "Middle East": 0.12},
        "refinery_region_weights": {"Europe": 0.86, "Other": 0.14},
        "retail_country_weights": {"Poland": 0.55, "Czech Republic": 0.12, "Germany": 0.11, "Lithuania": 0.1, "Other": 0.12},
        "destination_market_weights": {"Europe": 0.79, "Other": 0.21},
        "international_sales_ratio": 0.41,
        "upstream_mix_share": 0.32,
        "downstream_mix_share": 0.68,
        "profile_confidence": "Low",
    },
    "0857.HK": {
        "production_region_weights": {"China": 0.66, "Central Asia": 0.1, "Middle East": 0.08, "Other": 0.16},
        "refinery_region_weights": {"China": 0.84, "Other": 0.16},
        "retail_country_weights": {"China": 0.94, "Other": 0.06},
        "destination_market_weights": {"Asia": 0.86, "Other": 0.14},
        "international_sales_ratio": 0.26,
        "upstream_mix_share": 0.45,
        "downstream_mix_share": 0.55,
        "profile_confidence": "Low",
    },
    "0386.HK": {
        "production_region_weights": {"China": 0.58, "Middle East": 0.12, "Africa": 0.08, "Other": 0.22},
        "refinery_region_weights": {"China": 0.88, "Other": 0.12},
        "retail_country_weights": {"China": 0.96, "Other": 0.04},
        "destination_market_weights": {"Asia": 0.9, "Other": 0.1},
        "international_sales_ratio": 0.23,
        "upstream_mix_share": 0.35,
        "downstream_mix_share": 0.65,
        "profile_confidence": "Low",
    },
    "EQNR": {
        "production_region_weights": {"Europe": 0.66, "North America": 0.24, "Other": 0.1},
        "refinery_region_weights": {"Europe": 0.8, "Other": 0.2},
        "retail_country_weights": {"Norway": 0.5, "Other": 0.5},
        "destination_market_weights": {"Europe": 0.74, "Asia": 0.16, "Other": 0.1},
        "international_sales_ratio": 0.72,
        "upstream_mix_share": 0.82,
        "downstream_mix_share": 0.18,
        "profile_confidence": "Medium",
    },
    "CVE": {
        "production_region_weights": {"North America": 0.86, "Other": 0.14},
        "refinery_region_weights": {"North America": 0.9, "Other": 0.1},
        "retail_country_weights": {"Canada": 0.7, "US": 0.3},
        "destination_market_weights": {"North America": 0.79, "Asia": 0.11, "Other": 0.1},
        "international_sales_ratio": 0.35,
        "upstream_mix_share": 0.62,
        "downstream_mix_share": 0.38,
        "profile_confidence": "Low",
    },
    "PBR": {
        "production_region_weights": {"Latin America": 0.79, "Other": 0.21},
        "refinery_region_weights": {"Latin America": 0.78, "Other": 0.22},
        "retail_country_weights": {"Brazil": 0.9, "Other": 0.1},
        "destination_market_weights": {"Asia": 0.38, "Latin America": 0.37, "Europe": 0.12, "Other": 0.13},
        "international_sales_ratio": 0.44,
        "upstream_mix_share": 0.67,
        "downstream_mix_share": 0.33,
        "profile_confidence": "Low",
    },
    "2222.SR": {
        "production_region_weights": {"Middle East": 0.92, "Other": 0.08},
        "refinery_region_weights": {"Middle East": 0.65, "Asia Pacific": 0.21, "Other": 0.14},
        "retail_country_weights": {"Saudi Arabia": 0.6, "Other": 0.4},
        "destination_market_weights": {"Asia": 0.66, "Europe": 0.14, "Other": 0.2},
        "international_sales_ratio": 0.79,
        "upstream_mix_share": 0.71,
        "downstream_mix_share": 0.29,
        "profile_confidence": "Medium",
    },
    "0883.HK": {
        "production_region_weights": {"China": 0.62, "Africa": 0.12, "Latin America": 0.12, "Other": 0.14},
        "refinery_region_weights": {"China": 0.95, "Other": 0.05},
        "retail_country_weights": {"China": 1.0},
        "destination_market_weights": {"Asia": 0.88, "Other": 0.12},
        "international_sales_ratio": 0.33,
        "upstream_mix_share": 0.84,
        "downstream_mix_share": 0.16,
        "profile_confidence": "Low",
    },
}


def _default_profile(row: pd.Series) -> dict[str, Any]:
    hq = str(row.get("headquarters_country") or "Unknown")
    if hq in {"United States", "Canada"}:
        prod = {"North America": 0.8, "Other": 0.2}
        refine = {"North America": 0.85, "Other": 0.15}
        retail = {"US": 0.7, "Canada": 0.3}
    elif hq in {"United Kingdom", "France", "Italy", "Spain", "Austria", "Portugal", "Poland"}:
        prod = {"Europe": 0.55, "Middle East": 0.2, "Africa": 0.15, "Other": 0.1}
        refine = {"Europe": 0.7, "Other": 0.3}
        retail = {"Europe": 0.85, "Other": 0.15}
    elif hq in {"China"}:
        prod = {"China": 0.7, "Middle East": 0.1, "Other": 0.2}
        refine = {"China": 0.85, "Other": 0.15}
        retail = {"China": 0.95, "Other": 0.05}
    elif hq in {"Saudi Arabia"}:
        prod = {"Middle East": 0.9, "Other": 0.1}
        refine = {"Middle East": 0.6, "Asia Pacific": 0.25, "Other": 0.15}
        retail = {"Saudi Arabia": 0.7, "Other": 0.3}
    elif hq in {"Brazil"}:
        prod = {"Latin America": 0.8, "Other": 0.2}
        refine = {"Latin America": 0.75, "Other": 0.25}
        retail = {"Brazil": 0.9, "Other": 0.1}
    else:
        prod = {"Other": 1.0}
        refine = {"Other": 1.0}
        retail = {"Other": 1.0}

    return {
        "production_region_weights": prod,
        "refinery_region_weights": refine,
        "retail_country_weights": retail,
        "destination_market_weights": {"Other": 1.0},
        "international_sales_ratio": 0.5,
        "upstream_mix_share": 0.5,
        "downstream_mix_share": 0.5,
        "profile_confidence": "Low",
    }


def build_company_profiles(
    included_df: pd.DataFrame,
    assumptions_registry: AssumptionsRegistry,
    source_logger: SourceLogger,
    missing_logger: MissingDataLogger,
) -> pd.DataFrame:
    profiles: list[CompanyProfile] = []

    for _, row in included_df.iterrows():
        ticker = str(row["ticker"])
        profile_raw = PROFILE_LIBRARY.get(ticker, _default_profile(row))

        prod_weights = normalize_weights(profile_raw["production_region_weights"])
        ref_weights = normalize_weights(profile_raw["refinery_region_weights"])
        retail_weights = normalize_weights(profile_raw["retail_country_weights"])
        dest_weights = normalize_weights(profile_raw["destination_market_weights"])

        profile = CompanyProfile(
            company_name=row["company_name"],
            ticker=ticker,
            bucket_classification=row["bucket_classification"],
            production_region_weights=prod_weights,
            refinery_region_weights=ref_weights,
            retail_country_weights=retail_weights,
            destination_market_weights=dest_weights,
            international_sales_ratio=profile_raw.get("international_sales_ratio"),
            upstream_mix_share=profile_raw.get("upstream_mix_share"),
            downstream_mix_share=profile_raw.get("downstream_mix_share"),
            profile_confidence=profile_raw.get("profile_confidence", "Medium"),
            source_links=(str(row.get("source_links", "")).split(" | ") if row.get("source_links") else []),
            profile_notes="Regional profile built from public company disclosures and analyst mapping.",
        )
        profiles.append(profile)

        assumptions_registry.add(
            field_name="production_region_weights",
            company=profile.company_name,
            estimate_value=str(profile.production_region_weights),
            estimate_type="analyst_estimate",
            reasoning="Regional production split normalized from public segment disclosures and qualitative footprint references.",
            source_urls=profile.source_links,
            confidence=profile.profile_confidence,
            model_version="2.0",
        )
        assumptions_registry.add(
            field_name="refinery_region_weights",
            company=profile.company_name,
            estimate_value=str(profile.refinery_region_weights),
            estimate_type="analyst_estimate",
            reasoning="Refining geography approximated from disclosed refinery asset maps and operating regions.",
            source_urls=profile.source_links,
            confidence=profile.profile_confidence,
            model_version="2.0",
        )
        assumptions_registry.add(
            field_name="retail_country_weights",
            company=profile.company_name,
            estimate_value=str(profile.retail_country_weights),
            estimate_type="proxy_estimate",
            reasoning="Retail country relevance estimated by known station footprints and strategic market focus.",
            source_urls=profile.source_links,
            confidence=profile.profile_confidence,
            model_version="2.0",
        )

        for src in profile.source_links:
            source_logger.add(
                company=profile.company_name,
                field="company_profile",
                source_url=src,
                source_tier="Tier 1",
                evidence_flag="estimated",
                comments="Profile and geography weight support",
            )

        if profile.international_sales_ratio is None:
            missing_logger.add(
                company=profile.company_name,
                field_name="international_sales_ratio",
                reason="No clear international sales ratio disclosed; fallback default not available.",
                attempted_sources=profile.source_links,
            )

    out = pd.DataFrame([p.model_dump() for p in profiles])
    logger.info("Built %s company operational profiles", len(out))
    return out
