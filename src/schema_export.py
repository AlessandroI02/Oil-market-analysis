from __future__ import annotations

from pathlib import Path
from typing import Any

from src.storage_paths import ensure_storage_layout, write_json


def _company_case_packet_schema() -> dict[str, Any]:
    required = [
        "as_of_date",
        "company_name",
        "ticker",
        "bucket_classification",
        "archetype",
        "inclusion_confidence",
        "data_confidence",
        "source_confidence",
        "route_confidence",
        "event_confidence",
        "regime_confidence",
        "packet_confidence",
        "downstream_readiness_flag",
        "regime_label",
        "regime_summary",
        "latest_brent_price",
        "latest_wti_price",
        "latest_fuel_proxy",
        "stock_price_latest",
        "stock_return_1w",
        "stock_return_1m",
        "oil_beta",
        "sector_beta",
        "market_beta",
        "rates_beta",
        "dxy_beta",
        "rolling_oil_beta_20d",
        "rolling_oil_beta_60d",
        "rolling_oil_beta_90d",
        "route_exposure_central",
        "route_exposure_low",
        "route_exposure_high",
        "hormuz_share_pct",
        "bab_el_mandeb_share_pct",
        "suez_share_pct",
        "non_chokepoint_share_pct",
        "route_risk_label",
        "rerouting_flexibility",
        "pipeline_bypass_optionality",
        "event_day_hit_rate_up_oil",
        "event_day_hit_rate_down_oil",
        "avg_abnormal_return_oil_up",
        "avg_abnormal_return_oil_down",
        "analogue_summary",
        "next_relevant_catalysts",
        "market_constraint_summary",
        "suggested_discount_rate_uplift_bps",
        "suggested_discount_rate_uplift_range_bps",
        "suggested_beta_adjustment",
        "suggested_beta_adjustment_range",
        "suggested_risk_premium_bucket",
        "suggested_scenario_probability_shift",
        "constraint_confidence",
        "confidence_summary",
        "key_notes",
        "source_summary",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CompanyCasePacket",
        "type": "object",
        "required": required,
        "properties": {
            "as_of_date": {"type": "string", "format": "date"},
            "company_name": {"type": "string"},
            "ticker": {"type": "string"},
            "bucket_classification": {"type": "string", "enum": ["primary", "secondary", "rejected"]},
            "archetype": {"type": "string"},
            "inclusion_confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "data_confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "source_confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "route_confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "event_confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "regime_confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "packet_confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "downstream_readiness_flag": {"type": "boolean"},
            "regime_label": {"type": "string"},
            "regime_summary": {"type": "string"},
            "next_relevant_catalysts": {"type": "array", "items": {"type": "string"}},
            "source_summary": {"type": "string"},
        },
        "additionalProperties": True,
    }


def _market_constraints_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "MarketConstraints",
        "type": "object",
        "required": [
            "as_of_date",
            "company_name",
            "ticker",
            "suggested_discount_rate_uplift_bps",
            "suggested_beta_adjustment",
            "suggested_risk_premium_bucket",
            "scenario_probability_shift",
            "geopolitical_stress_multiplier",
            "confidence_penalty_recommendation",
            "constraint_confidence",
            "methodology_label",
            "market_regime_impact_note",
        ],
        "properties": {
            "as_of_date": {"type": "string", "format": "date"},
            "company_name": {"type": "string"},
            "ticker": {"type": "string"},
            "suggested_discount_rate_uplift_bps": {"type": "number"},
            "suggested_discount_rate_uplift_range_bps": {"type": "string"},
            "suggested_beta_adjustment": {"type": "number"},
            "suggested_beta_adjustment_range": {"type": "string"},
            "suggested_risk_premium_bucket": {"type": "string"},
            "scenario_probability_shift": {"type": "string"},
            "geopolitical_stress_multiplier": {"type": "number"},
            "confidence_penalty_recommendation": {"type": "number"},
            "constraint_confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "methodology_label": {"type": "string"},
            "market_regime_impact_note": {"type": "string"},
        },
        "additionalProperties": True,
    }


def _regime_state_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "RegimeState",
        "type": "object",
        "required": [
            "as_of_date",
            "regime_label",
            "regime_confidence",
            "regime_notes",
            "regime_score_total",
        ],
        "properties": {
            "as_of_date": {"type": "string", "format": "date"},
            "regime_label": {"type": "string"},
            "regime_confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "regime_notes": {"type": "string"},
            "regime_score_total": {"type": "number"},
        },
        "additionalProperties": True,
    }


def _event_episode_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "EventEpisode",
        "type": "object",
        "required": [
            "episode_id",
            "start_date",
            "end_date",
            "episode_type",
            "summary",
            "involved_event_days",
            "cumulative_brent_move",
            "cumulative_wti_move",
            "mean_abs_move",
            "peak_move",
            "attribution_confidence",
            "top_articles",
            "episode_source_summary",
        ],
        "properties": {
            "episode_id": {"type": "string"},
            "start_date": {"type": "string", "format": "date"},
            "end_date": {"type": "string", "format": "date"},
            "episode_type": {"type": "string"},
            "summary": {"type": "string"},
            "involved_event_days": {"type": "string"},
            "cumulative_brent_move": {"type": "number"},
            "cumulative_wti_move": {"type": "number"},
            "mean_abs_move": {"type": "number"},
            "peak_move": {"type": "number"},
            "attribution_confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "top_articles": {"type": "string"},
            "episode_source_summary": {"type": "string"},
        },
        "additionalProperties": True,
    }


def export_handoff_schemas(root: Path | None = None) -> dict[str, Path]:
    paths = ensure_storage_layout(root)
    schema_dir = paths["schemas"]
    exported = {
        "company_case_packet_schema": write_json(_company_case_packet_schema(), schema_dir / "company_case_packet.schema.json"),
        "market_constraints_schema": write_json(_market_constraints_schema(), schema_dir / "market_constraints.schema.json"),
        "regime_state_schema": write_json(_regime_state_schema(), schema_dir / "regime_state.schema.json"),
        "event_episode_schema": write_json(_event_episode_schema(), schema_dir / "event_episode.schema.json"),
    }
    return exported
