from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


ConfidenceLevel = Literal["High", "Medium", "Low"]
EstimateType = Literal[
    "analyst_estimate",
    "proxy_estimate",
    "inferred_exact",
    "disclosed_exact",
]
DataFlag = Literal["exact", "estimated", "proxy", "missing"]
BucketType = Literal["primary", "secondary", "rejected"]


class CompanyUniverseRecord(BaseModel):
    """Represents a reviewed company in the universe screening stage."""

    company_name: str
    ticker: str
    exchange: str
    isin: Optional[str] = None
    headquarters_country: Optional[str] = None
    listing_country: Optional[str] = None
    bucket_candidate: Optional[str] = None
    bucket_classification: BucketType
    reason_included: Optional[str] = None
    reason_excluded: Optional[str] = None
    upstream_exists: bool
    international_sales_exists: bool
    retail_stations_exists: bool
    source_links: list[str] = Field(default_factory=list)
    short_business_description: Optional[str] = None
    primary_operating_regions: list[str] = Field(default_factory=list)
    archetype: Optional[str] = None
    notes: Optional[str] = None
    confidence: ConfidenceLevel = "Medium"
    inclusion_confidence: Optional[ConfidenceLevel] = None


class CompanyProfile(BaseModel):
    """Operational profile used in estimation modules."""

    company_name: str
    ticker: str
    bucket_classification: Literal["primary", "secondary"]
    production_region_weights: dict[str, float] = Field(default_factory=dict)
    refinery_region_weights: dict[str, float] = Field(default_factory=dict)
    retail_country_weights: dict[str, float] = Field(default_factory=dict)
    destination_market_weights: dict[str, float] = Field(default_factory=dict)
    international_sales_ratio: Optional[float] = None
    upstream_mix_share: Optional[float] = None
    downstream_mix_share: Optional[float] = None
    archetype: Optional[str] = None
    profile_confidence: ConfidenceLevel = "Medium"
    source_links: list[str] = Field(default_factory=list)
    profile_notes: Optional[str] = None

    @field_validator(
        "production_region_weights",
        "refinery_region_weights",
        "retail_country_weights",
        "destination_market_weights",
    )
    @classmethod
    def _validate_weight_map(cls, value: dict[str, float]) -> dict[str, float]:
        for key, weight in value.items():
            if weight < 0:
                raise ValueError(f"Weight for {key} cannot be negative.")
        return value


class ExposureEstimate(BaseModel):
    company_name: str
    ticker: str
    crude_exposure_pct: float
    refined_exposure_pct: float
    combined_exposure_pct: float
    physical_exposure_pct: Optional[float] = None
    economic_exposure_pct: Optional[float] = None
    earnings_exposure_pct: Optional[float] = None
    exposure_low_pct: Optional[float] = None
    exposure_high_pct: Optional[float] = None
    confidence: ConfidenceLevel
    methodology_note: str
    exact_vs_estimated: DataFlag
    ranking: Optional[int] = None


class RouteRiskRecord(BaseModel):
    company_name: str
    ticker: str
    likely_route_summary: str
    hormuz_used: bool
    hormuz_share_pct: Optional[float] = None
    bab_el_mandeb_share_pct: Optional[float] = None
    suez_share_pct: Optional[float] = None
    non_chokepoint_share_pct: Optional[float] = None
    pipeline_bypass_optionality: Optional[str] = None
    rerouting_flexibility: Optional[str] = None
    other_chokepoints: list[str] = Field(default_factory=list)
    disruption_notes: str
    qualitative_route_risk: str


class SourceLogEntry(BaseModel):
    company: str
    field: str
    source_url: str
    source_title: Optional[str] = None
    source_tier: Optional[str] = None
    provider_used: Optional[str] = None
    access_date: datetime
    evidence_flag: DataFlag
    comments: Optional[str] = None


class AssumptionRecord(BaseModel):
    field_name: str
    company: str
    estimate_value: str
    estimate_type: EstimateType
    reasoning: str
    source_urls: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel
    model_version: Optional[str] = None
    timestamp: datetime


class MissingDataRecord(BaseModel):
    company: str
    field_name: str
    reason: str
    attempted_sources: list[str] = Field(default_factory=list)
    severity: Optional[str] = None
    timestamp: datetime


class OperatingMixRecord(BaseModel):
    company_name: str
    ticker: str
    currency_raw: Optional[str] = None
    fx_to_usd: Optional[float] = None
    upstream_volume: Optional[float] = None
    downstream_volume: Optional[float] = None
    upstream_revenue: Optional[float] = None
    downstream_revenue: Optional[float] = None
    total_revenue: Optional[float] = None
    upstream_revenue_usd: Optional[float] = None
    downstream_revenue_usd: Optional[float] = None
    total_revenue_usd: Optional[float] = None
    upstream_share_pct: Optional[float] = None
    downstream_share_pct: Optional[float] = None
    data_flag: DataFlag
    notes: str


class EarningsRecord(BaseModel):
    company_name: str
    ticker: str
    next_earnings_date: Optional[datetime] = None
    following_earnings_date: Optional[datetime] = None
    source_url: Optional[str] = None
    source_date: Optional[datetime] = None
    exact_vs_estimated: DataFlag = "missing"
    notes: Optional[str] = None


class PipelineArtifacts(BaseModel):
    timestamp_label: str
    universe_review_csv: Optional[str] = None
    universe_review_xlsx: Optional[str] = None
    included_companies_csv: Optional[str] = None
    rejected_companies_csv: Optional[str] = None
    hormuz_exposure_csv: Optional[str] = None
    route_risks_csv: Optional[str] = None
    fuel_weights_csv: Optional[str] = None
    crude_tracker_csv: Optional[str] = None
    fuel_tracker_csv: Optional[str] = None
    equity_tracker_csv: Optional[str] = None
    operating_mix_csv: Optional[str] = None
    earnings_csv: Optional[str] = None
    source_log_csv: Optional[str] = None
    assumptions_registry_csv: Optional[str] = None
    missing_data_log_csv: Optional[str] = None
    excel_path: Optional[str] = None
    word_path: Optional[str] = None

