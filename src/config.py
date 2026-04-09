from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "Hormuz Integrated Oil Research"
    timezone: str = "UTC"
    currency: str = "USD"


class RunDefaults(BaseModel):
    lookback_months: int = 3
    frequency: str = "W-FRI"
    max_companies: int = 50
    rebuild_cache: bool = False
    debug: bool = False


class PathsConfig(BaseModel):
    data_root: str = "data"
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    cache: str = "data/cache"
    logs: str = "data/logs"
    output_root: str = "outputs"
    output_excel: str = "outputs/excel"
    output_word: str = "outputs/word"
    output_charts: str = "outputs/charts"
    output_debug: str = "outputs/debug"


class UniverseConfig(BaseModel):
    min_market_cap_usd: int = 10_000_000_000
    include_secondary_bucket: bool = True
    default_exchange_fallback: str = "Unknown"


class DataSourceConfig(BaseModel):
    request_timeout_seconds: int = 30
    max_retries: int = 3
    retry_wait_seconds: int = 2
    user_agent: str = "hormuz-research-bot/1.0"
    yfinance_enabled: bool = True


class RiskConfig(BaseModel):
    chokepoints: list[str] = Field(default_factory=list)


class WeightConfig(BaseModel):
    upstream_weight: float = 0.6
    downstream_weight: float = 0.4


class V2RankingWeights(BaseModel):
    route_exposure: float = 0.20
    oil_sensitivity: float = 0.12
    downstream_pass_through: float = 0.08
    valuation: float = 0.18
    balance_sheet: float = 0.12
    catalyst: float = 0.10
    market_positioning: float = 0.10
    scenario_resilience: float = 0.10


class V2Config(BaseModel):
    model_version: str = "2.0"
    assumptions_version: str = "2.0"
    ranking_weights: V2RankingWeights = Field(default_factory=V2RankingWeights)
    confidence_penalty: dict[str, float] = Field(
        default_factory=lambda: {"High": 0.0, "Medium": 0.05, "Low": 0.12}
    )
    scenario_brent_levels: list[float] = Field(default_factory=lambda: [80.0, 90.0, 100.0, 120.0])


class QualityGateConfig(BaseModel):
    fail_on_missing_core_market_data: bool = False
    allow_degraded_run: bool = True
    minimum_non_null_ratio_for_sheet: float = 0.20
    minimum_non_null_ratio_for_chart: float = 0.20
    minimum_non_null_ratio_for_section: float = 0.25
    minimum_required_price_points: int = 4
    minimum_required_equity_names: int = 4
    minimum_required_valuation_coverage: float = 0.35
    minimum_required_factor_coverage: float = 0.35
    news_price_move_threshold_pct: float = 2.0
    news_lookback_days: int = 30
    max_articles_per_event_day: int = 3


class Settings(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    run: RunDefaults = Field(default_factory=RunDefaults)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    data_sources: DataSourceConfig = Field(default_factory=DataSourceConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    weights: WeightConfig = Field(default_factory=WeightConfig)
    v2: V2Config = Field(default_factory=V2Config)
    quality: QualityGateConfig = Field(default_factory=QualityGateConfig)


@dataclass
class RuntimeConfig:
    start_date: date
    end_date: date
    lookback_months: int
    frequency: str
    output_dir: Optional[Path]
    max_companies: Optional[int]
    rebuild_cache: bool
    skip_word: bool
    skip_excel: bool
    only_universe: bool
    only_review: bool
    debug: bool


@dataclass
class AppContext:
    root_dir: Path
    settings_path: Path
    settings: Settings
    runtime: RuntimeConfig


def load_settings(settings_path: Path) -> Settings:
    if not settings_path.exists():
        raise FileNotFoundError(f"Missing settings file: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Settings.model_validate(raw)


def resolve_path(root: Path, path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def ensure_directories(root: Path, settings: Settings, output_override: Optional[Path] = None) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for field_name, path_value in settings.paths.model_dump().items():
        if field_name == "output_root" and output_override is not None:
            resolved = output_override.resolve()
        elif field_name.startswith("output_") and output_override is not None:
            suffix = Path(path_value).name
            resolved = (output_override / suffix).resolve()
        else:
            resolved = resolve_path(root, path_value)
        resolved.mkdir(parents=True, exist_ok=True)
        paths[field_name] = resolved
    return paths


def timestamp_label(now: Optional[datetime] = None) -> str:
    now_ts = now or datetime.utcnow()
    return now_ts.strftime("%Y%m%d_%H%M")
