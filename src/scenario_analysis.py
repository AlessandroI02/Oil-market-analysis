from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from src.utils.yfinance_utils import first_valid

logger = logging.getLogger(__name__)


def _subset_with_defaults(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = pd.NA
    return out[columns]


def _sensitivity_coeff(upstream_pct: float, downstream_pct: float) -> float:
    up = upstream_pct / 100
    down = downstream_pct / 100
    return (up * 1.25) - (down * 0.35)


def _resilience_bucket(leverage_ratio: float | None, dividend_yield_pct: float | None) -> str:
    lev = leverage_ratio if leverage_ratio is not None else 2.0
    div = dividend_yield_pct if dividend_yield_pct is not None else 0.0
    score = (3.0 - min(lev, 3.0)) + (div / 8)
    if score >= 2.0:
        return "Strong"
    if score >= 1.2:
        return "Medium"
    return "Weak"


def build_scenario_analysis(
    exposure_df: pd.DataFrame,
    operating_mix_df: pd.DataFrame,
    valuation_df: pd.DataFrame,
    crude_tracker_df: pd.DataFrame,
    brent_levels: Iterable[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if exposure_df is None or exposure_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    base_brent = first_valid(crude_tracker_df["brent_price"]) if not crude_tracker_df.empty else None
    if base_brent is None:
        base_brent = 90.0

    base = _subset_with_defaults(
        exposure_df,
        ["company_name", "ticker", "combined_exposure_pct", "confidence"],
    )
    base = base.merge(
        _subset_with_defaults(
            operating_mix_df,
            ["company_name", "ticker", "upstream_share_pct", "downstream_share_pct"],
        ),
        on=["company_name", "ticker"],
        how="left",
    )
    base = base.merge(
        _subset_with_defaults(
            valuation_df,
            ["company_name", "ticker", "leverage_ratio", "dividend_yield_pct", "market_cap_usd"],
        ),
        on=["company_name", "ticker"],
        how="left",
    )

    scenario_rows: list[dict[str, object]] = []
    for _, row in base.iterrows():
        up = float(row.get("upstream_share_pct") or 50.0)
        down = float(row.get("downstream_share_pct") or 50.0)
        coeff = _sensitivity_coeff(up, down)

        for level in brent_levels:
            delta = (float(level) - float(base_brent)) / 10.0
            ebitda_sens = coeff * delta * 8.0
            fcf_sens = ebitda_sens * 0.85
            eps_sens = ebitda_sens * 0.75

            scenario_rows.append(
                {
                    "company_name": row["company_name"],
                    "ticker": row["ticker"],
                    "base_brent": base_brent,
                    "scenario_brent": float(level),
                    "ebitda_sensitivity_pct": ebitda_sens,
                    "fcf_sensitivity_pct": fcf_sens,
                    "eps_sensitivity_pct": eps_sens,
                    "downside_support_comment": "Integrated downstream partially cushions downside" if down >= 45 else "Higher upstream beta increases downside volatility",
                    "payout_resilience": _resilience_bucket(
                        None if pd.isna(row.get("leverage_ratio")) else float(row.get("leverage_ratio")),
                        None if pd.isna(row.get("dividend_yield_pct")) else float(row.get("dividend_yield_pct")),
                    ),
                    "confidence": row.get("confidence", "Medium"),
                }
            )

    scenario_df = pd.DataFrame(scenario_rows)

    event_path_rows: list[dict[str, object]] = []
    event_paths = [
        ("Hormuz reopens in 2 weeks", 0.35, "Short-lived risk premium; fade likely"),
        ("Hormuz disrupted for 2 months", 0.70, "Sustained freight dislocation and supply risk premium"),
        ("Hormuz disrupted through earnings cycle", 1.00, "Full-quarter earnings/cash flow repricing"),
    ]

    for _, row in base.iterrows():
        up = float(row.get("upstream_share_pct") or 50.0)
        down = float(row.get("downstream_share_pct") or 50.0)
        coeff = _sensitivity_coeff(up, down)
        exposure = float(row.get("combined_exposure_pct") or 0.0)

        for scenario_name, severity, note in event_paths:
            scenario_impact = (coeff * 12 * severity) - (exposure * 0.08 * severity)
            event_path_rows.append(
                {
                    "company_name": row["company_name"],
                    "ticker": row["ticker"],
                    "event_path": scenario_name,
                    "scenario_impact_score": scenario_impact,
                    "expected_path_commentary": note,
                    "confidence": row.get("confidence", "Medium"),
                }
            )

    event_path_df = pd.DataFrame(event_path_rows)

    logger.info("Built scenario analysis rows: %s, event-path rows: %s", len(scenario_df), len(event_path_df))
    return scenario_df, event_path_df
