from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _df_non_null_ratio(df: pd.DataFrame, cols: list[str] | None = None) -> float:
    if df is None or df.empty:
        return 0.0
    cols = cols or list(df.columns)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return 0.0

    block = df[cols].copy()
    denom = len(block) * len(cols)
    if denom == 0:
        return 0.0
    return float(block.notna().sum().sum() / denom)


def _confidence_label(ratio: float) -> str:
    if ratio >= 0.75:
        return "High"
    if ratio >= 0.40:
        return "Medium"
    return "Low"


def build_sheet_health(
    datasets: dict[str, pd.DataFrame],
    min_non_null_ratio: float,
) -> pd.DataFrame:
    core_fields_map: dict[str, list[str]] = {
        "Crude_Tracker": ["brent_price", "wti_price"],
        "Fuel_Tracker": ["blended_petrol_price", "blended_diesel_price", "blended_combined_fuels_price"],
        "Equity_Tracker": ["share_price"],
        "Hormuz_Ranking": ["combined_exposure_pct"],
        "Route_Exposure_Build": ["value"],
        "Route_Risks": ["hormuz_share_pct", "qualitative_route_risk"],
        "Valuation": ["market_cap_usd", "valuation_score"],
        "Factor_Decomposition": ["beta_brent_ret"],
        "Historical_Analogues": ["brent_return_pct"],
        "Core_Ranking": ["final_score", "score_default_share"],
    }

    rows: list[dict[str, Any]] = []
    for sheet_name, df in datasets.items():
        frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        row_count = int(len(frame))
        total_ratio = _df_non_null_ratio(frame)
        core_cols = core_fields_map.get(sheet_name, [])
        core_ratio = _df_non_null_ratio(frame, core_cols) if core_cols else total_ratio

        if frame.empty:
            status = "UNAVAILABLE"
            reason = "DataFrame empty"
        elif core_ratio == 0:
            status = "UNAVAILABLE"
            reason = "Core fields are fully null"
        elif core_ratio < min_non_null_ratio:
            status = "DEGRADED"
            reason = "Core-field population below threshold"
        else:
            status = "POPULATED"
            reason = "Sheet materially populated"

        rows.append(
            {
                "sheet_name": sheet_name,
                "sheet_status": status,
                "populated": status == "POPULATED",
                "row_count": row_count,
                "non_null_ratio": round(total_ratio, 4),
                "core_non_null_ratio": round(core_ratio, 4),
                "core_fields": ", ".join(core_cols),
                "data_confidence": _confidence_label(core_ratio),
                "reason": reason,
            }
        )

    return pd.DataFrame(rows).sort_values(["sheet_status", "sheet_name"]).reset_index(drop=True)


def build_section_health(
    datasets: dict[str, pd.DataFrame],
    run_summary: dict[str, Any],
    min_non_null_ratio: float,
) -> pd.DataFrame:
    section_map = [
        ("route_exposure", "Hormuz_Ranking", ["combined_exposure_pct"]),
        ("route_risks", "Route_Risks", ["hormuz_share_pct", "qualitative_route_risk"]),
        ("market_prices", "Crude_Tracker", ["brent_price", "wti_price"]),
        ("fuel_prices", "Fuel_Tracker", ["blended_combined_fuels_price"]),
        ("equity_performance", "Equity_Tracker", ["share_price"]),
        ("valuation", "Valuation", ["market_cap_usd", "valuation_score"]),
        ("factor_decomposition", "Factor_Decomposition", ["beta_brent_ret"]),
        ("historical_analogues", "Historical_Analogues", ["brent_return_pct"]),
        ("catalyst_map", "Catalyst_Calendar", ["event_date"]),
        ("recommendations", "Recommendation_Framework", ["final_score"]),
    ]

    invalid_mode = str(run_summary.get("run_status", "VALID")).upper() == "INVALID"
    rows: list[dict[str, Any]] = []

    for section_name, sheet_name, cols in section_map:
        df = datasets.get(sheet_name, pd.DataFrame())
        ratio = _df_non_null_ratio(df, cols)
        row_count = int(len(df)) if isinstance(df, pd.DataFrame) else 0

        if row_count == 0:
            status = "UNAVAILABLE"
            reason = f"{sheet_name} is empty"
        elif ratio == 0:
            status = "INVALID"
            reason = f"{sheet_name} core fields are null"
        elif ratio < min_non_null_ratio:
            status = "DEGRADED"
            reason = f"Core-field non-null ratio {ratio:.2f} below threshold"
        else:
            status = "VALID"
            reason = "Section materially supported by data"

        if invalid_mode and section_name in {
            "market_prices",
            "fuel_prices",
            "equity_performance",
            "valuation",
            "factor_decomposition",
            "historical_analogues",
            "recommendations",
        }:
            status = "INVALID"
            reason = "Run status is INVALID; section suppressed for publishable conclusions"

        rows.append(
            {
                "section_name": section_name,
                "dataset": sheet_name,
                "section_status": status,
                "row_count": row_count,
                "core_non_null_ratio": round(ratio, 4),
                "minimum_required_ratio": min_non_null_ratio,
                "reason": reason,
            }
        )

    return pd.DataFrame(rows)


def build_chart_health(chart_sheets: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for chart in chart_sheets:
        rows.append(
            {
                "chart_name": chart.get("sheet_name", ""),
                "chart_status": chart.get("status", "UNKNOWN"),
                "required_columns": ", ".join(chart.get("required_columns", [])) if isinstance(chart.get("required_columns"), list) else str(chart.get("required_columns", "")),
                "row_count": int(chart.get("row_count", 0) or 0),
                "non_null_count": int(chart.get("non_null_count", 0) or 0),
                "minimum_non_null_ratio": chart.get("minimum_non_null_ratio", ""),
                "reason": chart.get("reason", ""),
                "image_path": chart.get("image_path", ""),
            }
        )
    return pd.DataFrame(rows)


def build_ranking_health(core_ranking_df: pd.DataFrame, extended_ranking_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "company_name",
        "ticker",
        "bucket_classification",
        "final_score",
        "score_real_data_share",
        "score_proxy_share",
        "score_default_share",
        "final_rating_confidence",
        "rating_status",
        "rating_gate_reason",
        "final_rating_confidence_reason",
        "publishable_flag",
        "publishable_gate_reason",
    ]

    ext = extended_ranking_df.copy() if extended_ranking_df is not None else pd.DataFrame()
    for c in cols:
        if c not in ext.columns:
            ext[c] = pd.NA

    ext = ext[cols]

    def _health_status(row: pd.Series) -> str:
        status = str(row.get("rating_status") or "").lower()
        publishable = bool(row.get("publishable_flag")) if pd.notna(row.get("publishable_flag")) else False
        if status == "unrated":
            return "UNRATED"
        if publishable:
            return "GOOD"
        if status == "rated":
            return "PROVISIONAL"
        return "DEGRADED"

    ext["ranking_health_status"] = ext.apply(_health_status, axis=1)
    ext["ranking_health_reason"] = ext.apply(
        lambda r: (
            "passes_publishable_gate"
            if bool(r.get("publishable_flag"))
            else "; ".join(
                [
                    str(r.get("rating_gate_reason", "")).strip(),
                    str(r.get("publishable_gate_reason", "")).strip(),
                    str(r.get("final_rating_confidence_reason", "")).strip(),
                ]
            ).strip("; ")
        ),
        axis=1,
    )
    ext["is_core"] = ext["ticker"].isin(core_ranking_df.get("ticker", pd.Series(dtype=str)))
    return ext
