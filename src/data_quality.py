from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _bucket(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def build_data_quality_table(
    included_df: pd.DataFrame,
    assumptions_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    source_log_df: pd.DataFrame,
) -> pd.DataFrame:
    if included_df is None or included_df.empty:
        return pd.DataFrame(
            columns=[
                "company_name",
                "ticker",
                "assumption_count",
                "proxy_assumption_count",
                "analyst_assumption_count",
                "exact_assumption_count",
                "proxy_assumption_share",
                "missing_field_count",
                "high_severity_missing_count",
                "source_count",
                "tier1_2_source_share",
                "exact_evidence_share",
                "data_quality_score",
                "data_quality_bucket",
                "quality_notes",
            ]
        )

    assumptions_df = assumptions_df if assumptions_df is not None else pd.DataFrame()
    missing_df = missing_df if missing_df is not None else pd.DataFrame()
    source_log_df = source_log_df if source_log_df is not None else pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, r in included_df.iterrows():
        company = r.get("company_name")
        ticker = r.get("ticker")

        ass = assumptions_df[assumptions_df.get("company") == company] if not assumptions_df.empty else pd.DataFrame()
        miss = missing_df[missing_df.get("company") == company] if not missing_df.empty else pd.DataFrame()
        src = source_log_df[source_log_df.get("company") == company] if not source_log_df.empty else pd.DataFrame()

        assumption_count = int(len(ass))
        proxy_assumption_count = int((ass.get("estimate_type") == "proxy_estimate").sum()) if not ass.empty else 0
        analyst_assumption_count = int((ass.get("estimate_type") == "analyst_estimate").sum()) if not ass.empty else 0
        exact_assumption_count = int((ass.get("estimate_type").isin(["inferred_exact", "disclosed_exact"])).sum()) if not ass.empty else 0
        proxy_assumption_share = _safe_ratio(proxy_assumption_count, assumption_count)

        missing_field_count = int(len(miss))
        high_sev = 0
        if not miss.empty and "severity" in miss.columns:
            high_sev = int((miss["severity"].astype(str).str.lower() == "high").sum())

        source_count = int(len(src))
        tier12 = 0
        exact_evidence = 0
        if not src.empty:
            src_tier = src.get("source_tier", pd.Series(dtype=str)).astype(str)
            tier12 = int(src_tier.isin(["Tier 1", "Tier 2"]).sum())
            evidence = src.get("evidence_flag", pd.Series(dtype=str)).astype(str)
            exact_evidence = int((evidence == "exact").sum())

        tier12_share = _safe_ratio(tier12, source_count)
        exact_evidence_share = _safe_ratio(exact_evidence, source_count)

        # Penalize proxy burden and missing data; reward stronger source tiers and exact evidence.
        score = 100.0
        score -= proxy_assumption_share * 30.0
        score -= min(missing_field_count, 25) * 2.0
        score -= min(high_sev, 10) * 3.5
        score += tier12_share * 12.0
        score += exact_evidence_share * 8.0
        score = float(np.clip(score, 0.0, 100.0))

        notes = []
        if proxy_assumption_share >= 0.6:
            notes.append("High proxy burden")
        if missing_field_count >= 6:
            notes.append("Material missing-data load")
        if tier12_share < 0.3 and source_count > 0:
            notes.append("Low Tier 1/2 source coverage")
        if not notes:
            notes.append("Data quality acceptable for directional use")

        rows.append(
            {
                "company_name": company,
                "ticker": ticker,
                "assumption_count": assumption_count,
                "proxy_assumption_count": proxy_assumption_count,
                "analyst_assumption_count": analyst_assumption_count,
                "exact_assumption_count": exact_assumption_count,
                "proxy_assumption_share": round(proxy_assumption_share, 4),
                "missing_field_count": missing_field_count,
                "high_severity_missing_count": high_sev,
                "source_count": source_count,
                "tier1_2_source_share": round(tier12_share, 4),
                "exact_evidence_share": round(exact_evidence_share, 4),
                "data_quality_score": round(score, 2),
                "data_quality_bucket": _bucket(score),
                "quality_notes": "; ".join(notes),
            }
        )

    out = pd.DataFrame(rows).sort_values("data_quality_score", ascending=False).reset_index(drop=True)
    logger.info("Built data quality table rows: %s", len(out))
    return out

