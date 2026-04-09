from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


QUALITY_MAP = {"High": 1.0, "Medium": 0.7, "Low": 0.4}


def _to_float(value: object, fallback: float) -> float:
    try:
        if value is None or pd.isna(value):
            return fallback
        return float(value)
    except Exception:
        return fallback


def _subset_with_defaults(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns]


def _normalize_series(series: pd.Series, inverse: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([50.0] * len(series), index=series.index)
    lo = s.min()
    hi = s.max()
    if hi == lo:
        base = pd.Series([50.0] * len(series), index=series.index)
    else:
        base = (s - lo) / (hi - lo) * 100
    if inverse:
        base = 100 - base
    return base.fillna(50.0)


def _nearest_catalyst_score(catalyst_df: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "event_date", "event"}
    if catalyst_df.empty or not required.issubset(set(catalyst_df.columns)):
        return pd.DataFrame(columns=["ticker", "catalyst_score", "catalyst_note"])

    c = catalyst_df.copy()
    c = c[c["ticker"] != "GLOBAL"].copy()
    if c.empty:
        return pd.DataFrame(columns=["ticker", "catalyst_score", "catalyst_note"])

    score_map = {"Strong": 85, "Medium": 62, "Weak": 40}
    c["event_date"] = pd.to_datetime(c["event_date"], errors="coerce")
    c = c.sort_values("event_date")
    nearest = c.groupby("ticker", as_index=False).head(1)
    support_score = nearest.get("near_term_event_support", pd.Series(dtype=str)).map(score_map).fillna(50)
    if "market_relevance_score" in nearest.columns:
        relevance_score = pd.to_numeric(nearest["market_relevance_score"], errors="coerce")
    else:
        relevance_score = pd.Series([np.nan] * len(nearest), index=nearest.index)
    nearest["catalyst_score"] = support_score
    if relevance_score.notna().any():
        nearest["catalyst_score"] = nearest[["catalyst_score"]].join(relevance_score.rename("relevance")).max(axis=1)
    confidence_bonus = nearest.get("catalyst_confidence", pd.Series(dtype=str)).map({"High": 4, "Medium": 0, "Low": -6}).fillna(0)
    nearest["catalyst_score"] = (pd.to_numeric(nearest["catalyst_score"], errors="coerce").fillna(50) + confidence_bonus).clip(lower=25, upper=95)
    nearest["catalyst_note"] = nearest["event"]
    return nearest[["ticker", "catalyst_score", "catalyst_note"]]


def _component_status(value: object, mode: str) -> str:
    if value is None:
        return "default"
    try:
        if pd.isna(value):
            return "default"
    except Exception:
        pass

    if mode == "proxy":
        return "proxy"
    if mode == "real":
        return "real"
    return "real"


def _rating_status(
    real_share: float,
    proxy_share: float,
    default_share: float,
    data_conf_score: float,
    source_tier_share: float,
    exact_evidence_share: float,
    route_model_burden_flag: bool,
    route_risk_label: str,
    event_support_score: float,
) -> tuple[str, str]:
    hard_unrated_reasons: list[str] = []
    if default_share > 0.35:
        hard_unrated_reasons.append("high_default_component_share")
    if data_conf_score < 58:
        hard_unrated_reasons.append("low_data_confidence_score")
    if proxy_share > 0.46:
        hard_unrated_reasons.append("high_proxy_component_share")
    if source_tier_share < 0.28:
        hard_unrated_reasons.append("weak_tier1_2_source_coverage")
    if exact_evidence_share < 0.22:
        hard_unrated_reasons.append("low_exact_evidence_share")
    if len(hard_unrated_reasons) >= 2:
        return "unrated", "; ".join(hard_unrated_reasons)

    rated_checks = [
        real_share >= 0.62,
        default_share <= 0.15,
        proxy_share <= 0.40,
        data_conf_score >= 72,
        source_tier_share >= 0.34,
        exact_evidence_share >= 0.28,
        route_risk_label != "High",
        event_support_score >= 56,
    ]
    if all(rated_checks):
        if route_model_burden_flag:
            if data_conf_score >= 86 and source_tier_share >= 0.40 and exact_evidence_share >= 0.32 and event_support_score >= 62:
                return "rated", "rated_with_route_model_penalty_but_supported_by_strong_data_and_sources"
            return "provisional", "route_model_burden_requires_more_evidence"
        return "rated", "meets_strict_rating_thresholds"

    provisional_reasons: list[str] = []
    if real_share < 0.62:
        provisional_reasons.append("insufficient_real_data_share")
    if proxy_share > 0.40:
        provisional_reasons.append("proxy_component_share_elevated")
    if default_share > 0.15:
        provisional_reasons.append("default_component_share_elevated")
    if route_risk_label == "High":
        provisional_reasons.append("route_risk_too_high_for_rated_status")
    if event_support_score < 56:
        provisional_reasons.append("event_support_below_rated_threshold")
    if source_tier_share < 0.34:
        provisional_reasons.append("tier1_2_source_share_below_threshold")
    if exact_evidence_share < 0.28:
        provisional_reasons.append("exact_evidence_share_below_threshold")
    if not provisional_reasons:
        provisional_reasons.append("mixed_quality_signal")
    return "provisional", "; ".join(provisional_reasons)


def _rating_confidence(
    status: str,
    data_conf_score: float,
    default_share: float,
    proxy_share: float,
    source_tier_share: float,
    exact_evidence_share: float,
    route_model_burden_flag: bool,
    route_risk_label: str,
    event_support_score: float,
) -> tuple[str, str]:
    status_norm = str(status).lower()
    if status_norm == "unrated":
        return "Low", "unrated_names_cannot_be_high_confidence"

    high_conditions = [
        data_conf_score >= 86,
        default_share <= 0.10,
        proxy_share <= 0.38,
        source_tier_share >= 0.40,
        exact_evidence_share >= 0.33,
        route_risk_label == "Low",
        event_support_score >= 64,
    ]
    if all(high_conditions):
        if route_model_burden_flag and not (
            data_conf_score >= 90 and exact_evidence_share >= 0.34 and source_tier_share >= 0.40 and event_support_score >= 66
        ):
            return "Medium", "route_model_burden_caps_high_rating_confidence"
        return "High", "meets_strict_high_confidence_thresholds"

    medium_conditions = [
        data_conf_score >= 70,
        default_share <= 0.20,
        proxy_share <= 0.42,
        source_tier_share >= 0.32,
        exact_evidence_share >= 0.25,
        event_support_score >= 55,
    ]
    if all(medium_conditions):
        return "Medium", "meets_medium_confidence_thresholds"
    return "Low", "fails_medium_confidence_thresholds"


def build_rankings(
    included_df: pd.DataFrame,
    archetypes_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    route_risks_df: pd.DataFrame,
    valuation_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    event_path_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    catalyst_df: pd.DataFrame,
    data_quality_df: pd.DataFrame,
    weight_config: dict[str, float],
    confidence_penalty_map: dict[str, float],
) -> dict[str, pd.DataFrame]:
    if included_df is None or included_df.empty:
        empty = pd.DataFrame()
        return {
            "core_ranking": empty,
            "extended_ranking": empty,
            "recommendation_framework": empty,
            "company_scorecards": empty,
        }

    base = _subset_with_defaults(
        included_df,
        ["company_name", "ticker", "bucket_classification", "confidence"],
    )
    base = base.merge(
        _subset_with_defaults(archetypes_df, ["ticker", "archetype", "inclusion_confidence"]),
        on="ticker",
        how="left",
    )
    base = base.merge(
        _subset_with_defaults(
            exposure_df,
            [
                "ticker",
                "combined_exposure_pct",
                "confidence",
                "exposure_low_pct",
                "exposure_high_pct",
                "exact_vs_estimated",
            ],
        ),
        on="ticker",
        how="left",
        suffixes=("", "_exposure"),
    )
    base = base.merge(
        _subset_with_defaults(route_risks_df, ["ticker", "qualitative_route_risk"]),
        on="ticker",
        how="left",
    )
    base = base.merge(
        _subset_with_defaults(
            valuation_df,
            [
                "ticker",
                "valuation_score",
                "leverage_ratio",
                "fcf_yield_pct",
                "dividend_yield_pct",
                "data_flag",
            ],
        ),
        on="ticker",
        how="left",
    )

    scen = (
        _subset_with_defaults(
            scenario_df[scenario_df["scenario_brent"] == 100],
            ["ticker", "ebitda_sensitivity_pct", "fcf_sensitivity_pct", "payout_resilience"],
        )
        if (scenario_df is not None and not scenario_df.empty and "scenario_brent" in scenario_df.columns)
        else pd.DataFrame(columns=["ticker", "ebitda_sensitivity_pct", "fcf_sensitivity_pct", "payout_resilience"])
    )
    base = base.merge(scen, on="ticker", how="left")

    ep = (
        _subset_with_defaults(
            event_path_df[event_path_df["event_path"] == "Hormuz disrupted through earnings cycle"],
            ["ticker", "scenario_impact_score"],
        )
        if (event_path_df is not None and not event_path_df.empty and "event_path" in event_path_df.columns)
        else pd.DataFrame(columns=["ticker", "scenario_impact_score"])
    )
    base = base.merge(ep, on="ticker", how="left")

    fac = (
        _subset_with_defaults(
            factor_df,
            ["ticker", "beta_brent_ret", "beta_market_ret", "beta_energy_ret", "idiosyncratic_residual_pct", "r2"],
        )
        if factor_df is not None
        else pd.DataFrame(columns=["ticker", "beta_brent_ret", "beta_market_ret", "beta_energy_ret", "idiosyncratic_residual_pct", "r2"])
    )
    base = base.merge(fac, on="ticker", how="left")

    cat = _nearest_catalyst_score(catalyst_df)
    base = base.merge(cat, on="ticker", how="left")
    dq = _subset_with_defaults(
        data_quality_df,
        [
            "ticker",
            "data_quality_score",
            "data_quality_bucket",
            "proxy_assumption_share",
            "missing_field_count",
            "tier1_2_source_share",
            "exact_evidence_share",
        ],
    )
    base = base.merge(dq, on="ticker", how="left")

    # Component scores
    base["route_exposure_score"] = _normalize_series(base["combined_exposure_pct"], inverse=True)
    base["oil_sensitivity_score"] = _normalize_series(base["beta_brent_ret"], inverse=False)
    base["downstream_pass_through_score"] = _normalize_series(base["fcf_sensitivity_pct"], inverse=False)
    base["valuation_component_score"] = pd.to_numeric(base["valuation_score"], errors="coerce").fillna(50)
    base["balance_sheet_score"] = _normalize_series(base["leverage_ratio"], inverse=True)
    base["market_positioning_score"] = _normalize_series(base["idiosyncratic_residual_pct"], inverse=False)
    base["scenario_resilience_score"] = _normalize_series(base["scenario_impact_score"], inverse=False)
    base["catalyst_score"] = pd.to_numeric(base["catalyst_score"], errors="coerce").fillna(50)

    # Component availability accounting
    component_modes = {
        "route_exposure": lambda r: "proxy" if str(r.get("exact_vs_estimated", "")).lower() in {"estimated", "proxy"} else "real",
        "oil_sensitivity": lambda r: "real",
        "downstream_pass_through": lambda r: "proxy",
        "valuation": lambda r: "proxy" if str(r.get("data_flag", "")).lower() in {"estimated", "proxy"} else "real",
        "balance_sheet": lambda r: "proxy" if str(r.get("data_flag", "")).lower() in {"estimated", "proxy"} else "real",
        "catalyst": lambda r: "real",
        "market_positioning": lambda r: "real",
        "scenario_resilience": lambda r: "proxy",
    }
    component_values = {
        "route_exposure": "combined_exposure_pct",
        "oil_sensitivity": "beta_brent_ret",
        "downstream_pass_through": "fcf_sensitivity_pct",
        "valuation": "valuation_score",
        "balance_sheet": "leverage_ratio",
        "catalyst": "catalyst_score",
        "market_positioning": "idiosyncratic_residual_pct",
        "scenario_resilience": "scenario_impact_score",
    }

    real_shares: list[float] = []
    proxy_shares: list[float] = []
    default_shares: list[float] = []
    details: list[str] = []

    for _, row in base.iterrows():
        statuses: list[str] = []
        parts: list[str] = []
        for comp, col in component_values.items():
            mode = component_modes[comp](row)
            status = _component_status(row.get(col), mode)
            statuses.append(status)
            parts.append(f"{comp}:{status}")

        total = len(statuses)
        real_count = statuses.count("real")
        proxy_count = statuses.count("proxy")
        default_count = statuses.count("default")

        real_shares.append(real_count / total)
        proxy_shares.append(proxy_count / total)
        default_shares.append(default_count / total)
        details.append("; ".join(parts))

    base["score_real_data_share"] = real_shares
    base["score_proxy_share"] = proxy_shares
    base["score_default_share"] = default_shares
    base["score_component_detail"] = details

    # Confidence penalty
    conf = base["inclusion_confidence"].fillna(base["confidence"]).fillna("Medium")
    base["confidence_penalty_base"] = conf.map(confidence_penalty_map).fillna(confidence_penalty_map.get("Medium", 0.05)) * 100
    base["data_confidence_score"] = pd.to_numeric(base["data_quality_score"], errors="coerce")
    fallback_conf_score = conf.map(QUALITY_MAP).fillna(0.7) * 100
    base["data_confidence_score"] = base["data_confidence_score"].fillna(fallback_conf_score)

    base["proxy_burden_penalty"] = (
        pd.to_numeric(base["proxy_assumption_share"], errors="coerce").fillna(0.0) * 6.0
        + pd.to_numeric(base["missing_field_count"], errors="coerce").fillna(0.0).clip(0, 10) * 0.25
    )
    base["missing_component_penalty"] = base["score_default_share"].fillna(1.0) * 20.0
    base["confidence_penalty"] = (
        base["confidence_penalty_base"] + base["proxy_burden_penalty"] + base["missing_component_penalty"]
    )

    base["final_score_raw"] = (
        base["route_exposure_score"] * weight_config.get("route_exposure", 0.2)
        + base["oil_sensitivity_score"] * weight_config.get("oil_sensitivity", 0.12)
        + base["downstream_pass_through_score"] * weight_config.get("downstream_pass_through", 0.08)
        + base["valuation_component_score"] * weight_config.get("valuation", 0.18)
        + base["balance_sheet_score"] * weight_config.get("balance_sheet", 0.12)
        + base["catalyst_score"] * weight_config.get("catalyst", 0.10)
        + base["market_positioning_score"] * weight_config.get("market_positioning", 0.10)
        + base["scenario_resilience_score"] * weight_config.get("scenario_resilience", 0.10)
    )
    base["final_score"] = (base["final_score_raw"] - base["confidence_penalty"]).clip(lower=0, upper=100)

    base["route_model_burden_flag"] = (
        base.get("exact_vs_estimated", pd.Series(dtype=str)).astype(str).str.lower().isin({"estimated", "proxy"})
    )
    base["event_support_score"] = pd.to_numeric(base.get("catalyst_score"), errors="coerce").fillna(50.0)
    base["tier1_2_source_share"] = pd.to_numeric(base.get("tier1_2_source_share"), errors="coerce").fillna(0.0)
    base["exact_evidence_share"] = pd.to_numeric(base.get("exact_evidence_share"), errors="coerce").fillna(0.0)
    base["proxy_assumption_share"] = pd.to_numeric(base.get("proxy_assumption_share"), errors="coerce").fillna(0.0)
    base["missing_field_count"] = pd.to_numeric(base.get("missing_field_count"), errors="coerce").fillna(0.0)

    rating_status_and_reason = base.apply(
        lambda r: _rating_status(
            real_share=_to_float(r.get("score_real_data_share"), 0.0),
            proxy_share=_to_float(r.get("score_proxy_share"), 1.0),
            default_share=_to_float(r.get("score_default_share"), 1.0),
            data_conf_score=_to_float(r.get("data_confidence_score"), 0.0),
            source_tier_share=_to_float(r.get("tier1_2_source_share"), 0.0),
            exact_evidence_share=_to_float(r.get("exact_evidence_share"), 0.0),
            route_model_burden_flag=bool(r.get("route_model_burden_flag", False)),
            route_risk_label=str(r.get("qualitative_route_risk", "")),
            event_support_score=_to_float(r.get("event_support_score"), 0.0),
        ),
        axis=1,
    )
    base["rating_status"] = rating_status_and_reason.map(lambda x: x[0])
    base["rating_gate_reason"] = rating_status_and_reason.map(lambda x: x[1])

    rating_conf_and_reason = base.apply(
        lambda r: _rating_confidence(
            status=str(r.get("rating_status") or "unrated"),
            data_conf_score=_to_float(r.get("data_confidence_score"), 0.0),
            default_share=_to_float(r.get("score_default_share"), 1.0),
            proxy_share=_to_float(r.get("score_proxy_share"), 1.0),
            source_tier_share=_to_float(r.get("tier1_2_source_share"), 0.0),
            exact_evidence_share=_to_float(r.get("exact_evidence_share"), 0.0),
            route_model_burden_flag=bool(r.get("route_model_burden_flag", False)),
            route_risk_label=str(r.get("qualitative_route_risk", "")),
            event_support_score=_to_float(r.get("event_support_score"), 0.0),
        ),
        axis=1,
    )
    base["final_rating_confidence"] = rating_conf_and_reason.map(lambda x: x[0])
    base["final_rating_confidence_reason"] = rating_conf_and_reason.map(lambda x: x[1])

    def _publishable_eval(row: pd.Series) -> tuple[bool, str]:
        checks = {
            "rated_status_required": str(row.get("rating_status", "")).lower() == "rated",
            "high_rating_confidence_required": str(row.get("final_rating_confidence", "")) == "High",
            "data_confidence_score_threshold": _to_float(row.get("data_confidence_score"), 0.0) >= 82.0,
            "default_share_threshold": _to_float(row.get("score_default_share"), 1.0) <= 0.12,
            "proxy_component_threshold": _to_float(row.get("score_proxy_share"), 1.0) <= 0.38,
            "proxy_assumption_threshold": _to_float(row.get("proxy_assumption_share"), 1.0) <= 0.30,
            "tier1_2_source_threshold": _to_float(row.get("tier1_2_source_share"), 0.0) >= 0.35,
            "exact_evidence_threshold": _to_float(row.get("exact_evidence_share"), 0.0) >= 0.30,
            "route_risk_not_high": str(row.get("qualitative_route_risk", "")) != "High",
            "event_support_threshold": _to_float(row.get("event_support_score"), 0.0) >= 60.0,
        }
        passed = bool(all(checks.values()))
        failed = [name for name, ok in checks.items() if not ok]
        reason = "passes_publishable_gate" if passed else "; ".join(failed)
        return passed, reason

    publishable_eval = base.apply(_publishable_eval, axis=1)
    base["publishable_flag"] = publishable_eval.map(lambda x: bool(x[0]))
    base["publishable_gate_reason"] = publishable_eval.map(lambda x: x[1])

    base = base.sort_values(["publishable_flag", "final_score"], ascending=[False, False]).reset_index(drop=True)

    extended = base.copy()
    extended["extended_rank"] = extended.index + 1

    core = extended[extended["bucket_classification"] == "primary"].copy().reset_index(drop=True)
    core["core_rank"] = core.index + 1

    # Recommendation framework
    recommendations: list[dict[str, Any]] = []

    def _append_category(df: pd.DataFrame, label: str, n: int, rationale: str, publishable_only: bool = False) -> None:
        if df.empty:
            return
        pool = df.copy()
        if publishable_only and "publishable_flag" in pool.columns:
            pool = pool[pool["publishable_flag"] == True]  # noqa: E712
        if pool.empty:
            return
        for _, r in pool.head(n).iterrows():
            recommendations.append(
                {
                    "category": label,
                    "company_name": r["company_name"],
                    "ticker": r["ticker"],
                    "final_score": r["final_score"],
                    "confidence": r.get("final_rating_confidence", r.get("inclusion_confidence", r.get("confidence", "Medium"))),
                    "rating_status": r.get("rating_status", "provisional"),
                    "publishable_flag": bool(r.get("publishable_flag", False)),
                    "rationale": rationale,
                }
            )

    _append_category(
        core.sort_values("final_score", ascending=False),
        "Market-side attractive expression",
        3,
        "Higher composite score with sufficient publishable market-data support.",
        publishable_only=True,
    )
    _append_category(
        extended.sort_values("oil_sensitivity_score", ascending=False),
        "Higher-beta market expression",
        2,
        "Higher oil beta and event sensitivity for tactical market expression.",
    )
    _append_category(
        core.sort_values(["route_exposure_score", "balance_sheet_score"], ascending=False),
        "Lower-route-risk market expression",
        2,
        "Lower route concentration and stronger balance-sheet resilience.",
        publishable_only=True,
    )
    lower_conf_pool = extended[
        (extended["publishable_flag"] == False)  # noqa: E712
        | (extended["rating_status"].astype(str).str.lower() != "rated")
    ].sort_values(["score_default_share", "final_score"], ascending=[False, True])
    _append_category(
        lower_conf_pool,
        "Lower-confidence screen",
        4,
        "Output screened with lower confidence due to unrated/provisional status or elevated default/proxy burden.",
    )

    recommendation_df = pd.DataFrame(recommendations).drop_duplicates(subset=["category", "ticker"])

    scorecards = extended[[
        "company_name",
        "ticker",
        "bucket_classification",
        "archetype",
        "combined_exposure_pct",
        "exposure_low_pct",
        "exposure_high_pct",
        "route_exposure_score",
        "valuation_component_score",
        "balance_sheet_score",
        "scenario_resilience_score",
        "catalyst_score",
        "market_positioning_score",
        "data_confidence_score",
        "proxy_assumption_share",
        "missing_field_count",
        "score_real_data_share",
        "score_proxy_share",
        "score_default_share",
        "route_model_burden_flag",
        "event_support_score",
        "tier1_2_source_share",
        "exact_evidence_share",
        "rating_status",
        "rating_gate_reason",
        "final_rating_confidence",
        "final_rating_confidence_reason",
        "publishable_flag",
        "publishable_gate_reason",
        "confidence_penalty",
        "final_score",
        "score_component_detail",
    ]].copy()

    logger.info("Built rankings: core=%s extended=%s recs=%s", len(core), len(extended), len(recommendation_df))
    return {
        "core_ranking": core,
        "extended_ranking": extended,
        "recommendation_framework": recommendation_df,
        "company_scorecards": scorecards,
    }
