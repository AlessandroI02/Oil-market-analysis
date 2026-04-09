from __future__ import annotations

import numpy as np
import pandas as pd


def _label(score: float) -> str:
    if score >= 80:
        return "High"
    if score >= 58:
        return "Medium"
    return "Low"


def _score_from_label(label: str) -> float:
    mapping = {"High": 82.0, "Medium": 62.0, "Low": 38.0}
    return mapping.get(str(label), 55.0)


def _as_float(value: object, fallback: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return fallback
        return float(value)
    except Exception:
        return fallback


def build_confidence_framework(
    included_df: pd.DataFrame,
    data_quality_df: pd.DataFrame,
    source_log_df: pd.DataFrame,
    route_risks_df: pd.DataFrame,
    event_study_summary_df: pd.DataFrame,
    regime_state_df: pd.DataFrame,
    ranking_health_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if included_df is None or included_df.empty:
        return pd.DataFrame()

    base = included_df[["company_name", "ticker", "bucket_classification", "confidence"]].copy()

    dq_cols = [
        "ticker",
        "data_quality_score",
        "proxy_assumption_share",
        "missing_field_count",
        "tier1_2_source_share",
        "exact_evidence_share",
    ]
    dq = (
        data_quality_df[[c for c in dq_cols if c in data_quality_df.columns]].copy()
        if (data_quality_df is not None and not data_quality_df.empty and "data_quality_score" in data_quality_df.columns)
        else pd.DataFrame(columns=dq_cols)
    )
    for col in dq_cols:
        if col not in dq.columns:
            dq[col] = pd.NA
    base = base.merge(dq, on="ticker", how="left")

    route = route_risks_df[["ticker", "qualitative_route_risk", "hormuz_share_pct"]].copy() if (
        route_risks_df is not None and not route_risks_df.empty and "ticker" in route_risks_df.columns
    ) else pd.DataFrame(columns=["ticker", "qualitative_route_risk", "hormuz_share_pct"])
    base = base.merge(route, on="ticker", how="left")

    event = event_study_summary_df[["ticker", "event_count", "event_day_hit_rate_up_oil", "event_day_hit_rate_down_oil"]].copy() if (
        event_study_summary_df is not None and not event_study_summary_df.empty and "ticker" in event_study_summary_df.columns
    ) else pd.DataFrame(columns=["ticker", "event_count", "event_day_hit_rate_up_oil", "event_day_hit_rate_down_oil"])
    base = base.merge(event, on="ticker", how="left")

    src_score_df = pd.DataFrame(columns=["ticker", "source_confidence_score", "source_summary"])
    if source_log_df is not None and not source_log_df.empty:
        src = source_log_df.copy()
        src["tier_score"] = src.get("source_tier", pd.Series(dtype=str)).astype(str).map(
            {"Tier 1": 1.0, "Tier 2": 0.85, "Tier 3": 0.65, "Tier 4": 0.45}
        ).fillna(0.4)
        src["evidence_score"] = src.get("evidence_flag", pd.Series(dtype=str)).astype(str).map(
            {"exact": 1.0, "estimated": 0.6, "proxy": 0.55, "missing": 0.1}
        ).fillna(0.45)
        src["composite"] = src["tier_score"] * 0.65 + src["evidence_score"] * 0.35
        src["domain"] = src.get("source_url", pd.Series(dtype=str)).astype(str).str.extract(r"https?://([^/]+)")[0]

        summary = (
            src.groupby("company")["domain"]
            .apply(lambda s: ", ".join(s.dropna().astype(str).value_counts().head(3).index.tolist()))
            .reset_index(name="source_summary")
        )
        by_company = src.groupby("company", as_index=False)["composite"].mean()
        by_company["source_confidence_score"] = by_company["composite"] * 100
        src_score_df = (
            by_company.merge(included_df[["company_name", "ticker"]], left_on="company", right_on="company_name", how="left")
            .merge(summary, on="company", how="left")
        )[["ticker", "source_confidence_score", "source_summary"]]

    base = base.merge(src_score_df, on="ticker", how="left")

    ranking_cols = [
        "ticker",
        "final_rating_confidence",
        "rating_status",
        "publishable_flag",
        "rating_gate_reason",
        "publishable_gate_reason",
        "score_real_data_share",
        "score_proxy_share",
        "score_default_share",
        "ranking_health_status",
    ]
    if ranking_health_df is not None and not ranking_health_df.empty and "ticker" in ranking_health_df.columns:
        rank = ranking_health_df.copy()
        for col in ranking_cols:
            if col not in rank.columns:
                rank[col] = pd.NA
        base = base.merge(rank[ranking_cols], on="ticker", how="left")
    else:
        for col in ranking_cols:
            if col != "ticker":
                base[col] = pd.NA

    regime_conf_label = "Medium"
    regime_conf_score = 62.0
    if regime_state_df is not None and not regime_state_df.empty:
        regime_conf_label = str(regime_state_df.iloc[0].get("regime_confidence", "Medium"))
        regime_conf_score = _score_from_label(regime_conf_label)

    rows: list[dict[str, object]] = []
    for _, row in base.iterrows():
        data_score_raw = _as_float(row.get("data_quality_score"), np.nan)
        if np.isnan(data_score_raw):
            data_score = _score_from_label(str(row.get("confidence", "Medium")))
        else:
            data_score = float(np.clip(data_score_raw, 0, 100))

        source_score_raw = _as_float(row.get("source_confidence_score"), np.nan)
        source_score = float(np.clip(source_score_raw, 0, 100)) if not np.isnan(source_score_raw) else 52.0
        if str(row.get("source_summary", "")).strip() == "":
            source_score = float(np.clip(source_score - 8.0, 0, 100))

        route_label = str(row.get("qualitative_route_risk", "Medium"))
        route_base = {"Low": 82.0, "Medium": 60.0, "High": 46.0}.get(route_label, 60.0)
        hormuz_share = _as_float(row.get("hormuz_share_pct"), np.nan)
        if pd.notna(hormuz_share):
            route_score = float(np.clip(route_base - (float(hormuz_share) * 0.15), 30.0, 90.0))
        else:
            route_score = route_base

        hit_up = _as_float(row.get("event_day_hit_rate_up_oil"), np.nan)
        hit_down = _as_float(row.get("event_day_hit_rate_down_oil"), np.nan)
        event_count = _as_float(row.get("event_count"), np.nan)
        if pd.notna(hit_up) or pd.notna(hit_down):
            mean_hit = np.nanmean([hit_up, hit_down])
            event_score = float(np.clip(28.0 + (mean_hit * 68.0), 20.0, 92.0))
            if pd.notna(event_count):
                event_score = float(np.clip(event_score + min(float(event_count), 12.0), 20.0, 92.0))
        else:
            event_score = 42.0

        regime_score = regime_conf_score
        packet_score_pre = float(
            np.clip(
                (0.35 * data_score)
                + (0.22 * source_score)
                + (0.14 * route_score)
                + (0.14 * event_score)
                + (0.15 * regime_score),
                0.0,
                100.0,
            )
        )

        rating_status = str(row.get("rating_status", "") or "").strip().lower() or "unrated"
        rating_conf = str(row.get("final_rating_confidence", "") or "Low")
        publishable = bool(row.get("publishable_flag")) if pd.notna(row.get("publishable_flag")) else False
        score_default_share = _as_float(row.get("score_default_share"), 0.0)
        score_proxy_share = _as_float(row.get("score_proxy_share"), 0.0)
        proxy_assumption_share = _as_float(row.get("proxy_assumption_share"), 0.0)
        missing_field_count = _as_float(row.get("missing_field_count"), 0.0)
        tier1_2_source_share = _as_float(row.get("tier1_2_source_share"), 0.0)
        exact_evidence_share = _as_float(row.get("exact_evidence_share"), 0.0)

        data_conf = _label(data_score)
        source_conf = _label(source_score)
        route_conf = _label(route_score)
        event_conf = _label(event_score)
        regime_conf = _label(regime_score)
        component_low_count = int(sum(lbl == "Low" for lbl in [data_conf, source_conf, route_conf, event_conf, regime_conf]))

        contradiction_reasons: list[str] = []
        if publishable and rating_status != "rated":
            contradiction_reasons.append("publishable_without_rated_status")
        if publishable and rating_conf == "Low":
            contradiction_reasons.append("publishable_with_low_rating_confidence")
        if rating_status == "rated" and not publishable:
            contradiction_reasons.append("rated_but_not_publishable")
        if rating_status == "unrated" and rating_conf in {"High", "Medium"}:
            contradiction_reasons.append("unrated_with_non_low_rating_confidence")
        contradiction_count = len(contradiction_reasons)

        high_gate_checks = {
            "strong_data_confidence": data_score >= 82.0,
            "strong_source_confidence": source_score >= 74.0 and tier1_2_source_share >= 0.34 and exact_evidence_share >= 0.30,
            "route_confidence_at_least_moderate": route_score >= 58.0,
            "event_confidence_at_least_moderate": event_score >= 58.0,
            "regime_confidence_not_weak": regime_score >= 58.0,
            "low_proxy_burden": proxy_assumption_share <= 0.30 and score_proxy_share <= 0.40,
            "low_missing_burden": missing_field_count <= 1.0 and score_default_share <= 0.12,
            "low_component_weakness": component_low_count <= 0,
            "low_contradiction_count": contradiction_count == 0,
        }
        high_eligible = bool(all(high_gate_checks.values()))
        failed_high_checks = [name for name, ok in high_gate_checks.items() if not ok]

        packet_score = packet_score_pre
        alignment_reasons: list[str] = []
        if not high_eligible and packet_score > 74.0:
            packet_score = 74.0
            alignment_reasons.append("high_packet_blocked_by_strict_gate")
        if rating_status == "unrated":
            packet_score = min(packet_score, 44.0)
            alignment_reasons.append("capped_for_unrated_status")
        elif rating_status != "rated":
            packet_score = min(packet_score, 54.0)
            alignment_reasons.append("capped_for_provisional_status")
        if not publishable:
            packet_score = min(packet_score, 57.0)
            alignment_reasons.append("capped_for_non_publishable_status")
        if rating_conf == "Low":
            packet_score = min(packet_score, 50.0)
            alignment_reasons.append("capped_for_low_rating_confidence")
        if score_default_share > 0.12:
            packet_score -= min((score_default_share - 0.12) * 55.0, 14.0)
            alignment_reasons.append("penalized_for_default_component_share")
        if score_proxy_share > 0.40:
            packet_score -= min((score_proxy_share - 0.40) * 30.0, 10.0)
            alignment_reasons.append("penalized_for_proxy_component_share")
        if proxy_assumption_share > 0.30:
            packet_score -= min((proxy_assumption_share - 0.30) * 36.0, 8.0)
            alignment_reasons.append("penalized_for_proxy_assumption_burden")
        if missing_field_count > 1.0:
            packet_score -= min((missing_field_count - 1.0) * 2.8, 10.0)
            alignment_reasons.append("penalized_for_missing_field_burden")
        if contradiction_count > 0:
            packet_score -= min(contradiction_count * 3.0, 9.0)
            alignment_reasons.append("penalized_for_confidence_contradictions")
        packet_score = float(np.clip(packet_score, 0.0, 100.0))

        packet_conf = _label(packet_score)
        if packet_conf == "High" and not high_eligible:
            packet_conf = "Medium"

        readiness_checks = {
            "rated_status_required": rating_status == "rated",
            "publishable_flag_required": publishable,
            "strong_rating_confidence_required": rating_conf == "High",
            "packet_score_threshold": packet_score >= 68.0,
            "data_source_quality_threshold": data_score >= 78.0 and source_score >= 72.0,
            "route_event_quality_threshold": route_score >= 58.0 and event_score >= 58.0,
            "model_burden_threshold": proxy_assumption_share <= 0.30 and score_default_share <= 0.12 and missing_field_count <= 1.0,
            "no_confidence_contradictions": contradiction_count == 0,
        }
        downstream_ready = bool(all(readiness_checks.values()))
        failed_readiness_checks = [name for name, ok in readiness_checks.items() if not ok]

        alignment_note = "; ".join(alignment_reasons) if alignment_reasons else "no_alignment_cap_applied"
        packet_threshold_reasons = "passes_high_threshold" if high_eligible else "; ".join(failed_high_checks)
        downstream_readiness_reason = "downstream_ready" if downstream_ready else "; ".join(failed_readiness_checks)
        proxy_burden_bucket = "Low" if proxy_assumption_share <= 0.30 else ("Medium" if proxy_assumption_share <= 0.45 else "High")
        missing_burden_bucket = "Low" if missing_field_count <= 1.0 else ("Medium" if missing_field_count <= 3.0 else "High")

        rows.append(
            {
                "company_name": row["company_name"],
                "ticker": row["ticker"],
                "bucket_classification": row.get("bucket_classification", ""),
                "input_data_confidence": data_conf,
                "route_model_confidence": route_conf,
                "event_model_confidence": event_conf,
                "regime_model_confidence": regime_conf,
                "data_confidence": data_conf,
                "source_confidence": source_conf,
                "route_confidence": route_conf,
                "event_confidence": event_conf,
                "regime_confidence": regime_conf,
                "packet_confidence": packet_conf,
                "company_packet_confidence": packet_conf,
                "downstream_readiness_flag": downstream_ready,
                "downstream_readiness_reason": downstream_readiness_reason,
                "publishable_flag": publishable,
                "rating_status": rating_status,
                "final_rating_confidence": rating_conf,
                "rating_gate_reason": row.get("rating_gate_reason", ""),
                "publishable_gate_reason": row.get("publishable_gate_reason", ""),
                "ranking_health_status": row.get("ranking_health_status", ""),
                "data_confidence_score": round(data_score, 4),
                "source_confidence_score": round(source_score, 4),
                "route_confidence_score": round(route_score, 4),
                "event_confidence_score": round(event_score, 4),
                "regime_confidence_score": round(regime_score, 4),
                "packet_confidence_score_pre_alignment": round(packet_score_pre, 4),
                "packet_confidence_score": round(packet_score, 4),
                "packet_high_eligibility": high_eligible,
                "packet_high_eligibility_reasons": packet_threshold_reasons,
                "component_low_count": component_low_count,
                "confidence_contradiction_count": contradiction_count,
                "confidence_contradiction_reasons": "; ".join(contradiction_reasons) if contradiction_reasons else "",
                "score_proxy_share": round(score_proxy_share, 4),
                "score_default_share": round(score_default_share, 4),
                "proxy_assumption_share": round(proxy_assumption_share, 4),
                "missing_field_count": int(round(missing_field_count)),
                "tier1_2_source_share": round(tier1_2_source_share, 4),
                "exact_evidence_share": round(exact_evidence_share, 4),
                "proxy_burden_bucket": proxy_burden_bucket,
                "missing_burden_bucket": missing_burden_bucket,
                "alignment_note": alignment_note,
                "source_summary": row.get("source_summary", ""),
            }
        )

    out = pd.DataFrame(rows).sort_values("packet_confidence_score", ascending=False).reset_index(drop=True)
    return out


def build_confidence_audit(
    confidence_framework_df: pd.DataFrame,
    ranking_health_df: pd.DataFrame,
) -> pd.DataFrame:
    if confidence_framework_df is None or confidence_framework_df.empty:
        return pd.DataFrame()

    conf = confidence_framework_df.copy()
    if ranking_health_df is not None and not ranking_health_df.empty and "ticker" in ranking_health_df.columns:
        rank = ranking_health_df[
            [
                c
                for c in [
                    "ticker",
                    "final_rating_confidence",
                    "rating_status",
                    "publishable_flag",
                    "ranking_health_status",
                ]
                if c in ranking_health_df.columns
            ]
        ].copy()
        conf = conf.drop(columns=[c for c in rank.columns if c != "ticker" and c in conf.columns], errors="ignore")
        conf = conf.merge(rank, on="ticker", how="left")

    conf["rating_status"] = conf.get("rating_status", pd.Series(dtype=str)).astype(str).str.lower().replace({"": "unrated"})
    conf["publishable_flag"] = conf.get("publishable_flag", pd.Series(dtype=bool)).fillna(False).astype(bool)
    conf["final_rating_confidence"] = conf.get("final_rating_confidence", pd.Series(dtype=str)).fillna("Low").astype(str)
    conf["packet_confidence"] = conf.get("packet_confidence", pd.Series(dtype=str)).fillna("Low").astype(str)
    conf["downstream_readiness_flag"] = conf.get("downstream_readiness_flag", pd.Series(dtype=bool)).fillna(False).astype(bool)
    conf["packet_high_eligibility"] = conf.get("packet_high_eligibility", pd.Series(dtype=bool)).fillna(False).astype(bool)
    conf["confidence_contradiction_count"] = pd.to_numeric(
        conf.get("confidence_contradiction_count", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    conf["component_low_count"] = pd.to_numeric(
        conf.get("component_low_count", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    conf["packet_confidence_score"] = pd.to_numeric(
        conf.get("packet_confidence_score", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)

    contradiction_checks = {
        "high_packet_without_high_eligibility": (conf["packet_confidence"] == "High") & (~conf["packet_high_eligibility"]),
        "high_or_medium_packet_with_unrated_status": (conf["packet_confidence"].isin(["High", "Medium"])) & (conf["rating_status"] == "unrated"),
        "high_or_medium_packet_non_publishable": (conf["packet_confidence"].isin(["High", "Medium"])) & (~conf["publishable_flag"]),
        "downstream_ready_without_strong_rating_confidence": conf["downstream_readiness_flag"] & (conf["final_rating_confidence"] != "High"),
        "downstream_ready_with_packet_low": conf["downstream_readiness_flag"] & (conf["packet_confidence"] == "Low"),
    }
    contradiction_count_series = sum(mask.astype(int) for mask in contradiction_checks.values())
    conf["audit_contradiction_count"] = contradiction_count_series

    contradiction_reason_map = []
    for idx in conf.index:
        reasons = [name for name, mask in contradiction_checks.items() if bool(mask.loc[idx])]
        contradiction_reason_map.append("; ".join(reasons))
    conf["audit_contradiction_reasons"] = contradiction_reason_map

    conf["confidence_contradiction_flag"] = conf["audit_contradiction_count"] > 0
    conf["audit_status"] = np.where(conf["confidence_contradiction_flag"], "MISALIGNED", "ALIGNED")
    conf["audit_note"] = np.where(
        conf["confidence_contradiction_flag"],
        "Confidence chain mismatch detected between packet/rating/publishable/readiness gates",
        "Confidence chain aligned across packet/rating/publishable/readiness gates",
    )

    out_cols = [
        "company_name",
        "ticker",
        "input_data_confidence",
        "source_confidence",
        "route_model_confidence",
        "event_model_confidence",
        "regime_model_confidence",
        "packet_confidence",
        "packet_confidence_score",
        "packet_high_eligibility",
        "packet_high_eligibility_reasons",
        "final_rating_confidence",
        "rating_status",
        "rating_gate_reason",
        "publishable_flag",
        "publishable_gate_reason",
        "downstream_readiness_flag",
        "downstream_readiness_reason",
        "component_low_count",
        "score_proxy_share",
        "score_default_share",
        "proxy_assumption_share",
        "missing_field_count",
        "tier1_2_source_share",
        "exact_evidence_share",
        "confidence_contradiction_count",
        "confidence_contradiction_reasons",
        "audit_contradiction_count",
        "audit_contradiction_reasons",
        "confidence_contradiction_flag",
        "audit_status",
        "audit_note",
        "alignment_note",
    ]
    for col in out_cols:
        if col not in conf.columns:
            conf[col] = ""
    return conf[out_cols].sort_values(
        ["confidence_contradiction_flag", "audit_contradiction_count", "packet_confidence_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
