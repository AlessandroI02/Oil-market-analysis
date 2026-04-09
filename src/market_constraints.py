from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def _safe_num(value: object, fallback: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return fallback
        return float(value)
    except Exception:
        return fallback


def _bucket_label(score: float) -> str:
    if score >= 74:
        return "High overlay"
    if score >= 48:
        return "Moderate overlay"
    return "Low overlay"


def _round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(round(value / step) * step, 6)


def _range_text(center: float, low_width: float, high_width: float) -> str:
    lo = max(0.0, center - low_width)
    hi = center + high_width
    return f"{int(round(lo))}-{int(round(hi))}"


def _beta_range_text(center: float, width: float = 0.05) -> str:
    lo = center - width
    hi = center + width
    return f"{lo:.2f} to {hi:.2f}"


def _scenario_shift_text(stress_score: float, regime_label: str, route_label: str, event_torque: float) -> str:
    supply_labels = {"geopolitical supply-shock regime", "route-disruption regime", "sanctions-driven regime"}
    if regime_label in supply_labels and stress_score >= 70:
        return "Supply-shock tilt (+15 to +20pp high-oil stress probability)"
    if regime_label in supply_labels and (stress_score >= 50 or route_label == "High"):
        return "Supply-shock tilt (+10 to +15pp high-oil stress probability)"
    if event_torque >= 60 or stress_score >= 48:
        return "Moderate stress tilt (+5 to +10pp stress probability)"
    if regime_label == "macro risk-off regime":
        return "Demand-downside tilt (+5 to +10pp downside probability)"
    return "Neutral-to-light stress tilt (0 to +5pp stress probability)"


def _confidence_penalty_score(
    proxy_share: float,
    quality_score: float,
    packet_conf: str,
    publishable: bool,
) -> float:
    packet_base = {"High": 0.05, "Medium": 0.12, "Low": 0.22}.get(str(packet_conf), 0.15)
    penalty = packet_base + max(0.0, proxy_share - 0.30) * 0.35 + max(0.0, (65.0 - quality_score) / 100.0)
    if not publishable:
        penalty += 0.08
    return float(np.clip(penalty, 0.0, 0.45))


def _constraint_confidence(
    packet_conf: str,
    quality_score: float,
    proxy_share: float,
    publishable: bool,
) -> str:
    conf = str(packet_conf)
    if conf == "High" and quality_score >= 75 and proxy_share <= 0.35 and publishable:
        return "High"
    if conf in {"High", "Medium"} and quality_score >= 55:
        return "Medium"
    return "Low"


def build_market_constraints(
    included_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    route_risks_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    data_quality_df: pd.DataFrame,
    regime_state_df: pd.DataFrame,
    as_of_date: date,
    confidence_framework_df: pd.DataFrame | None = None,
    event_study_summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if included_df is None or included_df.empty:
        return pd.DataFrame()

    base = included_df[["company_name", "ticker", "bucket_classification"]].copy()

    exp_cols = [c for c in ["ticker", "combined_exposure_pct", "exposure_low_pct", "exposure_high_pct"] if c in exposure_df.columns]
    route_cols = [c for c in ["ticker", "qualitative_route_risk", "hormuz_share_pct", "bab_el_mandeb_share_pct", "suez_share_pct"] if c in route_risks_df.columns]
    factor_cols = [c for c in ["ticker", "beta_brent_ret", "beta_market_ret", "beta_energy_ret"] if c in factor_df.columns]
    qual_cols = [c for c in ["ticker", "data_quality_score", "proxy_assumption_share"] if c in data_quality_df.columns]

    if exp_cols:
        base = base.merge(exposure_df[exp_cols], on="ticker", how="left")
    if route_cols:
        base = base.merge(route_risks_df[route_cols], on="ticker", how="left")
    if factor_cols:
        base = base.merge(factor_df[factor_cols], on="ticker", how="left")
    if qual_cols:
        base = base.merge(data_quality_df[qual_cols], on="ticker", how="left")

    if scenario_df is not None and not scenario_df.empty and {"ticker", "scenario_brent", "fcf_sensitivity_pct"}.issubset(set(scenario_df.columns)):
        scen_100 = scenario_df[pd.to_numeric(scenario_df["scenario_brent"], errors="coerce") == 100.0].copy()
        scen_100 = scen_100[["ticker", "fcf_sensitivity_pct"]].rename(columns={"fcf_sensitivity_pct": "scenario_fcf_sensitivity_pct"})
        base = base.merge(scen_100, on="ticker", how="left")
    else:
        base["scenario_fcf_sensitivity_pct"] = np.nan

    if event_study_summary_df is not None and not event_study_summary_df.empty and "ticker" in event_study_summary_df.columns:
        evt_cols = [c for c in ["ticker", "event_day_hit_rate_up_oil", "event_day_hit_rate_down_oil", "avg_abnormal_return_oil_up", "avg_abnormal_return_oil_down"] if c in event_study_summary_df.columns]
        base = base.merge(event_study_summary_df[evt_cols], on="ticker", how="left")
    else:
        base["event_day_hit_rate_up_oil"] = np.nan
        base["event_day_hit_rate_down_oil"] = np.nan
        base["avg_abnormal_return_oil_up"] = np.nan
        base["avg_abnormal_return_oil_down"] = np.nan

    if confidence_framework_df is not None and not confidence_framework_df.empty and "ticker" in confidence_framework_df.columns:
        conf_cols = [c for c in ["ticker", "packet_confidence", "downstream_readiness_flag", "publishable_flag"] if c in confidence_framework_df.columns]
        base = base.merge(confidence_framework_df[conf_cols], on="ticker", how="left")
    else:
        base["packet_confidence"] = "Medium"
        base["downstream_readiness_flag"] = False
        base["publishable_flag"] = False

    regime_label = "normal"
    regime_score = 35.0
    if regime_state_df is not None and not regime_state_df.empty:
        regime_label = str(regime_state_df.iloc[0].get("regime_label", "normal"))
        regime_score = _safe_num(regime_state_df.iloc[0].get("regime_score_total"), 35.0)

    regime_multiplier = {
        "normal": 0.95,
        "stressed oil regime": 1.10,
        "geopolitical supply-shock regime": 1.18,
        "sanctions-driven regime": 1.15,
        "route-disruption regime": 1.22,
        "macro risk-off regime": 1.12,
        "mixed / transition regime": 1.05,
    }.get(regime_label, 1.05)

    rows: list[dict[str, object]] = []
    for _, row in base.iterrows():
        exposure = _safe_num(row.get("combined_exposure_pct"), 0.0)
        route_label = str(row.get("qualitative_route_risk", "Medium"))
        route_score = {"Low": 26.0, "Medium": 58.0, "High": 84.0}.get(route_label, 58.0)
        hormuz_share = _safe_num(row.get("hormuz_share_pct"), 0.0)
        route_concentration = float(np.clip((exposure * 0.6) + (hormuz_share * 0.4), 0.0, 100.0))

        oil_beta = _safe_num(row.get("beta_brent_ret"), 1.0)
        market_beta = _safe_num(row.get("beta_market_ret"), 1.0)
        sector_beta = _safe_num(row.get("beta_energy_ret"), 1.0)
        quality_score = _safe_num(row.get("data_quality_score"), 60.0)
        proxy_share = _safe_num(row.get("proxy_assumption_share"), 0.30)
        scenario_sens = _safe_num(row.get("scenario_fcf_sensitivity_pct"), 0.0)
        hit_up = _safe_num(row.get("event_day_hit_rate_up_oil"), 0.50)
        hit_down = _safe_num(row.get("event_day_hit_rate_down_oil"), 0.50)
        abn_up = _safe_num(row.get("avg_abnormal_return_oil_up"), 0.0)
        abn_down = _safe_num(row.get("avg_abnormal_return_oil_down"), 0.0)

        event_torque = float(
            np.clip(
                (max(hit_up, hit_down) * 55.0) + (abs(abn_up - abn_down) * 700.0),
                0.0,
                100.0,
            )
        )
        beta_stress = float(np.clip((oil_beta - 0.9) * 42.0 + (sector_beta - market_beta) * 25.0, 0.0, 100.0))
        quality_penalty = float(np.clip(max((72.0 - quality_score), 0.0) * 0.9 + proxy_share * 18.0, 0.0, 45.0))

        stress_score = (
            (0.34 * route_concentration)
            + (0.20 * route_score)
            + (0.20 * beta_stress)
            + (0.16 * event_torque)
            + (0.10 * quality_penalty)
        ) * regime_multiplier + (regime_score * 0.08)
        stress_score = float(np.clip(stress_score, 0.0, 100.0))

        discount_center = _round_to_step(float(np.clip(35.0 + (stress_score * 2.5), 25.0, 310.0)), 5.0)
        discount_range = _range_text(discount_center, 20.0 if stress_score < 55 else 30.0, 30.0 if stress_score < 55 else 45.0)

        beta_center = float(
            np.clip(
                ((oil_beta - 1.0) * 0.18) + ((stress_score - 50.0) / 420.0),
                -0.12,
                0.36,
            )
        )
        beta_center = _round_to_step(beta_center, 0.01)
        beta_range = _beta_range_text(beta_center, width=0.04 if stress_score < 55 else 0.06)

        geo_multiplier = _round_to_step(float(np.clip(1.0 + (stress_score / 210.0), 1.0, 1.62)), 0.01)

        packet_conf = str(row.get("packet_confidence", "Medium"))
        publishable = bool(row.get("publishable_flag")) if pd.notna(row.get("publishable_flag")) else False
        confidence_penalty = _round_to_step(
            _confidence_penalty_score(
                proxy_share=proxy_share,
                quality_score=quality_score,
                packet_conf=packet_conf,
                publishable=publishable,
            ),
            0.01,
        )
        constraint_conf = _constraint_confidence(
            packet_conf=packet_conf,
            quality_score=quality_score,
            proxy_share=proxy_share,
            publishable=publishable,
        )
        risk_bucket = _bucket_label(stress_score)
        scenario_shift = _scenario_shift_text(
            stress_score=stress_score,
            regime_label=regime_label,
            route_label=route_label,
            event_torque=event_torque,
        )
        note = (
            f"Heuristic overlay (not a valuation engine): regime={regime_label}, route_label={route_label}, "
            f"route_concentration={route_concentration:.1f}, beta_stress={beta_stress:.1f}, event_torque={event_torque:.1f}, "
            f"quality_penalty={quality_penalty:.1f}, scenario_fcf_sensitivity={scenario_sens:.1f}%."
        )

        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "company_name": row["company_name"],
                "ticker": row["ticker"],
                "bucket_classification": row.get("bucket_classification", ""),
                "regime_label": regime_label,
                "heuristic_method_flag": True,
                "methodology_label": "heuristic_overlay_v1",
                "stress_score": round(stress_score, 2),
                "stress_driver_route": round(route_concentration, 2),
                "stress_driver_beta": round(beta_stress, 2),
                "stress_driver_event": round(event_torque, 2),
                "stress_driver_quality_penalty": round(quality_penalty, 2),
                "suggested_discount_rate_uplift_bps": int(round(discount_center)),
                "suggested_discount_rate_uplift_range_bps": discount_range,
                "suggested_beta_adjustment": round(beta_center, 2),
                "suggested_beta_adjustment_range": beta_range,
                "suggested_risk_premium_bucket": risk_bucket,
                "scenario_probability_shift": scenario_shift,
                "suggested_scenario_probability_shift": scenario_shift,
                "geopolitical_stress_multiplier": round(geo_multiplier, 2),
                "confidence_penalty_recommendation": round(confidence_penalty, 2),
                "constraint_confidence": constraint_conf,
                "market_regime_impact_note": note,
                "market_constraint_summary": (
                    f"{risk_bucket}; discount_uplift={int(round(discount_center))} bps "
                    f"(range {discount_range}); beta_adj={beta_center:.2f} ({beta_range}); "
                    f"scenario_shift={scenario_shift}; confidence={constraint_conf}."
                ),
            }
        )

    out = pd.DataFrame(rows).sort_values("stress_score", ascending=False).reset_index(drop=True)
    return out


def build_market_constraints_methodology(as_of_date: date) -> pd.DataFrame:
    rows = [
        {
            "as_of_date": as_of_date.isoformat(),
            "methodology_label": "heuristic_overlay_v1",
            "component": "route_concentration",
            "weight": 0.34,
            "description": "Combined route exposure and Hormuz-share concentration as transport-fragility input.",
        },
        {
            "as_of_date": as_of_date.isoformat(),
            "methodology_label": "heuristic_overlay_v1",
            "component": "qualitative_route_risk",
            "weight": 0.20,
            "description": "Route-risk label mapping (Low/Medium/High) to stress score uplift.",
        },
        {
            "as_of_date": as_of_date.isoformat(),
            "methodology_label": "heuristic_overlay_v1",
            "component": "beta_stress",
            "weight": 0.20,
            "description": "Oil and sector-vs-market beta profile as market transmission amplifier.",
        },
        {
            "as_of_date": as_of_date.isoformat(),
            "methodology_label": "heuristic_overlay_v1",
            "component": "event_torque",
            "weight": 0.16,
            "description": "Event-day hit-rate and asymmetry in abnormal returns around oil moves.",
        },
        {
            "as_of_date": as_of_date.isoformat(),
            "methodology_label": "heuristic_overlay_v1",
            "component": "quality_penalty",
            "weight": 0.10,
            "description": "Penalty from lower data-quality score and high proxy burden.",
        },
        {
            "as_of_date": as_of_date.isoformat(),
            "methodology_label": "heuristic_overlay_v1",
            "component": "regime_multiplier",
            "weight": "post-score",
            "description": "Regime label applies multiplicative tilt to stress score.",
        },
        {
            "as_of_date": as_of_date.isoformat(),
            "methodology_label": "heuristic_overlay_v1",
            "component": "output_policy",
            "weight": "n/a",
            "description": "Outputs are directional overlays for downstream valuation scenarios, not standalone valuation assumptions.",
        },
    ]
    return pd.DataFrame(rows)
