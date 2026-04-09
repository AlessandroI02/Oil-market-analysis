from datetime import date

import pandas as pd

from src.market_constraints import build_market_constraints, build_market_constraints_methodology


def test_market_constraints_are_differentiated_and_bucketed():
    included = pd.DataFrame(
        [
            {"company_name": "A", "ticker": "A", "bucket_classification": "primary"},
            {"company_name": "B", "ticker": "B", "bucket_classification": "primary"},
            {"company_name": "C", "ticker": "C", "bucket_classification": "secondary"},
        ]
    )
    exposure = pd.DataFrame(
        [
            {"ticker": "A", "combined_exposure_pct": 55},
            {"ticker": "B", "combined_exposure_pct": 20},
            {"ticker": "C", "combined_exposure_pct": 35},
        ]
    )
    route = pd.DataFrame(
        [
            {"ticker": "A", "qualitative_route_risk": "High", "hormuz_share_pct": 70},
            {"ticker": "B", "qualitative_route_risk": "Low", "hormuz_share_pct": 8},
            {"ticker": "C", "qualitative_route_risk": "Medium", "hormuz_share_pct": 20},
        ]
    )
    factor = pd.DataFrame(
        [
            {"ticker": "A", "beta_brent_ret": 1.4, "beta_market_ret": 0.9, "beta_energy_ret": 1.3},
            {"ticker": "B", "beta_brent_ret": 0.8, "beta_market_ret": 1.0, "beta_energy_ret": 0.9},
            {"ticker": "C", "beta_brent_ret": 1.1, "beta_market_ret": 1.0, "beta_energy_ret": 1.1},
        ]
    )
    scenario = pd.DataFrame(
        [
            {"ticker": "A", "scenario_brent": 100, "fcf_sensitivity_pct": 18},
            {"ticker": "B", "scenario_brent": 100, "fcf_sensitivity_pct": 4},
            {"ticker": "C", "scenario_brent": 100, "fcf_sensitivity_pct": 10},
        ]
    )
    quality = pd.DataFrame(
        [
            {"ticker": "A", "data_quality_score": 58, "proxy_assumption_share": 0.35},
            {"ticker": "B", "data_quality_score": 88, "proxy_assumption_share": 0.10},
            {"ticker": "C", "data_quality_score": 70, "proxy_assumption_share": 0.22},
        ]
    )
    regime = pd.DataFrame([{"regime_label": "route-disruption regime", "regime_score_total": 74}])
    confidence = pd.DataFrame(
        [
            {"ticker": "A", "packet_confidence": "Medium", "publishable_flag": False},
            {"ticker": "B", "packet_confidence": "High", "publishable_flag": True},
            {"ticker": "C", "packet_confidence": "Medium", "publishable_flag": True},
        ]
    )
    event = pd.DataFrame(
        [
            {"ticker": "A", "event_day_hit_rate_up_oil": 0.85, "event_day_hit_rate_down_oil": 0.20, "avg_abnormal_return_oil_up": 0.03, "avg_abnormal_return_oil_down": -0.02},
            {"ticker": "B", "event_day_hit_rate_up_oil": 0.40, "event_day_hit_rate_down_oil": 0.55, "avg_abnormal_return_oil_up": 0.00, "avg_abnormal_return_oil_down": -0.01},
            {"ticker": "C", "event_day_hit_rate_up_oil": 0.55, "event_day_hit_rate_down_oil": 0.50, "avg_abnormal_return_oil_up": 0.01, "avg_abnormal_return_oil_down": -0.005},
        ]
    )

    out = build_market_constraints(
        included_df=included,
        exposure_df=exposure,
        route_risks_df=route,
        factor_df=factor,
        scenario_df=scenario,
        data_quality_df=quality,
        regime_state_df=regime,
        as_of_date=date(2026, 4, 5),
        confidence_framework_df=confidence,
        event_study_summary_df=event,
    )

    assert {"suggested_discount_rate_uplift_range_bps", "constraint_confidence", "methodology_label"}.issubset(out.columns)
    assert out["methodology_label"].nunique() == 1
    assert out["methodology_label"].iloc[0] == "heuristic_overlay_v1"
    assert out["suggested_discount_rate_uplift_bps"].nunique() >= 2
    assert out["scenario_probability_shift"].nunique() >= 2

    methodology = build_market_constraints_methodology(as_of_date=date(2026, 4, 5))
    assert "output_policy" in set(methodology["component"].tolist())
