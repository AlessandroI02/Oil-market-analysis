from datetime import date
from pathlib import Path

import pandas as pd

from src.company_case_packets import build_company_case_packets


def test_company_case_packet_generation_writes_json():
    included = pd.DataFrame(
        [{"company_name": "A Co", "ticker": "A", "bucket_classification": "primary", "confidence": "High"}]
    )
    archetypes = pd.DataFrame(
        [{"ticker": "A", "archetype": "Integrated supermajor", "inclusion_confidence": "High"}]
    )
    exposure = pd.DataFrame(
        [{"ticker": "A", "combined_exposure_pct": 35, "exposure_low_pct": 27, "exposure_high_pct": 43}]
    )
    route = pd.DataFrame(
        [
            {
                "ticker": "A",
                "hormuz_share_pct": 45,
                "bab_el_mandeb_share_pct": 20,
                "suez_share_pct": 10,
                "non_chokepoint_share_pct": 25,
                "qualitative_route_risk": "High",
                "rerouting_flexibility": "Medium",
                "pipeline_bypass_optionality": "Low",
                "disruption_notes": "note",
            }
        ]
    )
    crude = pd.DataFrame({"date": ["2026-04-01"], "brent_price": [90], "wti_price": [85]})
    fuel = pd.DataFrame({"date": ["2026-04-01"], "ticker": ["A"], "blended_combined_fuels_price": [118]})
    equity = pd.DataFrame({"date": ["2026-03-21", "2026-03-28", "2026-04-04"], "ticker": ["A", "A", "A"], "share_price": [100, 102, 104]})
    factor = pd.DataFrame([{"ticker": "A", "beta_brent_ret": 1.2, "beta_market_ret": 1.0, "beta_energy_ret": 1.1}])
    regress = pd.DataFrame([{"ticker": "A", "coef_brent_ret": 1.2, "coef_xle_ret": 1.1, "coef_spy_ret": 1.0, "coef_rates_ret": -0.1, "coef_dxy_ret": 0.2}])
    rolling = pd.DataFrame([{"date": "2026-04-04", "ticker": "A", "rolling_oil_beta_20d": 1.25, "rolling_oil_beta_60d": 1.2, "rolling_oil_beta_90d": 1.15}])
    event_summary = pd.DataFrame([{"ticker": "A", "event_day_hit_rate_up_oil": 0.6, "event_day_hit_rate_down_oil": 0.4, "avg_abnormal_return_oil_up": 0.01, "avg_abnormal_return_oil_down": -0.02}])
    analogue = pd.DataFrame([{"period": "Test", "brent_return_pct": 12.0, "peer_median_return_pct": 8.0}])
    catalysts = pd.DataFrame([{"ticker": "A", "event_date": "2026-05-01", "event": "Earnings"}])
    constraints = pd.DataFrame([{"ticker": "A", "suggested_discount_rate_uplift_bps": 120, "suggested_risk_premium_bucket": "Moderate", "scenario_probability_shift": "Shift +5pp", "market_regime_impact_note": "note"}])
    confidence = pd.DataFrame([{"ticker": "A", "data_confidence": "High", "source_confidence": "Medium", "route_confidence": "Medium", "event_confidence": "Medium", "regime_confidence": "Medium", "packet_confidence": "Medium", "company_packet_confidence": "Medium", "source_summary": "reuters.com:2"}])
    regime = pd.DataFrame([{"regime_label": "normal", "regime_summary": "ok"}])
    source_log = pd.DataFrame([{"company": "A Co", "source_url": "https://reuters.com/a"}])

    root_dir = Path("outputs/debug/pytest_tmp_packets")
    out_dir = root_dir / "company_case_packets"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_df = build_company_case_packets(
        included_df=included,
        archetypes_df=archetypes,
        exposure_df=exposure,
        route_risks_df=route,
        crude_tracker_df=crude,
        fuel_tracker_df=fuel,
        equity_tracker_df=equity,
        factor_df=factor,
        regression_coefficients_df=regress,
        rolling_betas_df=rolling,
        event_study_summary_df=event_summary,
        historical_analogues_df=analogue,
        catalyst_primary_df=catalysts,
        market_constraints_df=constraints,
        confidence_framework_df=confidence,
        regime_state_df=regime,
        source_log_df=source_log,
        output_dir=out_dir,
        as_of_date=date(2026, 4, 4),
        root_dir=root_dir,
    )

    assert not index_df.empty
    packet_rel = Path(index_df.iloc[0]["packet_path"])
    assert (root_dir / packet_rel).exists()
