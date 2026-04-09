import pandas as pd

from src.ranking_framework import build_rankings


def test_publishable_gate_is_conservative_for_high_route_and_model_burden():
    included = pd.DataFrame(
        [{"company_name": "A", "ticker": "A", "bucket_classification": "primary", "confidence": "High"}]
    )
    archetypes = pd.DataFrame([{"ticker": "A", "archetype": "Integrated supermajor", "inclusion_confidence": "High"}])
    exposure = pd.DataFrame(
        [
            {
                "ticker": "A",
                "combined_exposure_pct": 30.0,
                "confidence": "High",
                "exposure_low_pct": 20.0,
                "exposure_high_pct": 40.0,
                "exact_vs_estimated": "estimated",
            }
        ]
    )
    route = pd.DataFrame([{"ticker": "A", "qualitative_route_risk": "High"}])
    valuation = pd.DataFrame([{"ticker": "A", "valuation_score": 75, "leverage_ratio": 1.0, "data_flag": "exact"}])
    scenario = pd.DataFrame([{"ticker": "A", "scenario_brent": 100, "fcf_sensitivity_pct": 10, "ebitda_sensitivity_pct": 8, "payout_resilience": "Strong"}])
    event_path = pd.DataFrame([{"ticker": "A", "event_path": "Hormuz disrupted through earnings cycle", "scenario_impact_score": 8}])
    factor = pd.DataFrame([{"ticker": "A", "beta_brent_ret": 1.1, "beta_market_ret": 0.9, "beta_energy_ret": 1.0, "idiosyncratic_residual_pct": 0.2, "r2": 0.6}])
    catalyst = pd.DataFrame(
        [
            {
                "ticker": "A",
                "event_date": "2026-04-20",
                "event": "Earnings",
                "near_term_event_support": "Strong",
                "market_relevance_score": 84,
                "catalyst_confidence": "High",
            }
        ]
    )
    data_quality = pd.DataFrame(
        [
            {
                "ticker": "A",
                "data_quality_score": 90,
                "proxy_assumption_share": 0.35,
                "missing_field_count": 0,
                "tier1_2_source_share": 0.5,
                "exact_evidence_share": 0.35,
            }
        ]
    )

    results = build_rankings(
        included_df=included,
        archetypes_df=archetypes,
        exposure_df=exposure,
        route_risks_df=route,
        valuation_df=valuation,
        scenario_df=scenario,
        event_path_df=event_path,
        factor_df=factor,
        catalyst_df=catalyst,
        data_quality_df=data_quality,
        weight_config={
            "route_exposure": 0.20,
            "oil_sensitivity": 0.12,
            "downstream_pass_through": 0.08,
            "valuation": 0.18,
            "balance_sheet": 0.12,
            "catalyst": 0.10,
            "market_positioning": 0.10,
            "scenario_resilience": 0.10,
        },
        confidence_penalty_map={"High": 0.00, "Medium": 0.05, "Low": 0.12},
    )

    row = results["extended_ranking"].iloc[0]
    assert bool(row["publishable_flag"]) is False
    assert str(row["rating_status"]).lower() in {"provisional", "unrated"}
