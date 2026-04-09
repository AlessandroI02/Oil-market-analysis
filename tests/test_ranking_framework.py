import pandas as pd

from src.ranking_framework import build_rankings


def test_core_and_extended_ranking_separation():
    included = pd.DataFrame(
        [
            {"company_name": "Core A", "ticker": "A", "bucket_classification": "primary", "confidence": "High"},
            {"company_name": "Core B", "ticker": "B", "bucket_classification": "primary", "confidence": "Low"},
            {"company_name": "Sec C", "ticker": "C", "bucket_classification": "secondary", "confidence": "Medium"},
        ]
    )
    archetypes = pd.DataFrame(
        [
            {"ticker": "A", "archetype": "Integrated supermajor", "inclusion_confidence": "High"},
            {"ticker": "B", "archetype": "Integrated regional major", "inclusion_confidence": "Low"},
            {"ticker": "C", "archetype": "Secondary expression", "inclusion_confidence": "Medium"},
        ]
    )
    exposure = pd.DataFrame(
        [
            {"ticker": "A", "combined_exposure_pct": 20.0, "confidence": "High", "exposure_low_pct": 16.0, "exposure_high_pct": 24.0},
            {"ticker": "B", "combined_exposure_pct": 25.0, "confidence": "Low", "exposure_low_pct": 10.0, "exposure_high_pct": 38.0},
            {"ticker": "C", "combined_exposure_pct": 15.0, "confidence": "Medium", "exposure_low_pct": 8.0, "exposure_high_pct": 22.0},
        ]
    )
    route = pd.DataFrame(
        [
            {"ticker": "A", "qualitative_route_risk": "Medium"},
            {"ticker": "B", "qualitative_route_risk": "High"},
            {"ticker": "C", "qualitative_route_risk": "Low"},
        ]
    )
    valuation = pd.DataFrame(
        [
            {"ticker": "A", "valuation_score": 70, "leverage_ratio": 1.2, "fcf_yield_pct": 8.0, "dividend_yield_pct": 4.0},
            {"ticker": "B", "valuation_score": 70, "leverage_ratio": 1.2, "fcf_yield_pct": 8.0, "dividend_yield_pct": 4.0},
            {"ticker": "C", "valuation_score": 70, "leverage_ratio": 1.2, "fcf_yield_pct": 8.0, "dividend_yield_pct": 4.0},
        ]
    )
    scenario = pd.DataFrame(
        [
            {"ticker": "A", "scenario_brent": 100, "ebitda_sensitivity_pct": 10, "fcf_sensitivity_pct": 8, "payout_resilience": "Strong"},
            {"ticker": "B", "scenario_brent": 100, "ebitda_sensitivity_pct": 10, "fcf_sensitivity_pct": 8, "payout_resilience": "Strong"},
            {"ticker": "C", "scenario_brent": 100, "ebitda_sensitivity_pct": 10, "fcf_sensitivity_pct": 8, "payout_resilience": "Strong"},
        ]
    )
    event_path = pd.DataFrame(
        [
            {"ticker": "A", "event_path": "Hormuz disrupted through earnings cycle", "scenario_impact_score": 10},
            {"ticker": "B", "event_path": "Hormuz disrupted through earnings cycle", "scenario_impact_score": 10},
            {"ticker": "C", "event_path": "Hormuz disrupted through earnings cycle", "scenario_impact_score": 10},
        ]
    )
    factor = pd.DataFrame(
        [
            {"ticker": "A", "beta_brent_ret": 1.0, "beta_market_ret": 0.8, "beta_energy_ret": 1.1, "idiosyncratic_residual_pct": 0.2, "r2": 0.5},
            {"ticker": "B", "beta_brent_ret": 1.0, "beta_market_ret": 0.8, "beta_energy_ret": 1.1, "idiosyncratic_residual_pct": 0.2, "r2": 0.5},
            {"ticker": "C", "beta_brent_ret": 1.0, "beta_market_ret": 0.8, "beta_energy_ret": 1.1, "idiosyncratic_residual_pct": 0.2, "r2": 0.5},
        ]
    )
    catalyst = pd.DataFrame(
        [
            {"ticker": "A", "event_date": "2026-04-10", "near_term_event_support": "Strong", "event": "Earnings"},
            {"ticker": "B", "event_date": "2026-04-11", "near_term_event_support": "Strong", "event": "Earnings"},
            {"ticker": "C", "event_date": "2026-04-12", "near_term_event_support": "Strong", "event": "Earnings"},
        ]
    )
    data_quality = pd.DataFrame(
        [
            {"ticker": "A", "data_quality_score": 85, "proxy_assumption_share": 0.2, "missing_field_count": 1},
            {"ticker": "B", "data_quality_score": 40, "proxy_assumption_share": 0.8, "missing_field_count": 6},
            {"ticker": "C", "data_quality_score": 70, "proxy_assumption_share": 0.3, "missing_field_count": 2},
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

    core = results["core_ranking"]
    ext = results["extended_ranking"]
    assert not core.empty
    assert not ext.empty
    assert set(core["bucket_classification"]) == {"primary"}
    assert set(ext["bucket_classification"]) == {"primary", "secondary"}
    assert "core_rank" in core.columns
    assert "extended_rank" in ext.columns

    # Lower confidence should apply stronger penalty holding other components broadly similar.
    score_a = float(ext.loc[ext["ticker"] == "A", "final_score"].iloc[0])
    score_b = float(ext.loc[ext["ticker"] == "B", "final_score"].iloc[0])
    assert score_a >= score_b
