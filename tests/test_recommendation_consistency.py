import pandas as pd

from src.ranking_framework import build_rankings


def test_no_publishable_attractive_expression_when_unrated():
    included = pd.DataFrame(
        [
            {"company_name": "A", "ticker": "A", "bucket_classification": "primary", "confidence": "Low"},
            {"company_name": "B", "ticker": "B", "bucket_classification": "primary", "confidence": "Low"},
        ]
    )
    archetypes = pd.DataFrame(
        [
            {"ticker": "A", "archetype": "Integrated regional major", "inclusion_confidence": "Low"},
            {"ticker": "B", "archetype": "Integrated regional major", "inclusion_confidence": "Low"},
        ]
    )
    exposure = pd.DataFrame(
        [
            {"ticker": "A", "combined_exposure_pct": None, "confidence": "Low", "exposure_low_pct": None, "exposure_high_pct": None},
            {"ticker": "B", "combined_exposure_pct": None, "confidence": "Low", "exposure_low_pct": None, "exposure_high_pct": None},
        ]
    )

    results = build_rankings(
        included_df=included,
        archetypes_df=archetypes,
        exposure_df=exposure,
        route_risks_df=pd.DataFrame(),
        valuation_df=pd.DataFrame(),
        scenario_df=pd.DataFrame(),
        event_path_df=pd.DataFrame(),
        factor_df=pd.DataFrame(),
        catalyst_df=pd.DataFrame(),
        data_quality_df=pd.DataFrame(),
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

    rec = results["recommendation_framework"]
    assert "Market-side attractive expression" not in set(rec.get("category", pd.Series(dtype=str)).tolist())
