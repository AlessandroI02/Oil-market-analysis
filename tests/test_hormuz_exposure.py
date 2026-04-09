import pandas as pd

from src.assumptions_registry import AssumptionsRegistry
from src.hormuz_exposure import estimate_hormuz_exposure


def test_hormuz_exposure_structure():
    profiles = pd.DataFrame(
        [
            {
                "company_name": "A",
                "ticker": "AAA",
                "production_region_weights": {"Middle East": 0.6, "North America": 0.4},
                "refinery_region_weights": {"Europe": 0.7, "Middle East": 0.3},
                "international_sales_ratio": 0.7,
                "upstream_mix_share": 0.6,
                "downstream_mix_share": 0.4,
                "profile_confidence": "Medium",
                "source_links": ["https://example.com"],
            },
            {
                "company_name": "B",
                "ticker": "BBB",
                "production_region_weights": {"North America": 1.0},
                "refinery_region_weights": {"North America": 1.0},
                "international_sales_ratio": 0.3,
                "upstream_mix_share": 0.5,
                "downstream_mix_share": 0.5,
                "profile_confidence": "High",
                "source_links": ["https://example.com"],
            },
        ]
    )

    registry = AssumptionsRegistry()
    out = estimate_hormuz_exposure(profiles, registry)

    required_cols = {
        "company_name",
        "ticker",
        "crude_exposure_pct",
        "refined_exposure_pct",
        "combined_exposure_pct",
        "physical_exposure_pct",
        "economic_exposure_pct",
        "earnings_exposure_pct",
        "exposure_low_pct",
        "exposure_high_pct",
        "confidence",
        "methodology_note",
        "exact_vs_estimated",
        "ranking",
    }
    assert required_cols.issubset(set(out.columns))
    assert (out["combined_exposure_pct"] >= 0).all()
    assert (out["combined_exposure_pct"] <= 100).all()
    assert (out["exposure_low_pct"] <= out["combined_exposure_pct"]).all()
    assert (out["exposure_high_pct"] >= out["combined_exposure_pct"]).all()
    assert set(out["ranking"]) == {1, 2}

    assumptions_df = registry.to_dataframe()
    assert not assumptions_df.empty
