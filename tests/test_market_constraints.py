from datetime import date

import pandas as pd

from src.market_constraints import build_market_constraints


def test_market_constraints_output_shape():
    included = pd.DataFrame(
        [
            {"company_name": "A Co", "ticker": "A", "bucket_classification": "primary"},
            {"company_name": "B Co", "ticker": "B", "bucket_classification": "secondary"},
        ]
    )
    exposure = pd.DataFrame(
        [
            {"ticker": "A", "combined_exposure_pct": 40},
            {"ticker": "B", "combined_exposure_pct": 20},
        ]
    )
    route = pd.DataFrame(
        [
            {"ticker": "A", "qualitative_route_risk": "High", "hormuz_share_pct": 60},
            {"ticker": "B", "qualitative_route_risk": "Low", "hormuz_share_pct": 10},
        ]
    )
    factor = pd.DataFrame(
        [
            {"ticker": "A", "beta_brent_ret": 1.4, "beta_market_ret": 1.0, "beta_energy_ret": 1.2},
            {"ticker": "B", "beta_brent_ret": 0.9, "beta_market_ret": 0.8, "beta_energy_ret": 0.9},
        ]
    )
    scenario = pd.DataFrame(
        [
            {"ticker": "A", "scenario_brent": 100, "fcf_sensitivity_pct": 12},
            {"ticker": "B", "scenario_brent": 100, "fcf_sensitivity_pct": 4},
        ]
    )
    quality = pd.DataFrame(
        [
            {"ticker": "A", "data_quality_score": 60, "proxy_assumption_share": 0.3},
            {"ticker": "B", "data_quality_score": 80, "proxy_assumption_share": 0.1},
        ]
    )
    regime = pd.DataFrame([{"regime_label": "route-disruption regime", "regime_score_total": 72}])

    out = build_market_constraints(
        included_df=included,
        exposure_df=exposure,
        route_risks_df=route,
        factor_df=factor,
        scenario_df=scenario,
        data_quality_df=quality,
        regime_state_df=regime,
        as_of_date=date(2026, 4, 4),
    )

    assert len(out) == 2
    assert {"suggested_discount_rate_uplift_bps", "scenario_probability_shift"}.issubset(set(out.columns))
