import pandas as pd

from src.health_reporting import build_section_health


def test_route_exposure_section_uses_hormuz_ranking_core_field():
    datasets = {
        "Hormuz_Ranking": pd.DataFrame({"ticker": ["A", "B"], "combined_exposure_pct": [30.0, 20.0]}),
        "Route_Exposure_Build": pd.DataFrame({"ticker": ["A"], "build_component": ["combined_exposure_pct"], "value": [30.0]}),
        "Route_Risks": pd.DataFrame({"ticker": ["A", "B"], "hormuz_share_pct": [40.0, 12.0], "qualitative_route_risk": ["High", "Low"]}),
        "Crude_Tracker": pd.DataFrame({"brent_price": [90.0], "wti_price": [85.0]}),
        "Fuel_Tracker": pd.DataFrame({"blended_combined_fuels_price": [120.0]}),
        "Equity_Tracker": pd.DataFrame({"share_price": [100.0]}),
        "Valuation": pd.DataFrame({"market_cap_usd": [1], "valuation_score": [50]}),
        "Factor_Decomposition": pd.DataFrame({"beta_brent_ret": [1.0]}),
        "Historical_Analogues": pd.DataFrame({"brent_return_pct": [8.0]}),
        "Catalyst_Calendar": pd.DataFrame({"event_date": ["2026-04-04"]}),
        "Recommendation_Framework": pd.DataFrame({"final_score": [60.0]}),
    }
    out = build_section_health(
        datasets=datasets,
        run_summary={"run_status": "VALID"},
        min_non_null_ratio=0.25,
    )
    route_row = out[out["section_name"] == "route_exposure"].iloc[0]
    assert route_row["section_status"] == "VALID"
