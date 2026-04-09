import pandas as pd

from src.run_health import assess_run_health


def test_run_health_invalid_when_core_backbone_missing():
    datasets = {
        "Crude_Tracker": pd.DataFrame({"date": ["2026-01-01"], "brent_price": [None], "wti_price": [None]}),
        "Fuel_Tracker": pd.DataFrame({"date": ["2026-01-01"], "blended_combined_fuels_price": [None]}),
        "Equity_Tracker": pd.DataFrame(columns=["date", "ticker", "share_price"]),
        "Valuation": pd.DataFrame(columns=["ticker", "market_cap_usd", "valuation_score"]),
        "Factor_Decomposition": pd.DataFrame(columns=["ticker", "beta_brent_ret"]),
        "Historical_Analogues": pd.DataFrame(columns=["period", "brent_return_pct"]),
        "Core_Ranking": pd.DataFrame({"ticker": ["A"], "score_default_share": [0.9], "final_score": [10]}),
    }

    summary, checks = assess_run_health(
        datasets=datasets,
        quality_cfg={
            "minimum_required_price_points": 2,
            "minimum_required_equity_names": 1,
            "minimum_required_valuation_coverage": 0.2,
            "minimum_required_factor_coverage": 0.2,
            "minimum_non_null_ratio_for_section": 0.2,
        },
    )

    assert summary["run_status"] == "INVALID"
    assert not checks.empty


def test_run_health_valid_with_populated_core_data():
    dates = pd.date_range("2026-01-01", periods=6, freq="W-FRI")
    datasets = {
        "Crude_Tracker": pd.DataFrame({"date": dates, "brent_price": [80, 81, 82, 83, 84, 85], "wti_price": [75, 76, 77, 78, 79, 80]}),
        "Fuel_Tracker": pd.DataFrame({"date": dates, "blended_petrol_price": [95, 96, 97, 98, 99, 100], "blended_diesel_price": [90, 91, 92, 93, 94, 95], "blended_combined_fuels_price": [92.5, 93.5, 94.5, 95.5, 96.5, 97.5]}),
        "Equity_Tracker": pd.DataFrame({"date": dates.tolist() * 2, "ticker": ["A"] * 6 + ["B"] * 6, "share_price": [1, 2, 3, 4, 5, 6, 2, 2, 2, 2, 2, 2]}),
        "Valuation": pd.DataFrame({"ticker": ["A", "B"], "market_cap_usd": [100, 200], "valuation_score": [50, 60]}),
        "Factor_Decomposition": pd.DataFrame({"ticker": ["A", "B"], "beta_brent_ret": [1.1, 0.9]}),
        "Historical_Analogues": pd.DataFrame({"period": ["p1", "p2"], "brent_return_pct": [10.0, 5.0]}),
        "Core_Ranking": pd.DataFrame({"ticker": ["A", "B"], "score_default_share": [0.2, 0.3], "final_score": [70, 65]}),
    }

    summary, _ = assess_run_health(
        datasets=datasets,
        quality_cfg={
            "minimum_required_price_points": 4,
            "minimum_required_equity_names": 2,
            "minimum_required_valuation_coverage": 0.5,
            "minimum_required_factor_coverage": 0.5,
            "minimum_non_null_ratio_for_section": 0.3,
        },
    )

    assert summary["run_status"] == "VALID"
