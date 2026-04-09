from datetime import date

import pandas as pd

from src.assumptions_registry import AssumptionsRegistry, MissingDataLogger
from src.fuel_prices import build_fuel_trackers
from src.source_logger import SourceLogger


def test_fuel_weighting_logic(monkeypatch):
    benchmark = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-09", "2026-01-16"]),
            "rbob_per_gal": [2.0, 2.1, 2.2],
            "heating_oil_per_gal": [2.5, 2.4, 2.6],
            "petrol_proxy_usd_per_bbl": [84.0, 88.2, 92.4],
            "diesel_proxy_usd_per_bbl": [105.0, 100.8, 109.2],
        }
    )
    monkeypatch.setattr("src.fuel_prices._proxy_fuel_benchmarks", lambda *args, **kwargs: benchmark)

    profiles_df = pd.DataFrame(
        [
            {
                "company_name": "TestCo",
                "ticker": "TST",
                "retail_country_weights": {"US": 0.6, "UK": 0.4},
                "source_links": ["https://example.com"],
                "profile_confidence": "Medium",
            }
        ]
    )

    crude_tracker = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-09", "2026-01-16"]),
            "brent_price": [80.0, 82.0, 84.0],
        }
    )

    fuel_tracker, weights = build_fuel_trackers(
        profiles_df=profiles_df,
        crude_tracker_df=crude_tracker,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        frequency="W-FRI",
        source_logger=SourceLogger(),
        assumptions_registry=AssumptionsRegistry(),
        missing_logger=MissingDataLogger(),
    )

    assert not fuel_tracker.empty
    assert set(weights["geography"]) == {"US", "UK"}
    assert "fuels_to_brent_ratio" in fuel_tracker.columns
    assert fuel_tracker["blended_combined_fuels_price"].iloc[0] > 0
