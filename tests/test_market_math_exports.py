from datetime import date
from pathlib import Path

import pandas as pd

import src.market_math as market_math


def test_market_math_exports_populated(monkeypatch):
    dates = pd.date_range("2026-01-01", periods=70, freq="D")
    equity_price = pd.DataFrame(
        {
            "date": dates,
            "A": [100 + i * 0.3 for i in range(len(dates))],
            "B": [95 + i * 0.15 for i in range(len(dates))],
        }
    )
    benchmark_price = pd.DataFrame(
        {
            "date": dates,
            "brent_price": [80 + i * 0.2 for i in range(len(dates))],
            "xle_price": [70 + i * 0.1 for i in range(len(dates))],
            "spy_price": [500 + i * 0.05 for i in range(len(dates))],
            "rates_price": [4 + i * 0.001 for i in range(len(dates))],
            "dxy_price": [103 + i * 0.01 for i in range(len(dates))],
        }
    )

    def _fake_fetch_daily_prices(included_df, start_date, end_date, raw_market_dir):
        return equity_price, benchmark_price

    monkeypatch.setattr(market_math, "_fetch_daily_prices", _fake_fetch_daily_prices)

    out = market_math.build_market_math_exports(
        included_df=pd.DataFrame([{"ticker": "A"}, {"ticker": "B"}]),
        exposure_df=pd.DataFrame({"ticker": ["A", "B"], "combined_exposure_pct": [35, 20]}),
        route_risks_df=pd.DataFrame({"ticker": ["A", "B"], "qualitative_route_risk": ["High", "Low"]}),
        event_days_df=pd.DataFrame(),
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
        raw_market_dir=Path("data/raw/market"),
        interim_market_dir=Path("data/interim/market"),
    )

    assert not out["returns_daily_matrix"].empty
    assert not out["regression_coefficients"].empty
    assert not out["rolling_betas"].empty
    assert {"ticker", "event_day_hit_rate_up_oil"}.issubset(set(out["event_study_summary"].columns))
