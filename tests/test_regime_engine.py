from datetime import date

import pandas as pd

from src.regime_engine import build_regime_state


def test_regime_state_outputs_label_and_history():
    benchmark = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=30, freq="D"),
            "brent_ret": [0.025 if i % 3 == 0 else 0.01 for i in range(30)],
            "xle_ret": [0.012] * 30,
            "spy_ret": [-0.001] * 30,
            "rates_ret": [0.0005] * 30,
            "dxy_ret": [0.0008] * 30,
        }
    )
    crude = pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=4, freq="W-FRI"), "brent_price": [82, 84, 87, 90]}
    )
    route = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "qualitative_route_risk": ["High", "High", "Medium"],
        }
    )
    episodes = pd.DataFrame(
        {
            "episode_id": ["EP001", "EP002"],
            "start_date": ["2026-01-05", "2026-01-12"],
            "end_date": ["2026-01-08", "2026-01-14"],
            "episode_type": ["Hormuz disruption", "Sanctions"],
            "mean_abs_move": [2.7, 2.1],
            "peak_move": [3.8, 3.2],
        }
    )

    state, history = build_regime_state(
        crude_tracker_df=crude,
        benchmark_returns_df=benchmark,
        route_risks_df=route,
        event_episodes_df=episodes,
        as_of_date=date(2026, 1, 30),
    )

    assert not state.empty
    assert not history.empty
    assert state.iloc[0]["regime_label"] in {
        "route-disruption regime",
        "geopolitical supply-shock regime",
        "sanctions-driven regime",
        "stressed oil regime",
    }
