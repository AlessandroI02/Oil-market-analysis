from datetime import date

import pandas as pd

from src.fetch_diagnostics import FetchDiagnostics
from src.market_data_providers import fetch_with_provider_chain


def test_provider_chain_uses_fallback_after_primary_failure():
    diagnostics = FetchDiagnostics()

    def primary(_s, _e):
        return pd.DataFrame(columns=["date", "price"])

    def fallback(_s, _e):
        return pd.DataFrame({"date": pd.date_range("2026-01-01", periods=5, freq="D"), "price": [1, 2, 3, 4, 5]})

    frame, provider, _url, fallback_used = fetch_with_provider_chain(
        dataset="test_dataset",
        identifier="TEST",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 10),
        diagnostics=diagnostics,
        providers=[
            {"name": "primary", "source_url": "https://primary", "fetcher": primary},
            {"name": "fallback", "source_url": "https://fallback", "fetcher": fallback},
        ],
        min_points=2,
    )

    assert not frame.empty
    assert provider == "fallback"
    assert fallback_used is True

    attempts = diagnostics.attempts_df()
    usage = diagnostics.provider_usage_df()
    assert len(attempts) >= 2
    assert usage.iloc[-1]["selected_provider"] == "fallback"
