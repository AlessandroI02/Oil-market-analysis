from datetime import date

import pandas as pd

from src.catalyst_calendar import split_catalyst_calendar


def test_catalyst_split_prioritizes_relevant_future_events():
    catalyst_df = pd.DataFrame(
        [
            {"ticker": "A", "event": "Earnings", "event_date": "2026-04-20", "event_type": "company", "confidence": "exact", "event_congestion": "Low", "notes": ""},
            {"ticker": "A", "event": "Following Earnings", "event_date": "2026-08-01", "event_type": "company", "confidence": "estimated", "event_congestion": "Low", "notes": "cadence estimated"},
            {"ticker": "A", "event": "Ex-Dividend", "event_date": "2026-04-18", "event_type": "capital_return", "confidence": "exact", "event_congestion": "Low", "notes": ""},
            {"ticker": "GLOBAL", "event": "OPEC+ Meeting", "event_date": "2026-05-06", "event_type": "macro", "confidence": "estimated", "event_congestion": "Medium", "notes": ""},
            {"ticker": "B", "event": "Past Event", "event_date": "2026-04-01", "event_type": "company", "confidence": "exact", "event_congestion": "Low", "notes": ""},
        ]
    )

    primary, archive = split_catalyst_calendar(
        catalyst_df=catalyst_df,
        as_of_date=date(2026, 4, 10),
        horizon_days=180,
    )

    assert {"market_relevance_score", "market_relevance_tier", "catalyst_confidence", "priority_rank"}.issubset(primary.columns)
    assert "Earnings" in set(primary["event"].tolist())
    assert "OPEC+ Meeting" in set(primary["event"].tolist())
    assert len(primary) >= 2
    assert (primary["ticker"] != "GLOBAL").any()
    assert "Past Event" not in set(primary["event"].tolist())
    assert "Following Earnings" in set(archive["event"].tolist()) or "Following Earnings" in set(primary["event"].tolist())
