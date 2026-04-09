import pandas as pd

from src.oil_news_report import _build_event_table, _theme_from_text


def test_oil_news_event_day_detection_threshold():
    brent = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "price": [80.0, 81.8, 78.0, 78.5, 80.2],
        }
    )
    wti = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "price": [75.0, 76.0, 74.0, 74.5, 75.5],
        }
    )

    events = _build_event_table(brent, wti, threshold_pct=2.0)
    assert not events.empty
    assert (events["brent_daily_move_pct"].abs() >= 2.0).all()


def test_theme_classification_from_headline():
    assert _theme_from_text("OPEC agrees deeper production cuts") == "OPEC / supply"
    assert _theme_from_text("Drone attack disrupts shipping near Hormuz") in {"Hormuz disruption", "Geopolitical attack"}
