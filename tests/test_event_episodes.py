import pandas as pd

from src.event_episodes import cluster_event_episodes


def test_event_episode_clustering_groups_consecutive_days():
    events = pd.DataFrame(
        {
            "date": ["2026-03-01", "2026-03-02", "2026-03-06"],
            "brent_daily_move_pct": [2.4, -2.1, 2.8],
            "wti_daily_move_pct": [2.0, -1.9, 2.4],
            "direction": ["up", "down", "up"],
            "candidate_catalyst": ["Hormuz disruption", "Hormuz disruption", "OPEC / supply"],
            "attribution_confidence": ["High", "Medium", "Medium"],
        }
    )
    articles = pd.DataFrame(
        {
            "event_date": ["2026-03-01", "2026-03-02", "2026-03-06"],
            "headline": ["A", "B", "C"],
            "domain": ["reuters.com", "reuters.com", "bloomberg.com"],
            "relevance_score": [5.0, 4.8, 4.2],
        }
    )

    episodes, episode_articles = cluster_event_episodes(events, articles, gap_days=2)

    assert len(episodes) == 2
    assert set(episode_articles["episode_id"]) >= {"EP001", "EP002"}
