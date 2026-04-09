import pandas as pd

from src.confidence_framework import build_confidence_audit, build_confidence_framework


def test_confidence_alignment_caps_unrated_names():
    included = pd.DataFrame(
        [
            {"company_name": "A Co", "ticker": "A", "bucket_classification": "primary", "confidence": "High"},
            {"company_name": "B Co", "ticker": "B", "bucket_classification": "primary", "confidence": "High"},
        ]
    )
    data_quality = pd.DataFrame(
        [
            {"ticker": "A", "data_quality_score": 92},
            {"ticker": "B", "data_quality_score": 90},
        ]
    )
    source_log = pd.DataFrame(
        [
            {"company": "A Co", "source_url": "https://reuters.com/a", "source_tier": "Tier 1", "evidence_flag": "exact"},
            {"company": "B Co", "source_url": "https://reuters.com/b", "source_tier": "Tier 1", "evidence_flag": "exact"},
        ]
    )
    route = pd.DataFrame(
        [
            {"ticker": "A", "qualitative_route_risk": "Low", "hormuz_share_pct": 10},
            {"ticker": "B", "qualitative_route_risk": "Low", "hormuz_share_pct": 10},
        ]
    )
    event = pd.DataFrame(
        [
            {"ticker": "A", "event_count": 8, "event_day_hit_rate_up_oil": 0.7, "event_day_hit_rate_down_oil": 0.6},
            {"ticker": "B", "event_count": 8, "event_day_hit_rate_up_oil": 0.7, "event_day_hit_rate_down_oil": 0.6},
        ]
    )
    regime = pd.DataFrame([{"regime_confidence": "High"}])
    ranking_health = pd.DataFrame(
        [
            {
                "ticker": "A",
                "rating_status": "unrated",
                "publishable_flag": False,
                "final_rating_confidence": "Low",
                "score_real_data_share": 0.7,
                "score_proxy_share": 0.2,
                "score_default_share": 0.0,
                "ranking_health_status": "UNRATED",
            },
            {
                "ticker": "B",
                "rating_status": "rated",
                "publishable_flag": True,
                "final_rating_confidence": "Medium",
                "score_real_data_share": 0.75,
                "score_proxy_share": 0.2,
                "score_default_share": 0.0,
                "ranking_health_status": "GOOD",
            },
        ]
    )

    out = build_confidence_framework(
        included_df=included,
        data_quality_df=data_quality,
        source_log_df=source_log,
        route_risks_df=route,
        event_study_summary_df=event,
        regime_state_df=regime,
        ranking_health_df=ranking_health,
    )
    by_ticker = out.set_index("ticker")

    assert by_ticker.loc["A", "packet_confidence"] == "Low"
    assert bool(by_ticker.loc["A", "downstream_readiness_flag"]) is False
    assert by_ticker.loc["B", "packet_confidence"] in {"Medium", "High"}
    assert by_ticker.loc["B", "confidence_contradiction_count"] == 0

    audit = build_confidence_audit(out, ranking_health)
    assert audit["confidence_contradiction_flag"].sum() == 0
