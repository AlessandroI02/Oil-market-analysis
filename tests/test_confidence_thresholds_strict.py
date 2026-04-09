import pandas as pd

from src.confidence_framework import build_confidence_framework


def test_high_packet_requires_strict_multi_component_gate():
    included = pd.DataFrame(
        [{"company_name": "Strict Co", "ticker": "S", "bucket_classification": "primary", "confidence": "High"}]
    )
    data_quality = pd.DataFrame(
        [
            {
                "ticker": "S",
                "data_quality_score": 95,
                "proxy_assumption_share": 0.34,
                "missing_field_count": 0,
                "tier1_2_source_share": 0.20,
                "exact_evidence_share": 0.20,
            }
        ]
    )
    source_log = pd.DataFrame(
        [
            {
                "company": "Strict Co",
                "source_url": "https://example.com/article",
                "source_tier": "Tier 3",
                "evidence_flag": "estimated",
            }
        ]
    )
    route = pd.DataFrame([{"ticker": "S", "qualitative_route_risk": "Low", "hormuz_share_pct": 2}])
    event = pd.DataFrame([{"ticker": "S", "event_count": 10, "event_day_hit_rate_up_oil": 0.8, "event_day_hit_rate_down_oil": 0.7}])
    regime = pd.DataFrame([{"regime_confidence": "High"}])
    ranking_health = pd.DataFrame(
        [
            {
                "ticker": "S",
                "rating_status": "rated",
                "publishable_flag": True,
                "final_rating_confidence": "High",
                "score_real_data_share": 0.7,
                "score_proxy_share": 0.42,
                "score_default_share": 0.05,
                "ranking_health_status": "GOOD",
            }
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
    row = out.iloc[0]
    assert row["packet_confidence"] != "High"
    assert bool(row["packet_high_eligibility"]) is False
    assert "low_proxy_burden" in str(row["packet_high_eligibility_reasons"])
