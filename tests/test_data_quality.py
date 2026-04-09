import pandas as pd

from src.data_quality import build_data_quality_table


def test_data_quality_penalizes_proxy_and_missing():
    included = pd.DataFrame(
        [
            {"company_name": "Good Co", "ticker": "GOOD"},
            {"company_name": "Weak Co", "ticker": "WEAK"},
        ]
    )
    assumptions = pd.DataFrame(
        [
            {"company": "Good Co", "estimate_type": "analyst_estimate"},
            {"company": "Good Co", "estimate_type": "disclosed_exact"},
            {"company": "Weak Co", "estimate_type": "proxy_estimate"},
            {"company": "Weak Co", "estimate_type": "proxy_estimate"},
            {"company": "Weak Co", "estimate_type": "proxy_estimate"},
        ]
    )
    missing = pd.DataFrame(
        [
            {"company": "Weak Co", "field_name": "x", "severity": "high"},
            {"company": "Weak Co", "field_name": "y", "severity": "high"},
            {"company": "Weak Co", "field_name": "z", "severity": "medium"},
        ]
    )
    source = pd.DataFrame(
        [
            {"company": "Good Co", "source_tier": "Tier 1", "evidence_flag": "exact"},
            {"company": "Good Co", "source_tier": "Tier 2", "evidence_flag": "exact"},
            {"company": "Weak Co", "source_tier": "Tier 4", "evidence_flag": "estimated"},
        ]
    )

    out = build_data_quality_table(
        included_df=included,
        assumptions_df=assumptions,
        missing_df=missing,
        source_log_df=source,
    )
    assert not out.empty
    score_good = float(out.loc[out["ticker"] == "GOOD", "data_quality_score"].iloc[0])
    score_weak = float(out.loc[out["ticker"] == "WEAK", "data_quality_score"].iloc[0])
    assert score_good > score_weak
