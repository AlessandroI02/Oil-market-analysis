import pandas as pd

from src.analyst_overrides import OverridePackage, apply_universe_overrides


def test_universe_override_exclude():
    universe = pd.DataFrame(
        [
            {
                "company_name": "Test Name",
                "ticker": "TST",
                "bucket_classification": "primary",
                "reason_excluded": "",
                "confidence": "High",
            }
        ]
    )
    overrides = OverridePackage(
        raw={},
        companies={
            "TST": {
                "include_flag": False,
                "analyst_notes": "Do not include while liquidity review pending.",
            }
        },
    )

    out = apply_universe_overrides(universe, overrides)
    assert out.loc[0, "bucket_classification"] == "rejected"
    assert "Analyst override exclude" in out.loc[0, "reason_excluded"]
    assert bool(out.loc[0, "override_applied"]) is True
