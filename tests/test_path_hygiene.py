from datetime import date
from pathlib import Path

import pandas as pd

from src.company_case_packets import build_company_case_packets


def test_company_packet_index_uses_relative_paths_when_root_provided():
    tmp_path = Path("outputs/debug/pytest_tmp_path_hygiene")
    tmp_path.mkdir(parents=True, exist_ok=True)

    included = pd.DataFrame([{"company_name": "A Co", "ticker": "A", "bucket_classification": "primary", "confidence": "Medium"}])
    archetypes = pd.DataFrame([{"ticker": "A", "archetype": "Integrated regional major", "inclusion_confidence": "Medium"}])
    exposure = pd.DataFrame([{"ticker": "A", "combined_exposure_pct": 20, "exposure_low_pct": 15, "exposure_high_pct": 25}])
    route = pd.DataFrame([{"ticker": "A", "qualitative_route_risk": "Low", "hormuz_share_pct": 10}])
    crude = pd.DataFrame({"date": ["2026-04-05"], "brent_price": [90], "wti_price": [86]})
    fuel = pd.DataFrame({"date": ["2026-04-05"], "ticker": ["A"], "blended_combined_fuels_price": [110]})
    equity = pd.DataFrame({"date": ["2026-04-05"], "ticker": ["A"], "share_price": [50]})

    index_df = build_company_case_packets(
        included_df=included,
        archetypes_df=archetypes,
        exposure_df=exposure,
        route_risks_df=route,
        crude_tracker_df=crude,
        fuel_tracker_df=fuel,
        equity_tracker_df=equity,
        factor_df=pd.DataFrame(),
        regression_coefficients_df=pd.DataFrame(),
        rolling_betas_df=pd.DataFrame(),
        event_study_summary_df=pd.DataFrame(),
        historical_analogues_df=pd.DataFrame(),
        catalyst_primary_df=pd.DataFrame(),
        market_constraints_df=pd.DataFrame(),
        confidence_framework_df=pd.DataFrame(),
        regime_state_df=pd.DataFrame([{"regime_label": "normal", "regime_summary": ""}]),
        source_log_df=pd.DataFrame(),
        output_dir=tmp_path / "outputs" / "packets" / "company_case_packets",
        as_of_date=date(2026, 4, 5),
        root_dir=tmp_path,
    )

    packet_path = Path(index_df.iloc[0]["packet_path"])
    assert not packet_path.is_absolute()
    assert (tmp_path / packet_path).exists()
