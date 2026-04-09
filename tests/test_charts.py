import pandas as pd
from pathlib import Path

from src.charts import create_charts


def test_scenario_heatmap_valid_with_numeric_scenario_columns():
    scenario_df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "scenario_brent": [80.0, 100.0, 80.0, 100.0, 80.0, 100.0],
            "fcf_sensitivity_pct": [5.0, 8.0, 4.0, 7.0, 6.0, 9.0],
        }
    )
    chart_dir = Path("outputs/debug/pytest_tmp_charts")
    chart_dir.mkdir(parents=True, exist_ok=True)
    charts = create_charts(
        exposure_df=pd.DataFrame(),
        crude_tracker_df=pd.DataFrame(),
        fuel_tracker_df=pd.DataFrame(),
        equity_tracker_df=pd.DataFrame(),
        operating_mix_df=pd.DataFrame(),
        route_risks_df=pd.DataFrame(),
        earnings_df=pd.DataFrame(),
        chart_dir=chart_dir,
        scenario_df=scenario_df,
    )

    scenario_meta = next(c for c in charts if c["sheet_name"] == "Chart_Scenario_Heatmap")
    assert scenario_meta["status"] == "VALID"
