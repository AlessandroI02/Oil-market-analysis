import pandas as pd

from src.health_reporting import build_chart_health, build_sheet_health


def test_sheet_health_classification():
    datasets = {
        "Crude_Tracker": pd.DataFrame({"date": ["2026-01-01", "2026-01-08"], "brent_price": [80.0, None], "wti_price": [75.0, 76.0]}),
        "Valuation": pd.DataFrame(),
    }
    out = build_sheet_health(datasets, min_non_null_ratio=0.4)

    crude = out[out["sheet_name"] == "Crude_Tracker"].iloc[0]
    val = out[out["sheet_name"] == "Valuation"].iloc[0]

    assert crude["sheet_status"] in {"POPULATED", "DEGRADED"}
    assert val["sheet_status"] == "UNAVAILABLE"


def test_chart_health_export_shape():
    chart_meta = [
        {
            "sheet_name": "Chart_A",
            "status": "VALID",
            "required_columns": ["x", "y"],
            "row_count": 10,
            "non_null_count": 20,
            "minimum_non_null_ratio": 0.2,
            "reason": "",
            "image_path": "outputs/charts/chart_a.png",
        },
        {
            "sheet_name": "Chart_B",
            "status": "UNAVAILABLE",
            "required_columns": ["x"],
            "row_count": 0,
            "non_null_count": 0,
            "minimum_non_null_ratio": 0.2,
            "reason": "Input dataframe is empty",
            "image_path": "outputs/charts/chart_b.png",
        },
    ]

    out = build_chart_health(chart_meta)
    assert len(out) == 2
    assert set(out["chart_status"]) == {"VALID", "UNAVAILABLE"}
