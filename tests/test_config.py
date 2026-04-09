from pathlib import Path

from src.config import load_settings


def test_load_settings():
    path = Path("config/settings.yaml")
    settings = load_settings(path)
    assert settings.project.name
    assert settings.run.lookback_months == 3
    assert settings.paths.output_excel.endswith("outputs/excel")
    assert settings.v2.model_version == "2.0"
    assert 100 in settings.v2.scenario_brent_levels
    assert settings.quality.news_price_move_threshold_pct == 2.0
