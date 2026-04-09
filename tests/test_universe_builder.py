import pandas as pd

from src.source_logger import SourceLogger
from src.universe_builder import build_universe


def test_universe_builder_primary_and_rejected(monkeypatch):
    monkeypatch.setattr("src.universe_builder._fetch_market_cap", lambda ticker: None)

    logger = SourceLogger()
    result = build_universe(
        include_secondary=True,
        min_market_cap_usd=10_000_000_000,
        source_logger=logger,
        max_companies=50,
    )

    assert not result.reviewed.empty
    assert "bucket_classification" in result.reviewed.columns
    assert (result.included["bucket_classification"].isin(["primary", "secondary"])).all()
    assert (result.rejected["bucket_classification"] == "rejected").all()
    assert "Exxon Mobil" in set(result.reviewed["company_name"])

    source_df = logger.to_dataframe()
    assert not source_df.empty
    assert "source_url" in source_df.columns
