from __future__ import annotations

import pandas as pd


SUPERMAJOR_TICKERS = {"XOM", "CVX", "SHEL", "BP", "TTE"}
NATIONAL_CHAMPION_TICKERS = {"2222.SR", "0857.HK", "0386.HK", "0883.HK", "PBR"}


def classify_archetype(ticker: str, bucket: str, upstream_share_pct: float | None, downstream_share_pct: float | None) -> str:
    if ticker in SUPERMAJOR_TICKERS:
        return "Integrated supermajor"
    if ticker in NATIONAL_CHAMPION_TICKERS:
        return "National champion listed major"

    up = upstream_share_pct if upstream_share_pct is not None else 50.0
    down = downstream_share_pct if downstream_share_pct is not None else 50.0

    if bucket == "secondary":
        if down >= up:
            return "Downstream-heavy near-match"
        return "Secondary expression"

    if up >= 65:
        return "Upstream-heavy integrated name"
    if down >= 60:
        return "Downstream-integrated name"
    return "Integrated regional major"


def build_archetypes(universe_df: pd.DataFrame, operating_mix_df: pd.DataFrame) -> pd.DataFrame:
    if universe_df.empty:
        return pd.DataFrame(
            columns=[
                "company_name",
                "ticker",
                "bucket_classification",
                "archetype",
                "inclusion_confidence",
            ]
        )

    mix = operating_mix_df[["ticker", "upstream_share_pct", "downstream_share_pct"]].copy() if not operating_mix_df.empty else pd.DataFrame(columns=["ticker", "upstream_share_pct", "downstream_share_pct"])
    out = universe_df.merge(mix, on="ticker", how="left")

    out["archetype"] = out.apply(
        lambda r: classify_archetype(
            ticker=str(r["ticker"]),
            bucket=str(r["bucket_classification"]),
            upstream_share_pct=(None if pd.isna(r.get("upstream_share_pct")) else float(r.get("upstream_share_pct"))),
            downstream_share_pct=(None if pd.isna(r.get("downstream_share_pct")) else float(r.get("downstream_share_pct"))),
        ),
        axis=1,
    )

    out["inclusion_confidence"] = out.get("confidence", "Medium")

    return out[
        [
            "company_name",
            "ticker",
            "bucket_classification",
            "archetype",
            "inclusion_confidence",
        ]
    ]
