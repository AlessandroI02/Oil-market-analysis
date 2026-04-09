from __future__ import annotations

import numpy as np
import pandas as pd


def _contains(series: pd.Series, text: str) -> pd.Series:
    return series.astype(str).str.lower().str.contains(text.lower(), na=False)


def build_peer_baskets(
    included_df: pd.DataFrame,
    archetypes_df: pd.DataFrame,
    route_risks_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if included_df is None or included_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    base = included_df[["company_name", "ticker", "bucket_classification"]].copy()
    if archetypes_df is not None and not archetypes_df.empty:
        base = base.merge(archetypes_df[["ticker", "archetype"]], on="ticker", how="left")
    else:
        base["archetype"] = ""
    if route_risks_df is not None and not route_risks_df.empty:
        route_cols = [c for c in ["ticker", "qualitative_route_risk"] if c in route_risks_df.columns]
        base = base.merge(route_risks_df[route_cols], on="ticker", how="left")
    else:
        base["qualitative_route_risk"] = "Medium"
    if factor_df is not None and not factor_df.empty:
        fac_cols = [c for c in ["ticker", "beta_brent_ret"] if c in factor_df.columns]
        base = base.merge(factor_df[fac_cols], on="ticker", how="left")
    else:
        base["beta_brent_ret"] = np.nan
    if exposure_df is not None and not exposure_df.empty:
        exp_cols = [c for c in ["ticker", "combined_exposure_pct"] if c in exposure_df.columns]
        base = base.merge(exposure_df[exp_cols], on="ticker", how="left")
    else:
        base["combined_exposure_pct"] = np.nan

    brent_beta = pd.to_numeric(base.get("beta_brent_ret"), errors="coerce")
    exposure = pd.to_numeric(base.get("combined_exposure_pct"), errors="coerce")
    beta_cut = float(brent_beta.median()) if brent_beta.notna().any() else 1.0
    exposure_hi = float(exposure.quantile(0.7)) if exposure.notna().any() else 35.0
    exposure_lo = float(exposure.quantile(0.3)) if exposure.notna().any() else 20.0

    masks: list[tuple[str, str, pd.Series]] = [
        (
            "integrated supermajors",
            "Archetype-classified integrated supermajors with broad global operating footprint.",
            _contains(base["archetype"], "supermajor"),
        ),
        (
            "integrated regional majors",
            "Integrated regional majors outside the supermajor cohort.",
            _contains(base["archetype"], "regional major"),
        ),
        (
            "downstream-heavy near-matches",
            "Downstream-oriented listed names that still offer oil-market transmission exposure.",
            _contains(base["archetype"], "downstream"),
        ),
        (
            "national champion listed majors",
            "Listed national champions with strategic state-linked positioning.",
            _contains(base["archetype"], "national champion"),
        ),
        (
            "high-beta oil expressions",
            "Names with Brent beta at or above cohort median.",
            brent_beta >= beta_cut,
        ),
        (
            "low-route-risk expressions",
            "Lower route-risk names by qualitative route label and lower exposure quantile.",
            (base["qualitative_route_risk"].astype(str).str.lower() == "low") | (exposure <= exposure_lo),
        ),
        (
            "high-route-risk expressions",
            "Higher route-risk names by qualitative route label and upper exposure quantile.",
            (base["qualitative_route_risk"].astype(str).str.lower() == "high") | (exposure >= exposure_hi),
        ),
    ]

    membership_rows: list[dict[str, object]] = []
    basket_rows: list[dict[str, object]] = []
    for basket_name, description, mask in masks:
        members = base[mask.fillna(False)].copy()
        members = members.drop_duplicates(subset=["ticker"])
        basket_rows.append(
            {
                "basket_name": basket_name,
                "basket_description": description,
                "selection_logic": description,
                "member_count": int(len(members)),
            }
        )
        for _, row in members.iterrows():
            membership_rows.append(
                {
                    "basket_name": basket_name,
                    "company_name": row["company_name"],
                    "ticker": row["ticker"],
                    "bucket_classification": row.get("bucket_classification", ""),
                    "archetype": row.get("archetype", ""),
                }
            )

    baskets_df = pd.DataFrame(basket_rows).sort_values("basket_name").reset_index(drop=True)
    membership_df = pd.DataFrame(membership_rows).sort_values(["basket_name", "ticker"]).reset_index(drop=True)
    return baskets_df, membership_df
