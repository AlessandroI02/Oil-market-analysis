from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _one_writeup(row: pd.Series, category: str, catalyst: str | None) -> dict[str, str]:
    exposure = row.get("combined_exposure_pct")
    score = row.get("final_score")
    archetype = row.get("archetype", "")
    rating_status = str(row.get("rating_status", "provisional"))

    business_mix = (
        f"{row.get('company_name')} is classified as {archetype} with a market-side composite score of {score:.1f} and rating status {rating_status}."
        if pd.notna(score)
        else f"{row.get('company_name')} is classified as {archetype} with rating status {rating_status}."
    )

    if pd.notna(exposure):
        exposure_text = f"Estimated Hormuz-linked international hydrocarbon exposure is {float(exposure):.1f}% (central estimate)."
    else:
        exposure_text = "Route exposure remains uncertain due limited data coverage."

    market_missing = (
        "Screen suggests market pricing may not fully reflect route optionality plus valuation support."
        if category in {"Market-implied attractive setup", "Higher-beta expression screen"}
        else "Screen remains lower confidence and should be interpreted as a provisional market read."
    )

    key_risk = (
        "Main risk is rapid de-escalation reducing embedded route/stress premium before catalysts materialize."
        if category in {"Market-implied attractive setup", "Higher-beta expression screen"}
        else "Main risk is that proxy-heavy inputs reduce signal reliability."
    )

    why_now = (
        "Near-term catalyst clustering and elevated geopolitical uncertainty keep this market expression relevant."
    )

    return {
        "company_name": row.get("company_name"),
        "ticker": row.get("ticker"),
        "writeup_bucket": category,
        "business_mix_summary": business_mix,
        "why_it_screens": exposure_text,
        "what_market_may_be_missing": market_missing,
        "key_catalyst": catalyst or "Upcoming earnings / macro catalyst window",
        "key_risk": key_risk,
        "why_now": why_now,
    }


def build_company_writeups(
    core_ranking_df: pd.DataFrame,
    extended_ranking_df: pd.DataFrame,
    recommendation_df: pd.DataFrame,
) -> pd.DataFrame:
    if extended_ranking_df.empty:
        return pd.DataFrame()

    publishable_top = extended_ranking_df[extended_ranking_df.get("publishable_flag", False) == True]  # noqa: E712
    top = (publishable_top if not publishable_top.empty else extended_ranking_df).sort_values("final_score", ascending=False).head(5)
    bottom = extended_ranking_df.sort_values("final_score", ascending=True).head(5)

    interesting_tickers = recommendation_df[
        recommendation_df["category"].isin(["Higher-beta market expression", "Lower-route-risk market expression"])
    ]["ticker"].unique().tolist() if not recommendation_df.empty else []
    interesting = extended_ranking_df[extended_ranking_df["ticker"].isin(interesting_tickers)].head(5)

    rows = []
    for _, r in top.iterrows():
        rows.append(_one_writeup(r, "Market-implied attractive setup", None))
    for _, r in bottom.iterrows():
        rows.append(_one_writeup(r, "Lower-confidence screen", None))
    for _, r in interesting.iterrows():
        rows.append(_one_writeup(r, "Higher-beta expression screen", None))

    out = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "writeup_bucket"])
    logger.info("Built company writeups rows: %s", len(out))
    return out
