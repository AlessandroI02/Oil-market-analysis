from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ReportInsights:
    executive_summary: str
    research_question: str
    variant_perception: str
    why_now: str
    core_top_names: list[str]
    core_bottom_names: list[str]
    extended_top_names: list[str]
    recommendation_text: str
    what_market_missing: str
    key_risks: list[str]
    falsifiers: list[str]
    run_status: str = "VALID"
    recommendation_allowed: bool = True
    section_caveats: list[str] | None = None


def _top_tickers(df: pd.DataFrame, n: int, ascending: bool = False, score_col: str = "final_score") -> list[str]:
    if df.empty or score_col not in df.columns:
        return []
    return df.sort_values(score_col, ascending=ascending).head(n)["ticker"].tolist()


def _fmt_num(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.1f}"
    except Exception:
        return "n/a"


def build_insights(
    core_ranking_df: pd.DataFrame,
    extended_ranking_df: pd.DataFrame,
    valuation_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    analogue_df: pd.DataFrame,
    catalyst_df: pd.DataFrame,
    recommendation_df: pd.DataFrame,
    quality_df: pd.DataFrame | None = None,
    run_summary: dict[str, object] | None = None,
    section_health_df: pd.DataFrame | None = None,
) -> ReportInsights:
    run_summary = run_summary or {"run_status": "VALID"}
    run_status = str(run_summary.get("run_status", "VALID")).upper()

    core_top = _top_tickers(core_ranking_df, 5, ascending=False)
    core_bottom = _top_tickers(core_ranking_df, 5, ascending=True)
    extended_top = _top_tickers(extended_ranking_df, 5, ascending=False)

    cheap_names = []
    if not valuation_df.empty and "valuation_score" in valuation_df.columns:
        cheap_names = valuation_df.sort_values("valuation_score", ascending=False).head(3)["ticker"].tolist()

    scenario_positive = []
    if not scenario_df.empty and "scenario_brent" in scenario_df.columns:
        snap = scenario_df[scenario_df["scenario_brent"] == 100]
        if not snap.empty:
            scenario_positive = snap.sort_values("fcf_sensitivity_pct", ascending=False).head(3)["ticker"].tolist()

    residual_names = []
    if not factor_df.empty and "idiosyncratic_residual_pct" in factor_df.columns:
        residual_names = factor_df.sort_values("idiosyncratic_residual_pct", ascending=False).head(3)["ticker"].tolist()

    near_catalysts = []
    if not catalyst_df.empty and {"ticker", "event_date"}.issubset(set(catalyst_df.columns)):
        tmp = catalyst_df[catalyst_df["ticker"] != "GLOBAL"].copy()
        tmp["event_date"] = pd.to_datetime(tmp["event_date"], errors="coerce")
        tmp = tmp.dropna(subset=["event_date"]).sort_values("event_date")
        near_catalysts = tmp.head(5)["ticker"].tolist()

    analogue_note = ""
    if not analogue_df.empty:
        last = analogue_df.tail(1)
        if not last.empty:
            brent_ret = _fmt_num(last.iloc[0].get("brent_return_pct"))
            peer_ret = _fmt_num(last.iloc[0].get("peer_median_return_pct"))
            analogue_note = (
                f"Historical analogue check shows {last.iloc[0].get('period')} had Brent {brent_ret}% "
                f"with peer median equity {peer_ret}% where data available."
            )

    section_caveats: list[str] = []
    if section_health_df is not None and not section_health_df.empty:
        bad = section_health_df[section_health_df["section_status"].isin(["DEGRADED", "INVALID", "UNAVAILABLE"])]
        for _, row in bad.iterrows():
            section_caveats.append(f"{row['section_name']}: {row['section_status']} ({row['reason']})")

    core_publishable_share = 0.0
    if not core_ranking_df.empty and "publishable_flag" in core_ranking_df.columns:
        core_publishable_share = float(pd.to_numeric(core_ranking_df["publishable_flag"], errors="coerce").fillna(0).mean())

    recommendation_allowed = run_status == "VALID" and core_publishable_share >= 0.5

    rec_summary = ""
    if recommendation_allowed and not recommendation_df.empty:
        rec_summary = "; ".join(
            recommendation_df.groupby("category")["ticker"]
            .apply(lambda s: ", ".join(s.head(3)))
            .head(6)
            .reset_index()
            .apply(lambda r: f"{r['category']}: {r['ticker']}", axis=1)
        )
    elif not recommendation_df.empty:
        rec_summary = (
            "Expression screens are available but upstream readiness is limited; "
            "prioritize lower-confidence diagnostic interpretation."
        )

    quality_note = ""
    if quality_df is not None and not quality_df.empty and "data_quality_score" in quality_df.columns:
        q = quality_df.sort_values("data_quality_score", ascending=False)
        top_q = ", ".join(q.head(3)["ticker"].tolist())
        low_q = ", ".join(q.tail(3)["ticker"].tolist())
        quality_note = f" Data-quality leaders: {top_q if top_q else 'N/A'}; lower-confidence names: {low_q if low_q else 'N/A'}."

    if run_status == "VALID":
        executive_summary = (
            "This V2 upstream note ranks listed integrated oil market expressions for a prolonged Hormuz disruption using a confidence-adjusted framework that combines route exposure, valuation context, scenario resilience, factors, and catalysts. "
            f"Core top names currently screen as {', '.join(core_top) if core_top else 'N/A'}, while weaker core names include {', '.join(core_bottom) if core_bottom else 'N/A'}. "
            f"Extended ranking leaders are {', '.join(extended_top) if extended_top else 'N/A'}.{quality_note}"
        )
    elif run_status == "DEGRADED":
        executive_summary = (
            "Run status is DEGRADED: the model produced partial outputs but one or more analytical sections failed minimum data-quality checks. "
            "Interpret rankings and tactical conclusions as provisional only. "
            f"Directional leaders from available data: {', '.join(core_top) if core_top else 'N/A'}.{quality_note}"
        )
    else:
        executive_summary = (
            "Run status is INVALID: core market-data backbone did not meet minimum quality thresholds. "
            "Outputs are diagnostic and should not be used for publishable conclusions or downstream conviction recommendations."
        )

    variant_perception = (
        "Market pricing often clusters energy names by broad oil beta. The framework tests whether route optionality, downstream pass-through, and valuation dislocations create better risk-adjusted market expressions than simple high-beta oil exposure, without producing a final investment recommendation."
    )

    why_now = (
        f"Why now: geopolitical route risk remains non-zero, valuation dispersion is meaningful (cheap screen: {', '.join(cheap_names) if cheap_names else 'N/A'}), "
        f"scenario upside sensitivity is concentrated ({', '.join(scenario_positive) if scenario_positive else 'N/A'}), and near-term catalysts are present ({', '.join(near_catalysts) if near_catalysts else 'N/A'})."
    )

    market_missing = (
        f"Potential market mispricing: residual alpha candidates ({', '.join(residual_names) if residual_names else 'N/A'}) may not be fully explained by crude/market/sector factors. {analogue_note}"
    )

    key_risks = [
        "Rapid de-escalation in Gulf tensions can compress route-risk premia before catalysts materialize.",
        "Demand slowdown or policy shock can weaken oil-linked scenario outcomes despite favorable route positioning.",
        "Data confidence varies by company; names with heavier proxy usage should carry higher model uncertainty.",
    ]
    if run_status != "VALID":
        key_risks.append("Run-level data quality is below publishable standard for some core sections.")

    falsifiers = [
        "Core low-exposure names fail to outperform despite favorable valuation and catalyst setup.",
        "Factor decomposition shows no persistent idiosyncratic signal once oil/market/sector effects are removed.",
        "Scenario and historical analogue evidence diverges materially from current ranking outputs.",
    ]

    return ReportInsights(
        executive_summary=executive_summary,
        research_question="Which listed integrated oil equities provide the strongest confidence-adjusted market expression of prolonged Hormuz disruption?",
        variant_perception=variant_perception,
        why_now=why_now,
        core_top_names=core_top,
        core_bottom_names=core_bottom,
        extended_top_names=extended_top,
        recommendation_text=rec_summary,
        what_market_missing=market_missing,
        key_risks=key_risks,
        falsifiers=falsifiers,
        run_status=run_status,
        recommendation_allowed=recommendation_allowed,
        section_caveats=section_caveats,
    )
