from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")


def _save_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _validate_chart_inputs(
    df: pd.DataFrame,
    required_columns: list[str],
    min_non_null_ratio: float,
    min_points: int,
) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "ok": False,
            "row_count": 0,
            "non_null_count": 0,
            "non_null_ratio": 0.0,
            "reason": "Input dataframe is empty",
        }

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return {
            "ok": False,
            "row_count": len(df),
            "non_null_count": 0,
            "non_null_ratio": 0.0,
            "reason": f"Missing required columns: {', '.join(missing)}",
        }

    block = df[required_columns].apply(pd.to_numeric, errors="coerce")
    row_count = len(block)
    non_null_count = int(block.notna().sum().sum())
    denom = max(row_count * len(required_columns), 1)
    ratio = non_null_count / denom

    min_col_points = int(block.notna().sum().min()) if len(required_columns) > 0 else row_count
    ok = ratio >= min_non_null_ratio and min_col_points >= min_points
    reason = ""
    if not ok:
        reason = (
            f"Insufficient numeric coverage (ratio={ratio:.2f}, min_col_points={min_col_points}, "
            f"threshold_ratio={min_non_null_ratio:.2f}, threshold_points={min_points})"
        )

    return {
        "ok": ok,
        "row_count": row_count,
        "non_null_count": non_null_count,
        "non_null_ratio": ratio,
        "minimum_required_ratio": min_non_null_ratio,
        "reason": reason,
    }


def _chart_meta(
    sheet_name: str,
    image_path: Path,
    explanation: str,
    takeaway: str,
    required_columns: list[str],
    validation: dict[str, Any],
    minimum_required_ratio: float | None = None,
) -> dict[str, Any]:
    threshold = (
        float(minimum_required_ratio)
        if minimum_required_ratio is not None
        else float(validation.get("minimum_required_ratio", 0.20))
    )
    return {
        "sheet_name": sheet_name,
        "image_path": str(image_path),
        "explanation": explanation,
        "takeaway": takeaway,
        "status": "VALID" if validation.get("ok") else "UNAVAILABLE",
        "reason": validation.get("reason", ""),
        "required_columns": required_columns,
        "row_count": validation.get("row_count", 0),
        "non_null_count": validation.get("non_null_count", 0),
        "non_null_ratio": validation.get("non_null_ratio", 0.0),
        "minimum_non_null_ratio": threshold,
    }


def create_charts(
    exposure_df: pd.DataFrame,
    crude_tracker_df: pd.DataFrame,
    fuel_tracker_df: pd.DataFrame,
    equity_tracker_df: pd.DataFrame,
    operating_mix_df: pd.DataFrame,
    route_risks_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
    chart_dir: Path,
    core_ranking_df: pd.DataFrame | None = None,
    extended_ranking_df: pd.DataFrame | None = None,
    scenario_df: pd.DataFrame | None = None,
    valuation_df: pd.DataFrame | None = None,
    catalyst_df: pd.DataFrame | None = None,
    min_non_null_ratio: float = 0.20,
    min_points: int = 3,
) -> list[dict[str, Any]]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    charts: list[dict[str, Any]] = []
    core_ranking_df = core_ranking_df if core_ranking_df is not None else pd.DataFrame()
    extended_ranking_df = extended_ranking_df if extended_ranking_df is not None else pd.DataFrame()
    scenario_df = scenario_df if scenario_df is not None else pd.DataFrame()
    valuation_df = valuation_df if valuation_df is not None else pd.DataFrame()
    catalyst_df = catalyst_df if catalyst_df is not None else pd.DataFrame()

    # 1) Hormuz exposure ranking
    path = chart_dir / "hormuz_exposure_ranking.png"
    ordered = exposure_df.sort_values("combined_exposure_pct", ascending=True) if not exposure_df.empty else pd.DataFrame()
    v = _validate_chart_inputs(ordered, ["combined_exposure_pct"], min_non_null_ratio, min_points)
    if v["ok"] and "ticker" in ordered.columns:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(ordered["ticker"], ordered["combined_exposure_pct"], color="#264653")
        ax.set_xlabel("Combined Hormuz Exposure (%)")
        ax.set_title("Hormuz Exposure Ranking (Least to Most)")
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
    else:
        _save_placeholder(path, "Hormuz Exposure Ranking", v["reason"] or "No exposure data available.")
    charts.append(_chart_meta("Chart_Hormuz_Ranking", path, "Ranks companies by estimated share of international hydrocarbon volume reliant on Hormuz transit.", "Higher bars indicate greater route concentration risk through Hormuz.", ["combined_exposure_pct"], v))

    # 2) crude vs blended fuels line
    path = chart_dir / "crude_vs_fuels_line.png"
    fuel_avg = fuel_tracker_df.groupby("date", as_index=False)["blended_combined_fuels_price"].mean() if (not fuel_tracker_df.empty and "blended_combined_fuels_price" in fuel_tracker_df.columns) else pd.DataFrame()
    merged = crude_tracker_df[["date", "brent_price"]].merge(fuel_avg, on="date", how="left") if (not crude_tracker_df.empty and {"date", "brent_price"}.issubset(set(crude_tracker_df.columns))) else pd.DataFrame()
    v = _validate_chart_inputs(merged, ["brent_price", "blended_combined_fuels_price"], min_non_null_ratio, min_points)
    if v["ok"]:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(merged["date"], merged["brent_price"], label="Brent ($/bbl)", color="#1d3557")
        ax.plot(merged["date"], merged["blended_combined_fuels_price"], label="Avg weighted blended fuels ($/bbl eq)", color="#e76f51")
        ax.set_title("Brent vs Weighted Blended Retail Fuels")
        ax.set_ylabel("USD per barrel-equivalent")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
    else:
        _save_placeholder(path, "Brent vs Blended Fuels", v["reason"] or "Insufficient data for line comparison.")
    charts.append(_chart_meta("Chart_Crude_vs_Fuel", path, "Compares weekly Brent benchmark against portfolio-average blended fuel prices.", "Divergence highlights margin/tax effects between wholesale crude and end-consumer fuel pricing.", ["brent_price", "blended_combined_fuels_price"], v))

    # 3) fuels/crude ratio
    path = chart_dir / "fuels_to_crude_ratio.png"
    ratio = fuel_tracker_df.groupby("date", as_index=False)["fuels_to_brent_ratio"].mean() if (not fuel_tracker_df.empty and "fuels_to_brent_ratio" in fuel_tracker_df.columns) else pd.DataFrame()
    v = _validate_chart_inputs(ratio, ["fuels_to_brent_ratio"], min_non_null_ratio, min_points)
    if v["ok"]:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(ratio["date"], ratio["fuels_to_brent_ratio"], color="#2a9d8f")
        ax.set_title("Weighted Fuels-to-Brent Ratio")
        ax.set_ylabel("Ratio")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
    else:
        _save_placeholder(path, "Fuels-to-Brent Ratio", v["reason"] or "No fuel ratio data available.")
    charts.append(_chart_meta("Chart_Fuel_Crude_Ratio", path, "Shows whether blended pump-price equivalents are widening or narrowing versus Brent.", "A rising ratio suggests fuel prices are outpacing crude (or crude is weakening faster).", ["fuels_to_brent_ratio"], v))

    # 4) indexed share price performance
    path = chart_dir / "indexed_equity_performance.png"
    v = _validate_chart_inputs(equity_tracker_df, ["equity_indexed_100"], min_non_null_ratio, min_points)
    if v["ok"] and "ticker" in equity_tracker_df.columns and "date" in equity_tracker_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        for ticker, gdf in equity_tracker_df.groupby("ticker"):
            ax.plot(gdf["date"], gdf["equity_indexed_100"], label=ticker, linewidth=1.2)
        ax.set_title("Indexed Share Price Performance (Base=100)")
        ax.set_ylabel("Indexed Level")
        ax.legend(ncol=4, fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
    else:
        _save_placeholder(path, "Indexed Equity Performance", v["reason"] or "No equity tracker data available.")
    charts.append(_chart_meta("Chart_Equity_Indexed", path, "Tracks relative equity moves for all included companies on a normalized starting point.", "Steeper lines indicate stronger outperformance over the sampled 3-month window.", ["equity_indexed_100"], v))

    # 5) scatter exposure vs equity performance
    path = chart_dir / "exposure_vs_equity_scatter.png"
    if not exposure_df.empty and not equity_tracker_df.empty and {"date", "equity_cumulative_pct", "company_name", "ticker"}.issubset(set(equity_tracker_df.columns)):
        final_perf = (
            equity_tracker_df.sort_values("date")
            .groupby(["company_name", "ticker"], as_index=False)
            .tail(1)[["company_name", "ticker", "equity_cumulative_pct"]]
        )
        scatter_df = exposure_df.merge(final_perf, on=["company_name", "ticker"], how="left")
    else:
        scatter_df = pd.DataFrame()
    v = _validate_chart_inputs(scatter_df, ["combined_exposure_pct", "equity_cumulative_pct"], min_non_null_ratio, min_points)
    if v["ok"]:
        valid = scatter_df.dropna(subset=["combined_exposure_pct", "equity_cumulative_pct"])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(valid["combined_exposure_pct"], valid["equity_cumulative_pct"], color="#457b9d")
        if "ticker" in valid.columns:
            for _, r in valid.iterrows():
                ax.annotate(r["ticker"], (r["combined_exposure_pct"], r["equity_cumulative_pct"]), fontsize=8)
        ax.set_xlabel("Combined Hormuz Exposure (%)")
        ax.set_ylabel("3M Equity Cumulative Return (%)")
        ax.set_title("Exposure vs Equity Performance")
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
    else:
        _save_placeholder(path, "Exposure vs Equity", v["reason"] or "Insufficient data for scatter plot.")
    charts.append(_chart_meta("Chart_Exposure_vs_Equity", path, "Compares estimated route exposure with realized 3-month equity returns.", "Points far from trend imply market pricing factors beyond pure Hormuz route sensitivity.", ["combined_exposure_pct", "equity_cumulative_pct"], v))

    # 6) bubble chart exposure vs upstream mix vs performance
    path = chart_dir / "bubble_exposure_upstream_performance.png"
    if not exposure_df.empty and not operating_mix_df.empty and not equity_tracker_df.empty and "date" in equity_tracker_df.columns:
        final_perf = (
            equity_tracker_df.sort_values("date")
            .groupby(["company_name", "ticker"], as_index=False)
            .tail(1)[["company_name", "ticker", "equity_cumulative_pct"]]
        )
        bubble_df = exposure_df.merge(operating_mix_df[["company_name", "ticker", "upstream_share_pct"]], on=["company_name", "ticker"], how="left")
        bubble_df = bubble_df.merge(final_perf, on=["company_name", "ticker"], how="left")
    else:
        bubble_df = pd.DataFrame()
    v = _validate_chart_inputs(bubble_df, ["combined_exposure_pct", "equity_cumulative_pct", "upstream_share_pct"], min_non_null_ratio, min_points)
    if v["ok"]:
        valid = bubble_df.dropna(subset=["combined_exposure_pct", "equity_cumulative_pct"])
        fig, ax = plt.subplots(figsize=(11, 6))
        size = valid["upstream_share_pct"].fillna(0) * 8
        ax.scatter(valid["combined_exposure_pct"], valid["equity_cumulative_pct"], s=size, alpha=0.65, color="#e63946", edgecolor="black")
        if "ticker" in valid.columns:
            for _, r in valid.iterrows():
                ax.annotate(r["ticker"], (r["combined_exposure_pct"], r["equity_cumulative_pct"]), fontsize=8)
        ax.set_xlabel("Combined Hormuz Exposure (%)")
        ax.set_ylabel("3M Equity Cumulative Return (%)")
        ax.set_title("Bubble: Exposure vs Performance (Bubble Size = Upstream Mix %)")
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
    else:
        _save_placeholder(path, "Bubble Exposure-Upstream-Performance", v["reason"] or "Insufficient data for bubble chart.")
    charts.append(_chart_meta("Chart_Bubble_Profile", path, "Visualizes how upstream weighting may interact with disruption exposure and returns.", "Larger bubbles with lower exposure can indicate stronger disruption resilience with oil-beta support.", ["combined_exposure_pct", "equity_cumulative_pct", "upstream_share_pct"], v))

    # 7) stacked bar upstream vs downstream mix
    path = chart_dir / "operating_mix_stacked.png"
    v = _validate_chart_inputs(operating_mix_df, ["upstream_share_pct", "downstream_share_pct"], min_non_null_ratio, min_points)
    if v["ok"] and "ticker" in operating_mix_df.columns:
        fig, ax = plt.subplots(figsize=(11, 6))
        idx = np.arange(len(operating_mix_df))
        ax.bar(idx, operating_mix_df["upstream_share_pct"], label="Upstream %", color="#1d3557")
        ax.bar(idx, operating_mix_df["downstream_share_pct"], bottom=operating_mix_df["upstream_share_pct"], label="Downstream %", color="#a8dadc")
        ax.set_xticks(idx)
        ax.set_xticklabels(operating_mix_df["ticker"], rotation=45, ha="right")
        ax.set_ylabel("Share (%)")
        ax.set_title("Operating Mix: Upstream vs Downstream")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
    else:
        _save_placeholder(path, "Operating Mix Stacked", v["reason"] or "No operating mix data available.")
    charts.append(_chart_meta("Chart_Operating_Mix", path, "Compares normalized operating mix across included companies.", "Higher upstream mix generally increases crude sensitivity while downstream can buffer volatility.", ["upstream_share_pct", "downstream_share_pct"], v))

    # 8) route risk heatmap
    path = chart_dir / "route_risk_heatmap.png"
    req_cols = ["hormuz_used", "qualitative_route_risk"]
    v = _validate_chart_inputs(route_risks_df.assign(hormuz_used_num=route_risks_df.get("hormuz_used", pd.Series(dtype=float)).map({True: 1, False: 0})) if not route_risks_df.empty else pd.DataFrame(), ["hormuz_used_num"], min_non_null_ratio, min_points)
    if v["ok"] and all(c in route_risks_df.columns for c in ["ticker", "hormuz_used", "qualitative_route_risk"]):
        risk_map = {"Low": 1, "Medium": 2, "High": 3}
        plot_df = route_risks_df[["ticker", "hormuz_used", "qualitative_route_risk"]].copy()
        plot_df["hormuz_score"] = plot_df["hormuz_used"].map({True: 3, False: 1})
        plot_df["route_risk_score"] = plot_df["qualitative_route_risk"].map(risk_map).fillna(1)
        matrix = plot_df[["hormuz_score", "route_risk_score"]].to_numpy().T

        fig, ax = plt.subplots(figsize=(12, 3.5))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Hormuz Use", "Route Risk"])
        ax.set_xticks(np.arange(len(plot_df)))
        ax.set_xticklabels(plot_df["ticker"], rotation=45, ha="right")
        ax.set_title("Route Risk Heatmap")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
    else:
        _save_placeholder(path, "Route Risk Heatmap", v["reason"] or "No route risk records available.")
    charts.append(_chart_meta("Chart_Route_Heatmap", path, "Heatmap of qualitative route exposure and Hormuz reliance across the company set.", "Darker cells indicate higher route concentration risk.", req_cols, v))

    # 9) earnings timeline
    path = chart_dir / "earnings_timeline.png"
    if not earnings_df.empty and "next_earnings_date" in earnings_df.columns:
        tmp = earnings_df.copy()
        tmp["next_earnings_date"] = pd.to_datetime(tmp["next_earnings_date"], errors="coerce")
        tmp = tmp.dropna(subset=["next_earnings_date"]).sort_values("next_earnings_date")
    else:
        tmp = pd.DataFrame()
    v = _validate_chart_inputs(tmp.assign(next_date_num=tmp.get("next_earnings_date").astype("int64") if not tmp.empty else pd.Series(dtype=float)), ["next_date_num"], min_non_null_ratio, 1)
    if v["ok"] and "ticker" in tmp.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        y = np.arange(len(tmp))
        ax.scatter(tmp["next_earnings_date"], y, color="#2a9d8f")
        ax.set_yticks(y)
        ax.set_yticklabels(tmp["ticker"])
        ax.set_title("Upcoming Earnings Timeline")
        ax.set_xlabel("Date")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
    else:
        _save_placeholder(path, "Earnings Timeline", v["reason"] or "No valid earnings dates available.")
    charts.append(_chart_meta("Chart_Earnings_Timeline", path, "Maps next earnings windows for the covered companies.", "Clusters of dates can concentrate event risk and repricing potential.", ["next_earnings_date"], v))

    # 10) core ranking score bars
    path = chart_dir / "core_ranking_scores.png"
    plot_df = core_ranking_df.sort_values("final_score", ascending=True).tail(12) if (not core_ranking_df.empty and "final_score" in core_ranking_df.columns) else pd.DataFrame()
    v = _validate_chart_inputs(plot_df, ["final_score"], min_non_null_ratio, min_points)
    if v["ok"] and "ticker" in plot_df.columns:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(plot_df["ticker"], plot_df["final_score"], color="#0a9396")
        ax.set_title("Core Ranking Final Scores")
        ax.set_xlabel("Score (0-100)")
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
    else:
        _save_placeholder(path, "Core Ranking Final Scores", v["reason"] or "No core ranking data available.")
    charts.append(_chart_meta("Chart_Core_Ranking", path, "Ranks core integrated names by confidence-adjusted composite score.", "Higher score suggests stronger risk-adjusted expression of the thesis.", ["final_score"], v))

    # 11) stacked chokepoint exposure
    path = chart_dir / "chokepoint_stacked_exposure.png"
    req_cols2 = ["hormuz_share_pct", "bab_el_mandeb_share_pct", "suez_share_pct", "non_chokepoint_share_pct"]
    v = _validate_chart_inputs(route_risks_df, req_cols2, min_non_null_ratio, min_points)
    if v["ok"] and "ticker" in route_risks_df.columns:
        plot_df = route_risks_df.copy()
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(plot_df))
        ax.bar(x, plot_df["hormuz_share_pct"], label="Hormuz", color="#bb3e03")
        ax.bar(x, plot_df["bab_el_mandeb_share_pct"], bottom=plot_df["hormuz_share_pct"], label="Bab el-Mandeb", color="#ca6702")
        ax.bar(x, plot_df["suez_share_pct"], bottom=plot_df["hormuz_share_pct"] + plot_df["bab_el_mandeb_share_pct"], label="Suez", color="#ee9b00")
        ax.bar(x, plot_df["non_chokepoint_share_pct"], bottom=plot_df["hormuz_share_pct"] + plot_df["bab_el_mandeb_share_pct"] + plot_df["suez_share_pct"], label="Non-chokepoint", color="#94d2bd")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["ticker"], rotation=45, ha="right")
        ax.set_ylabel("Estimated share (%)")
        ax.set_title("Estimated Route Exposure by Chokepoint")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
    else:
        _save_placeholder(path, "Estimated Route Exposure by Chokepoint", v["reason"] or "No chokepoint share data available.")
    charts.append(_chart_meta("Chart_Chokepoints_Stacked", path, "Shows estimated route share by key chokepoint and non-chokepoint flows per company.", "Higher Hormuz/Bab/Suez stacks imply greater transport fragility in disruption scenarios.", req_cols2, v))

    # 12) valuation vs exposure quadrant
    path = chart_dir / "valuation_vs_exposure_quadrant.png"
    if not exposure_df.empty and not valuation_df.empty and "fcf_yield_pct" in valuation_df.columns:
        quad_df = exposure_df.merge(valuation_df[["ticker", "fcf_yield_pct", "valuation_score"]], on="ticker", how="left")
    else:
        quad_df = pd.DataFrame()
    v = _validate_chart_inputs(quad_df, ["combined_exposure_pct", "fcf_yield_pct"], min_non_null_ratio, min_points)
    if v["ok"]:
        valid = quad_df.dropna(subset=["combined_exposure_pct", "fcf_yield_pct"]).copy()
        fig, ax = plt.subplots(figsize=(10, 6))
        sizes = valid["valuation_score"].fillna(50) * 4
        ax.scatter(valid["combined_exposure_pct"], valid["fcf_yield_pct"], s=sizes, alpha=0.65, color="#3a86ff", edgecolor="black")
        if "ticker" in valid.columns:
            for _, r in valid.iterrows():
                ax.annotate(str(r["ticker"]), (r["combined_exposure_pct"], r["fcf_yield_pct"]), fontsize=8)
        ax.set_xlabel("Combined Hormuz Exposure (%)")
        ax.set_ylabel("FCF Yield (%)")
        ax.set_title("Valuation vs Route Exposure (Bubble=Valuation Score)")
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
    else:
        _save_placeholder(path, "Valuation vs Route Exposure", v["reason"] or "Insufficient exposure/valuation data.")
    charts.append(_chart_meta("Chart_Valuation_Quadrant", path, "Compares route exposure against FCF yield, with bubble size reflecting composite valuation attractiveness.", "Names in high-yield/lower-exposure zones often represent cleaner risk-adjusted setups.", ["combined_exposure_pct", "fcf_yield_pct"], v))

    # 13) scenario heatmap
    path = chart_dir / "scenario_heatmap.png"
    if not scenario_df.empty and {"ticker", "scenario_brent", "fcf_sensitivity_pct"}.issubset(set(scenario_df.columns)):
        pivot = scenario_df.pivot_table(index="ticker", columns="scenario_brent", values="fcf_sensitivity_pct", aggfunc="mean")
    else:
        pivot = pd.DataFrame()
    pivot_for_validation = pd.DataFrame()
    required_cols = ["fcf_sensitivity_pct"]
    if not pivot.empty:
        pivot_for_validation = pivot.copy()
        pivot_for_validation.columns = [str(c) for c in pivot_for_validation.columns]
        required_cols = list(pivot_for_validation.columns)
    v = _validate_chart_inputs(
        pivot_for_validation.reset_index(drop=True) if not pivot_for_validation.empty else pd.DataFrame(),
        required_cols,
        min_non_null_ratio,
        min_points,
    )
    if not pivot.empty and v["ok"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"Brent {int(c)}" for c in pivot.columns])
        ax.set_title("Scenario Heatmap: FCF Sensitivity (%)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
    else:
        _save_placeholder(path, "Scenario Heatmap", v["reason"] or "No scenario sensitivity data available.")
    charts.append(_chart_meta("Chart_Scenario_Heatmap", path, "Heatmap of company free-cash-flow sensitivity across configured Brent scenarios.", "Warmer colors indicate stronger positive FCF torque under higher Brent cases.", ["fcf_sensitivity_pct"], v))

    # 14) catalyst timeline
    path = chart_dir / "catalyst_timeline.png"
    if not catalyst_df.empty and {"event_date", "ticker", "event"}.issubset(set(catalyst_df.columns)):
        tmp = catalyst_df.copy()
        tmp["event_date"] = pd.to_datetime(tmp["event_date"], errors="coerce")
        tmp = tmp.dropna(subset=["event_date"]).sort_values("event_date").head(40)
    else:
        tmp = pd.DataFrame()
    v = _validate_chart_inputs(tmp.assign(event_num=tmp.get("event_date").astype("int64") if not tmp.empty else pd.Series(dtype=float)), ["event_num"], min_non_null_ratio, 1)
    if v["ok"] and not tmp.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        y = np.arange(len(tmp))
        ax.scatter(tmp["event_date"], y, color="#6a4c93")
        labels = tmp.apply(lambda r: f"{r['ticker']} {r['event']}", axis=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title("Catalyst Timeline (Near-term)")
        ax.set_xlabel("Date")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
    else:
        _save_placeholder(path, "Catalyst Timeline", v["reason"] or "No catalyst calendar available.")
    charts.append(_chart_meta("Chart_Catalyst_Timeline", path, "Shows scheduled company and macro events that may influence thesis timing and rerating windows.", "Clusters of catalysts can create higher volatility and tactical entry/exit opportunities.", ["event_date"], v))

    logger.info("Generated %s chart images", len(charts))
    return charts
