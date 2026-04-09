from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _sheet_name(name: str) -> str:
    return name[:31]


def _format_columns(ws, df: pd.DataFrame, workbook) -> None:
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})
    pct_fmt = workbook.add_format({"num_format": "0.00"})
    date_fmt = workbook.add_format({"num_format": "yyyy-mm-dd"})
    wrap_fmt = workbook.add_format({"text_wrap": True})

    for col_idx, col in enumerate(df.columns):
        ws.write(0, col_idx, col, header_fmt)
        max_len = max(len(str(col)), 12)
        if not df.empty:
            sample_values = df.iloc[:, col_idx].head(20).tolist()
            sample_max_len = max((len(str(value)) for value in sample_values), default=0)
            max_len = min(max(max_len, sample_max_len), 60)

        col_lower = str(col).lower()
        fmt = None
        if "pct" in col_lower or "ratio" in col_lower:
            fmt = pct_fmt
        if "date" in col_lower:
            fmt = date_fmt
        if "notes" in col_lower or "summary" in col_lower or "description" in col_lower or "reason" in col_lower:
            fmt = wrap_fmt
            max_len = max(max_len, 36)

        ws.set_column(col_idx, col_idx, max_len, fmt)

    ws.freeze_panes(1, 0)
    if len(df.columns) > 0:
        ws.autofilter(0, 0, max(len(df), 1), len(df.columns) - 1)


def _sheet_with_health(df: pd.DataFrame, status: str, reason: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            {
                "sheet_status": [status],
                "reason": [reason],
                "note": ["Sheet unavailable or below minimum quality threshold."],
            }
        )

    out = df.copy()
    out.insert(0, "sheet_status", status)
    out.insert(1, "sheet_reason", reason)
    return out


def export_excel_workbook(
    output_path: Path,
    datasets: dict[str, pd.DataFrame],
    chart_sheets: list[dict[str, Any]],
    methodology_summary: str,
    metadata: dict[str, str] | None = None,
    run_summary: dict[str, Any] | None = None,
    run_health_df: pd.DataFrame | None = None,
    sheet_health_df: pd.DataFrame | None = None,
    chart_health_df: pd.DataFrame | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_summary = run_summary or {}
    run_status = str(run_summary.get("run_status", "VALID"))

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        metadata = metadata or {}
        readme_df = pd.DataFrame(
            {
                "Section": [
                    "Workbook Generated",
                    "Run Status",
                    "Model Version",
                    "Assumptions Version",
                    "Data Cut Date",
                    "Source Refresh Date",
                    "Change Log Summary",
                    "Methodology Summary",
                    "Caveats",
                    "Sheet Guide",
                ],
                "Details": [
                    datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    run_status,
                    metadata.get("model_version", "2.1"),
                    metadata.get("assumptions_version", "2.1"),
                    metadata.get("data_cut_date", ""),
                    metadata.get("source_refresh_date", ""),
                    metadata.get("change_log_summary", "V2 reliability upgrade with run-health gating and provider fallback diagnostics."),
                    methodology_summary,
                    "Contains exact disclosures, analyst estimates, proxy estimates, and missing values. If run status is DEGRADED or INVALID, conclusions are provisional or suppressed.",
                    "Run_Health / Sheet_Health / Chart_Health / Regime_State / Market_Constraints / Market_Constraints_Methodology / Event_Episodes / Confidence_Framework / Confidence_Audit / Peer_Baskets / Company_Case_Packet_Index / Regression_Summary / Rolling_Beta_Summary / Event_Study_Summary / Universe / Core_Ranking / Extended_Ranking / Route_Exposure_Build / Crude_Tracker / Fuel_Tracker / Equity_Tracker / Valuation / Scenario_Analysis / Factor_Decomposition / Historical_Analogues / Catalyst_Calendar_Primary / Catalyst_Calendar_Archive / Assumptions / Source_Log",
                ],
            }
        )
        readme_df.to_excel(writer, index=False, sheet_name="README")
        _format_columns(writer.sheets["README"], readme_df, workbook)

        if run_health_df is not None:
            run_health_df.to_excel(writer, index=False, sheet_name="Run_Health")
            _format_columns(writer.sheets["Run_Health"], run_health_df, workbook)
        if sheet_health_df is not None:
            sheet_health_df.to_excel(writer, index=False, sheet_name="Sheet_Health")
            _format_columns(writer.sheets["Sheet_Health"], sheet_health_df, workbook)
        if chart_health_df is not None:
            chart_health_df.to_excel(writer, index=False, sheet_name="Chart_Health")
            _format_columns(writer.sheets["Chart_Health"], chart_health_df, workbook)

        preferred_order = [
            "Universe",
            "Universe_Review",
            "Archetypes",
            "Regime_State",
            "Market_Constraints",
            "Market_Constraints_Methodology",
            "Event_Episodes",
            "Confidence_Framework",
            "Confidence_Audit",
            "Peer_Baskets",
            "Peer_Basket_Membership",
            "Company_Case_Packet_Index",
            "Regression_Summary",
            "Rolling_Beta_Summary",
            "Event_Study_Summary",
            "Core_Ranking",
            "Extended_Ranking",
            "Route_Exposure_Build",
            "Chokepoint_Exposure",
            "Hormuz_Ranking",
            "Route_Risks",
            "Crude_Tracker",
            "Fuel_Tracker",
            "Fuel_Geography_Weights",
            "Equity_Tracker",
            "Factor_Decomposition",
            "Operating_Mix",
            "Valuation",
            "Scenario_Analysis",
            "Scenario_Event_Path",
            "Historical_Analogues",
            "Historical_Analogue_Company",
            "Catalyst_Calendar",
            "Catalyst_Calendar_Primary",
            "Catalyst_Calendar_Archive",
            "Earnings",
            "Analyst_Overrides",
            "Recommendation_Framework",
            "Company_Scorecards",
            "Company_Writeups",
            "Data_Quality",
            "Assumptions",
            "Source_Log",
            "Missing_Data_Log",
            "Section_Health",
            "Ranking_Health",
        ]

        sheet_health_map = {}
        if sheet_health_df is not None and not sheet_health_df.empty and "sheet_name" in sheet_health_df.columns:
            for _, r in sheet_health_df.iterrows():
                sheet_health_map[str(r["sheet_name"])] = {
                    "status": str(r.get("sheet_status", "POPULATED")),
                    "reason": str(r.get("reason", "")),
                }

        for name in preferred_order:
            if name not in datasets:
                continue
            df = datasets[name].copy()
            health = sheet_health_map.get(name, {"status": "POPULATED", "reason": ""})

            if health["status"] in {"UNAVAILABLE", "DEGRADED"}:
                write_df = _sheet_with_health(df, health["status"], health["reason"])
            else:
                write_df = df

            sheet = _sheet_name(name)
            write_df.to_excel(writer, index=False, sheet_name=sheet)
            _format_columns(writer.sheets[sheet], write_df, workbook)

            if "source_url" in write_df.columns:
                ws = writer.sheets[sheet]
                url_col = list(write_df.columns).index("source_url")
                for row_idx, val in enumerate(write_df["source_url"], start=1):
                    if isinstance(val, str) and val.startswith("http"):
                        ws.write_url(row_idx, url_col, val, string="link")

        for meta in chart_sheets:
            sheet_name = _sheet_name(meta["sheet_name"])
            ws = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = ws

            title_fmt = workbook.add_format({"bold": True, "font_size": 13})
            text_fmt = workbook.add_format({"text_wrap": True})

            ws.write("A1", "What this chart shows", title_fmt)
            ws.write("A2", meta.get("explanation", ""), text_fmt)
            ws.write("A3", "Key takeaway", title_fmt)
            ws.write("A4", meta.get("takeaway", ""), text_fmt)
            ws.write("A5", f"Chart status: {meta.get('status', 'UNKNOWN')}")
            if meta.get("reason"):
                ws.write("A6", f"Reason: {meta.get('reason')}", text_fmt)
            ws.set_column("A:A", 120)

            image_path = Path(meta["image_path"])
            if image_path.exists():
                ws.insert_image("A8", str(image_path), {"x_scale": 0.9, "y_scale": 0.9})

    return output_path
