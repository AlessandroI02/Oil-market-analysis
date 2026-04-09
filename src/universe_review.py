from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_universe_review(
    reviewed_df: pd.DataFrame,
    included_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    debug_dir: Path,
) -> dict[str, Path]:
    debug_dir.mkdir(parents=True, exist_ok=True)

    review_csv = debug_dir / "company_universe_review.csv"
    review_xlsx = debug_dir / "company_universe_review.xlsx"
    included_csv = debug_dir / "included_companies.csv"
    rejected_csv = debug_dir / "rejected_companies.csv"

    reviewed_df.to_csv(review_csv, index=False)
    included_df.to_csv(included_csv, index=False)
    rejected_df.to_csv(rejected_csv, index=False)

    with pd.ExcelWriter(review_xlsx, engine="xlsxwriter") as writer:
        reviewed_df.to_excel(writer, index=False, sheet_name="Universe_Review")
        included_df.to_excel(writer, index=False, sheet_name="Included")
        rejected_df.to_excel(writer, index=False, sheet_name="Rejected")

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})
        text_fmt = workbook.add_format({"text_wrap": True})

        for sheet_name, frame in {
            "Universe_Review": reviewed_df,
            "Included": included_df,
            "Rejected": rejected_df,
        }.items():
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))
            for col_idx, col in enumerate(frame.columns):
                ws.write(0, col_idx, col, header_fmt)
                width = min(max(len(str(col)), 18), 48)
                ws.set_column(col_idx, col_idx, width, text_fmt)

    return {
        "company_universe_review.csv": review_csv,
        "company_universe_review.xlsx": review_xlsx,
        "included_companies.csv": included_csv,
        "rejected_companies.csv": rejected_csv,
    }
