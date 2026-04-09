from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def _date_arg(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Integrated oil Hormuz exposure research pipeline",
    )
    parser.add_argument("--start-date", type=_date_arg, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=_date_arg, help="End date (YYYY-MM-DD)")
    parser.add_argument("--lookback-months", type=int, default=None, help="Lookback window in months")
    parser.add_argument("--frequency", type=str, default=None, help="Pandas frequency string (default W-FRI)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output root directory")
    parser.add_argument("--max-companies", type=int, default=None, help="Limit number of included companies")
    parser.add_argument("--rebuild-cache", action="store_true", help="Ignore existing cache files")
    parser.add_argument("--skip-word", action="store_true", help="Skip Word thesis generation")
    parser.add_argument("--skip-excel", action="store_true", help="Skip Excel workbook generation")
    parser.add_argument("--only-universe", action="store_true", help="Run only universe build stage")
    parser.add_argument("--only-review", action="store_true", help="Run only universe review exports")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def parse_args():
    parser = build_parser()
    return parser.parse_args()
