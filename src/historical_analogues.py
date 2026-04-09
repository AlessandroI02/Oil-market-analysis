from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.assumptions_registry import MissingDataLogger
from src.fetch_diagnostics import FetchDiagnostics
from src.market_data_providers import fetch_generic_market_symbol
from src.source_logger import SourceLogger
from src.utils.yfinance_utils import first_valid

logger = logging.getLogger(__name__)


def _period_return(
    symbol: str,
    start: str,
    end: str,
    fetch_diagnostics: FetchDiagnostics | None,
    dataset: str,
) -> tuple[float | None, str, str]:
    frame, provider, source_url, _ = fetch_generic_market_symbol(
        symbol=symbol,
        start_date=pd.to_datetime(start).date(),
        end_date=pd.to_datetime(end).date(),
        diagnostics=fetch_diagnostics,
        dataset=dataset,
    )
    if frame.empty:
        return None, provider, source_url

    frame = frame.rename(columns={"price": "px"})
    first = first_valid(frame["px"])
    last = frame["px"].dropna().iloc[-1] if frame["px"].notna().any() else None
    if first is None or last is None or first == 0:
        return None, provider, source_url
    return float((last / first - 1) * 100), provider, source_url


def build_historical_analogues(
    included_df: pd.DataFrame,
    config_path: Path,
    source_logger: SourceLogger | None = None,
    missing_logger: MissingDataLogger | None = None,
    fetch_diagnostics: FetchDiagnostics | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    periods = cfg.get("analogue_periods", [])
    summary_rows: list[dict[str, Any]] = []
    company_rows: list[dict[str, Any]] = []

    for p in periods:
        name = p.get("name")
        start = p.get("start")
        end = p.get("end")
        context = p.get("context", "")

        brent, brent_provider, brent_url = _period_return("BZ=F", start, end, fetch_diagnostics, "historical_analogue_brent")
        xle, xle_provider, xle_url = _period_return("XLE", start, end, fetch_diagnostics, "historical_analogue_xle")
        spy, spy_provider, spy_url = _period_return("SPY", start, end, fetch_diagnostics, "historical_analogue_spy")

        if source_logger and brent is not None:
            source_logger.add(
                company="GLOBAL",
                field=f"historical_analogue_brent::{name}",
                source_url=brent_url or "https://finance.yahoo.com/quote/BZ%3DF",
                source_tier="Tier 3",
                evidence_flag="exact",
                comments=f"Provider={brent_provider}",
            )

        period_company_returns: list[float] = []
        period_company_records: list[dict[str, Any]] = []

        for _, row in included_df.iterrows():
            ticker = row["ticker"]
            ret, provider, src_url = _period_return(ticker, start, end, fetch_diagnostics, "historical_analogue_company")
            period_company_records.append(
                {
                    "period": name,
                    "start": start,
                    "end": end,
                    "company_name": row["company_name"],
                    "ticker": ticker,
                    "equity_return_pct": ret,
                    "provider": provider,
                }
            )
            if ret is not None:
                period_company_returns.append(ret)
            elif missing_logger is not None:
                missing_logger.add(
                    company=row["company_name"],
                    field_name=f"historical_analogue_return::{name}",
                    reason=f"No market data for {ticker} in analogue period",
                    attempted_sources=[f"https://finance.yahoo.com/quote/{ticker}", "https://stooq.com/"],
                    severity="medium",
                )

            if source_logger and ret is not None:
                source_logger.add(
                    company=row["company_name"],
                    field=f"historical_analogue_return::{name}",
                    source_url=src_url or f"https://finance.yahoo.com/quote/{ticker}",
                    source_tier="Tier 3",
                    evidence_flag="exact",
                    comments=f"Provider={provider}",
                )

        company_rows.extend(period_company_records)

        median_return = float(pd.Series(period_company_returns).median()) if period_company_returns else None
        top_ticker = None
        bottom_ticker = None
        if period_company_records:
            tmp = pd.DataFrame(period_company_records)
            tmp = tmp.dropna(subset=["equity_return_pct"])
            if not tmp.empty:
                top_ticker = tmp.sort_values("equity_return_pct", ascending=False).iloc[0]["ticker"]
                bottom_ticker = tmp.sort_values("equity_return_pct", ascending=True).iloc[0]["ticker"]

        summary_rows.append(
            {
                "period": name,
                "start": start,
                "end": end,
                "context": context,
                "brent_return_pct": brent,
                "xle_return_pct": xle,
                "spy_return_pct": spy,
                "brent_provider": brent_provider,
                "xle_provider": xle_provider,
                "spy_provider": spy_provider,
                "peer_median_return_pct": median_return,
                "top_outperformer": top_ticker,
                "bottom_underperformer": bottom_ticker,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    company_df = pd.DataFrame(company_rows)
    logger.info("Built historical analogues summary rows: %s, company rows: %s", len(summary_df), len(company_df))
    return summary_df, company_df
