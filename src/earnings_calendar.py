from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

from src.assumptions_registry import MissingDataLogger
from src.fetch_diagnostics import FetchDiagnostics
from src.market_data_providers import fetch_yahoo_calendar_events
from src.models import EarningsRecord
from src.source_logger import SourceLogger
from src.utils.dates import next_estimated_earnings_date

logger = logging.getLogger(__name__)


def _coerce_datetime(value) -> datetime | None:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        dt = ts.to_pydatetime()
        if dt.tzinfo is not None:
            dt = dt.astimezone(UTC).replace(tzinfo=None)
        return dt
    except Exception:
        return None


def _extract_upcoming_earnings_dates(ticker: str, fetch_diagnostics: FetchDiagnostics | None = None) -> tuple[list[datetime], str]:
    if yf is None:
        fallback = fetch_yahoo_calendar_events(ticker, diagnostics=fetch_diagnostics)
        return [d.to_pydatetime().replace(tzinfo=None) for d in fallback], "yahoo_calendar_api"

    dates: list[datetime] = []
    provider = "none"
    try:
        tk = yf.Ticker(ticker)
        df = tk.get_earnings_dates(limit=12)
        if df is not None and not df.empty:
            for idx in df.index:
                dt = pd.to_datetime(idx).to_pydatetime().replace(tzinfo=None)
                dates.append(dt)
            provider = "yfinance_get_earnings_dates"
            if fetch_diagnostics:
                fetch_diagnostics.log_attempt(
                    dataset="earnings_calendar",
                    identifier=ticker,
                    provider="yfinance_get_earnings_dates",
                    status="success",
                    source_url=f"https://finance.yahoo.com/quote/{ticker}",
                    message=f"dates={len(dates)}",
                    row_count=len(dates),
                    non_null_count=len(dates),
                )
    except Exception as exc:
        if fetch_diagnostics:
            fetch_diagnostics.log_attempt(
                dataset="earnings_calendar",
                identifier=ticker,
                provider="yfinance_get_earnings_dates",
                status="failure",
                source_url=f"https://finance.yahoo.com/quote/{ticker}",
                message=str(exc),
                row_count=0,
                non_null_count=0,
            )

    if not dates:
        try:
            tk = yf.Ticker(ticker)
            cal = tk.calendar
            if cal is not None:
                if isinstance(cal, pd.DataFrame) and not cal.empty:
                    vals: list[object] = []
                    if "Earnings Date" in cal.index:
                        vals.extend(cal.loc["Earnings Date"].tolist())
                    if "Earnings Date" in cal.columns:
                        vals.extend(cal["Earnings Date"].tolist())
                    for val in vals:
                        dt = _coerce_datetime(val)
                        if dt is not None:
                            dates.append(dt)
                elif isinstance(cal, dict):
                    raw = cal.get("Earnings Date")
                    if isinstance(raw, (list, tuple)):
                        for val in raw:
                            dt = _coerce_datetime(val)
                            if dt is not None:
                                dates.append(dt)
                    else:
                        dt = _coerce_datetime(raw)
                        if dt is not None:
                            dates.append(dt)
                if dates:
                    provider = "yfinance_calendar"
        except Exception:
            pass

    if not dates:
        fallback = fetch_yahoo_calendar_events(ticker, diagnostics=fetch_diagnostics)
        if fallback:
            dates.extend([d.to_pydatetime().replace(tzinfo=None) for d in fallback])
            provider = "yahoo_calendar_api"

    if not dates:
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            for key in ("earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd"):
                raw = info.get(key)
                if raw is None:
                    continue
                dt = _coerce_datetime(raw if isinstance(raw, str) else datetime.fromtimestamp(raw, UTC))
                if dt is not None:
                    dates.append(dt)
            if dates:
                provider = "yfinance_info_timestamps"
        except Exception:
            pass

    dates = sorted(set(dates))
    return dates, provider


def build_earnings_calendar(
    included_df: pd.DataFrame,
    source_logger: SourceLogger,
    missing_logger: MissingDataLogger,
    fetch_diagnostics: FetchDiagnostics | None = None,
) -> pd.DataFrame:
    today = datetime.now(UTC).replace(tzinfo=None)
    rows: list[EarningsRecord] = []

    for _, row in included_df.iterrows():
        company = row["company_name"]
        ticker = row["ticker"]
        source_url = f"https://finance.yahoo.com/quote/{ticker}"

        earnings_dates, provider = _extract_upcoming_earnings_dates(ticker, fetch_diagnostics=fetch_diagnostics)
        future_dates = [d for d in earnings_dates if d >= today]

        next_date = future_dates[0] if future_dates else None
        following_date = future_dates[1] if len(future_dates) > 1 else next_estimated_earnings_date(next_date)

        flag = "missing"
        notes = "No upcoming earnings date available"

        if next_date and len(future_dates) > 1:
            flag = "exact"
            notes = "Both next and following earnings dates available from market calendar feed."
        elif next_date and following_date:
            flag = "estimated"
            notes = "Following earnings date estimated using 90-day reporting cadence."
        elif next_date:
            flag = "exact"
            notes = "Next earnings date from feed; following date unavailable."

        if next_date is None:
            missing_logger.add(
                company=company,
                field_name="next_earnings_date",
                reason="Could not retrieve next earnings date from configured free market sources",
                attempted_sources=[
                    source_url,
                    f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=calendarEvents",
                ],
            )

        rows.append(
            EarningsRecord(
                company_name=company,
                ticker=ticker,
                next_earnings_date=next_date,
                following_earnings_date=following_date,
                source_url=source_url,
                source_date=datetime.now(UTC).replace(tzinfo=None),
                exact_vs_estimated=flag,
                notes=notes,
            )
        )

        source_logger.add(
            company=company,
            field="earnings_dates",
            source_url=source_url,
            source_tier="Tier 3",
            evidence_flag="exact" if flag == "exact" else ("estimated" if flag == "estimated" else "missing"),
            comments=f"{notes} Provider={provider}",
        )

        if fetch_diagnostics:
            fetch_diagnostics.log_provider_usage(
                dataset="earnings_calendar",
                identifier=ticker,
                selected_provider=provider,
                fallback_used=provider in {"yahoo_calendar_api", "yfinance_calendar", "yfinance_info_timestamps"},
                status="success" if next_date is not None else "failure",
                notes=f"flag={flag}",
            )

    out = pd.DataFrame([r.model_dump() for r in rows])
    logger.info("Built earnings calendar rows: %s", len(out))
    return out
