from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

from src.assumptions_registry import MissingDataLogger
from src.source_logger import SourceLogger

logger = logging.getLogger(__name__)


def _quality_from_days(days: int) -> str:
    if days <= 21:
        return "Strong"
    if days <= 60:
        return "Medium"
    return "Weak"


def _confidence_label(raw: str) -> str:
    text = str(raw).lower()
    if text == "exact":
        return "High"
    if text in {"estimated", "proxy"}:
        return "Medium"
    return "Low"


def _event_relevance_score(event: str, event_type: str, confidence: str, lag_days: int, congestion: str) -> float:
    txt = str(event).lower()
    etype = str(event_type).lower()
    conf = str(confidence).lower()
    congestion_text = str(congestion).lower()

    base = {
        "macro": 78.0,
        "rates": 76.0,
        "company": 68.0,
        "capital_return": 38.0,
    }.get(etype, 52.0)

    if "earnings" in txt and "following" not in txt:
        base += 12.0
    if "following earnings" in txt:
        base -= 22.0
    if any(k in txt for k in ["opec", "fomc", "sanction", "hormuz", "investor day", "cmd", "guidance"]):
        base += 14.0
    if any(k in txt for k in ["dividend", "ex-dividend", "buyback"]):
        base -= 12.0

    if lag_days < 0:
        base -= 45.0
    elif lag_days <= 30:
        base += 10.0
    elif lag_days <= 90:
        base += 4.0
    elif lag_days > 180:
        base -= 12.0

    if conf == "exact":
        base += 6.0
    elif conf == "estimated":
        base -= 8.0
    elif conf == "proxy":
        base -= 12.0
    else:
        base -= 14.0

    if congestion_text == "high":
        base -= 4.0
    elif congestion_text == "medium":
        base -= 1.5

    return float(max(0.0, min(base, 100.0)))


def _relevance_tier(score: float) -> str:
    if score >= 75:
        return "high_relevance_future"
    if score >= 55:
        return "medium_relevance_future"
    return "low_relevance_archive"


def _is_thesis_relevant_event(event: str) -> bool:
    txt = str(event).lower()
    return any(
        key in txt
        for key in [
            "earnings",
            "opec",
            "fomc",
            "sanction",
            "hormuz",
            "investor day",
            "cmd",
            "guidance",
            "production",
            "outage",
            "policy",
        ]
    )


def build_catalyst_calendar(
    included_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
    config_path: Path,
    source_logger: SourceLogger,
    missing_logger: MissingDataLogger,
) -> pd.DataFrame:
    now = datetime.now(UTC).replace(tzinfo=None)
    rows: list[dict[str, Any]] = []

    # Company earnings-driven catalysts
    if not earnings_df.empty:
        for _, r in earnings_df.iterrows():
            ticker = r["ticker"]
            company = r["company_name"]
            next_dt = pd.to_datetime(r.get("next_earnings_date"), errors="coerce")
            next2_dt = pd.to_datetime(r.get("following_earnings_date"), errors="coerce")

            if pd.notna(next_dt):
                days = int((next_dt.to_pydatetime().replace(tzinfo=None) - now).days)
                rows.append(
                    {
                        "entity": company,
                        "ticker": ticker,
                        "event": "Earnings",
                        "event_date": next_dt,
                        "event_type": "company",
                        "near_term_event_support": _quality_from_days(max(days, 0)),
                        "timing_advantage": "High" if days <= 21 else "Moderate",
                        "event_congestion": "Low",
                        "notes": r.get("notes", ""),
                        "source": r.get("source_url", ""),
                        "confidence": r.get("exact_vs_estimated", "missing"),
                    }
                )

            if pd.notna(next2_dt):
                days2 = int((next2_dt.to_pydatetime().replace(tzinfo=None) - now).days)
                rows.append(
                    {
                        "entity": company,
                        "ticker": ticker,
                        "event": "Following Earnings",
                        "event_date": next2_dt,
                        "event_type": "company",
                        "near_term_event_support": _quality_from_days(max(days2, 0)),
                        "timing_advantage": "Moderate",
                        "event_congestion": "Low",
                        "notes": "Second earnings date from cadence/market calendar",
                        "source": r.get("source_url", ""),
                        "confidence": "estimated" if r.get("exact_vs_estimated") != "exact" else "exact",
                    }
                )

    # Ex-dividend and corporate return-related catalysts where available.
    if yf is not None:
        for _, row in included_df.iterrows():
            ticker = row["ticker"]
            company = row["company_name"]
            try:
                tk = yf.Ticker(ticker)
                info = tk.info or {}
                ex_div = info.get("exDividendDate")
                if ex_div:
                    ex_div_dt = datetime.fromtimestamp(ex_div, UTC).replace(tzinfo=None)
                    rows.append(
                        {
                            "entity": company,
                            "ticker": ticker,
                            "event": "Ex-Dividend",
                            "event_date": ex_div_dt,
                            "event_type": "capital_return",
                            "near_term_event_support": _quality_from_days(max((ex_div_dt - now).days, 0)),
                            "timing_advantage": "Moderate",
                            "event_congestion": "Low",
                            "notes": "Ex-dividend date from market data feed",
                            "source": f"https://finance.yahoo.com/quote/{ticker}",
                            "confidence": "exact",
                        }
                    )
            except Exception:
                missing_logger.add(
                    company=company,
                    field_name="ex_dividend_date",
                    reason="Ex-dividend date unavailable",
                    attempted_sources=[f"https://finance.yahoo.com/quote/{ticker}"],
                    severity="low",
                )

    # Macro events from config
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        for ev in cfg.get("macro_events", []):
            dt = pd.to_datetime(ev.get("date"), errors="coerce")
            if pd.notna(dt):
                rows.append(
                    {
                        "entity": "GLOBAL",
                        "ticker": "GLOBAL",
                        "event": ev.get("event", "Macro Event"),
                        "event_date": dt,
                        "event_type": ev.get("type", "macro"),
                        "near_term_event_support": _quality_from_days(max((dt.to_pydatetime().replace(tzinfo=None) - now).days, 0)),
                        "timing_advantage": "High",
                        "event_congestion": "Low",
                        "notes": ev.get("notes", ""),
                        "source": "config/scenario_settings.yaml",
                        "confidence": "estimated",
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "entity",
                "ticker",
                "event",
                "event_date",
                "event_type",
                "near_term_event_support",
                "timing_advantage",
                "event_congestion",
                "notes",
                "source",
                "confidence",
            ]
        )

    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    out = out.sort_values("event_date")

    out["event_week"] = out["event_date"].dt.to_period("W").astype(str)
    week_counts = out.groupby("event_week")["event"].transform("count")
    out["event_congestion"] = week_counts.apply(lambda x: "High" if x >= 6 else ("Medium" if x >= 3 else "Low"))

    for _, r in out.iterrows():
        source_logger.add(
            company=str(r["entity"]),
            field="catalyst_event",
            source_url=str(r["source"]),
            source_tier="Tier 2" if str(r["entity"]) == "GLOBAL" else "Tier 3",
            evidence_flag="exact" if str(r["confidence"]) == "exact" else "estimated",
            comments=str(r["event"]),
        )

    logger.info("Built catalyst calendar rows: %s", len(out))
    return out


def split_catalyst_calendar(
    catalyst_df: pd.DataFrame,
    as_of_date: date | None = None,
    horizon_days: int = 180,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if catalyst_df is None or catalyst_df.empty:
        empty = pd.DataFrame(columns=catalyst_df.columns if isinstance(catalyst_df, pd.DataFrame) else [])
        return empty, empty

    as_of = as_of_date or datetime.now(UTC).date()
    horizon_end = as_of + timedelta(days=horizon_days)

    out = catalyst_df.copy()
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    out = out.dropna(subset=["event_date"]).sort_values("event_date")
    out["_row_id"] = range(len(out))
    out["event_lag_days"] = (out["event_date"].dt.date - as_of).apply(lambda x: x.days)

    out["market_relevance_score"] = out.apply(
        lambda r: _event_relevance_score(
            event=str(r.get("event", "")),
            event_type=str(r.get("event_type", "")),
            confidence=str(r.get("confidence", "")),
            lag_days=int(r.get("event_lag_days", 9999)),
            congestion=str(r.get("event_congestion", "")),
        ),
        axis=1,
    )
    out["market_relevance_tier"] = out["market_relevance_score"].apply(_relevance_tier)
    out["catalyst_confidence"] = out.get("confidence", pd.Series(dtype=str)).astype(str).apply(_confidence_label)
    out["is_cadence_estimated"] = out.get("event", pd.Series(dtype=str)).astype(str).str.lower().str.contains("following earnings") | (
        out.get("notes", pd.Series(dtype=str)).astype(str).str.lower().str.contains("cadence")
    )

    future_mask = (out["event_date"].dt.date >= as_of) & (out["event_date"].dt.date <= horizon_end)
    out["thesis_relevant_flag"] = out.get("event", pd.Series(dtype=str)).astype(str).apply(_is_thesis_relevant_event)
    out["company_specific_flag"] = out.get("ticker", pd.Series(dtype=str)).astype(str).ne("GLOBAL")

    event_type = out.get("event_type", pd.Series(dtype=str)).astype(str)
    confidence = out.get("catalyst_confidence", pd.Series(dtype=str)).astype(str)
    lag_days = pd.to_numeric(out.get("event_lag_days"), errors="coerce").fillna(9999)
    relevance = pd.to_numeric(out.get("market_relevance_score"), errors="coerce").fillna(0.0)

    global_keep = (out["ticker"] == "GLOBAL") & (relevance >= 65.0)
    company_high_keep = (
        out["company_specific_flag"]
        & (event_type == "company")
        & (relevance >= 60.0)
        & (lag_days <= 120)
        & (confidence.isin(["High", "Medium"]))
    )
    company_cadence_keep = (
        out["company_specific_flag"]
        & (event_type == "company")
        & (out["is_cadence_estimated"])
        & (lag_days <= 45)
        & (relevance >= 56.0)
        & (out["thesis_relevant_flag"])
    )
    capital_return_keep = (
        (event_type == "capital_return")
        & (relevance >= 74.0)
        & (lag_days <= 45)
        & (confidence == "High")
    )

    primary_mask = future_mask & (global_keep | company_high_keep | company_cadence_keep | capital_return_keep)
    primary_candidates = out[primary_mask].copy()
    primary_candidates = primary_candidates.sort_values(["market_relevance_score", "event_date"], ascending=[False, True])

    if not primary_candidates.empty:
        primary_candidates["ticker_rank"] = primary_candidates.groupby("ticker").cumcount() + 1
        global_cap_mask = (primary_candidates["ticker"] == "GLOBAL") & (primary_candidates["ticker_rank"] <= 4)
        company_cap_mask = (primary_candidates["ticker"] != "GLOBAL") & (primary_candidates["ticker_rank"] <= 1)
        primary = primary_candidates[global_cap_mask | company_cap_mask].copy()
    else:
        primary = primary_candidates.copy()

    if primary.empty:
        fallback = out[future_mask & out["thesis_relevant_flag"]].sort_values(
            ["market_relevance_score", "event_date"],
            ascending=[False, True],
        )
        primary = fallback.head(8).copy()

    primary = primary.sort_values(["market_relevance_score", "event_date"], ascending=[False, True]).reset_index(drop=True)
    primary = primary.head(30).copy()
    archive = out[~out["_row_id"].isin(primary.get("_row_id", pd.Series(dtype=int)).tolist())].copy()
    archive = archive.sort_values(["event_date", "market_relevance_score"], ascending=[True, False]).reset_index(drop=True)

    primary["priority_rank"] = range(1, len(primary) + 1)
    archive["priority_rank"] = range(1, len(archive) + 1)
    primary["selection_reason"] = primary.apply(
        lambda r: (
            "global_market_catalyst"
            if str(r.get("ticker")) == "GLOBAL"
            else (
                "company_catalyst_high_relevance"
                if float(r.get("market_relevance_score", 0.0)) >= 60.0
                else "company_catalyst_near_term_cadence_estimate"
            )
        ),
        axis=1,
    )
    archive["selection_reason"] = "archive_or_lower_priority"

    primary["catalyst_bucket"] = "primary_future"
    archive["catalyst_bucket"] = "archive_secondary"

    primary = primary.drop(columns=["_row_id", "ticker_rank"], errors="ignore")
    archive = archive.drop(columns=["_row_id", "ticker_rank"], errors="ignore")

    return primary, archive
