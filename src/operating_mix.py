from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

from src.assumptions_registry import AssumptionsRegistry, MissingDataLogger
from src.models import OperatingMixRecord
from src.source_logger import SourceLogger
from src.utils.yfinance_utils import download_history, first_valid, flatten_history_to_price_frame

logger = logging.getLogger(__name__)

CURRENCY_PAIR_MAP = {
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "CAD": "CADUSD=X",
    "NOK": "NOKUSD=X",
    "CNY": "CNYUSD=X",
    "HKD": "HKDUSD=X",
    "SAR": "SARUSD=X",
    "PLN": "PLNUSD=X",
    "BRL": "BRLUSD=X",
    "CHF": "CHFUSD=X",
    "JPY": "JPYUSD=X",
}


def _latest_fx_to_usd(currency: str) -> float | None:
    if currency in {"USD", "$", "US$"}:
        return 1.0
    pair = CURRENCY_PAIR_MAP.get(currency)
    if pair is None:
        return None
    end_date = pd.Timestamp.today().date()
    start_date = end_date - timedelta(days=30)
    hist = download_history(pair, start_date=start_date, end_date=end_date, interval="1d")
    if hist.empty:
        return None
    price = flatten_history_to_price_frame(hist, value_col="fx")
    return first_valid(price["fx"]) if not price.empty else None


def _extract_total_revenue_from_frame(df: pd.DataFrame) -> float | None:
    if df is None or df.empty:
        return None
    try:
        idx_str = pd.Index([str(x).strip().lower() for x in df.index])
        target_positions = [i for i, name in enumerate(idx_str) if "total revenue" in name]
        if not target_positions:
            return None
        row = df.iloc[target_positions[0]]
        numeric = pd.to_numeric(row, errors="coerce").dropna()
        if numeric.empty:
            return None
        return float(numeric.iloc[0])
    except Exception:
        return None


def _fetch_total_revenue(ticker: str) -> float | None:
    if yf is None:
        return None
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        rev = info.get("totalRevenue")
        if rev is not None:
            return float(rev)

        for attr in ("financials", "income_stmt", "quarterly_financials", "quarterly_income_stmt"):
            try:
                frame = getattr(tk, attr, None)
                value = _extract_total_revenue_from_frame(frame)
                if value is not None:
                    return value
            except Exception:
                continue

        return None
    except Exception:
        return None


def _fetch_ticker_info(ticker: str) -> dict:
    if yf is None:
        return {}
    try:
        tk = yf.Ticker(ticker)
        return tk.info or {}
    except Exception:
        return {}


def build_operating_mix(
    profiles_df: pd.DataFrame,
    source_logger: SourceLogger,
    assumptions_registry: AssumptionsRegistry,
    missing_logger: MissingDataLogger,
) -> pd.DataFrame:
    records: list[OperatingMixRecord] = []

    for _, row in profiles_df.iterrows():
        company = row["company_name"]
        ticker = row["ticker"]

        upstream_share = float(row.get("upstream_mix_share") or 0.5)
        downstream_share = float(row.get("downstream_mix_share") or 0.5)

        info = _fetch_ticker_info(ticker)
        currency_raw = str(info.get("financialCurrency") or info.get("currency") or "USD")
        fx_to_usd = _latest_fx_to_usd(currency_raw)
        total_revenue = _fetch_total_revenue(ticker)
        if total_revenue is None:
            missing_logger.add(
                company=company,
                field_name="total_revenue",
                reason="Could not fetch total revenue from yfinance",
                attempted_sources=[f"https://finance.yahoo.com/quote/{ticker}/financials"],
            )
        if fx_to_usd is None and currency_raw != "USD":
            missing_logger.add(
                company=company,
                field_name="fx_to_usd",
                reason=f"Could not fetch FX conversion for operating mix currency {currency_raw}",
                attempted_sources=["https://finance.yahoo.com/currencies"],
                severity="medium",
            )

        upstream_revenue = total_revenue * upstream_share if total_revenue is not None else None
        downstream_revenue = total_revenue * downstream_share if total_revenue is not None else None
        total_revenue_usd = (total_revenue * fx_to_usd) if (total_revenue is not None and fx_to_usd is not None) else None
        upstream_revenue_usd = (upstream_revenue * fx_to_usd) if (upstream_revenue is not None and fx_to_usd is not None) else None
        downstream_revenue_usd = (downstream_revenue * fx_to_usd) if (downstream_revenue is not None and fx_to_usd is not None) else None

        upstream_volume = upstream_share * 100
        downstream_volume = downstream_share * 100

        data_flag = "proxy"
        notes = (
            "Revenue split estimated from normalized upstream/downstream mix shares; volume fields are index proxies (base=100) when exact segment volumes are not consistently disclosed. Revenue normalized to USD when FX is available."
        )

        if total_revenue is not None and total_revenue_usd is not None:
            data_flag = "estimated"
        elif total_revenue is not None:
            data_flag = "proxy"

        if total_revenue is not None:
            source_logger.add(
                company=company,
                field="total_revenue",
                source_url=f"https://finance.yahoo.com/quote/{ticker}/financials",
                source_tier="Tier 3",
                evidence_flag="proxy",
                comments="Total revenue from public market feed; segment mix analyst-normalized",
            )

        assumptions_registry.add(
            field_name="operating_mix_split",
            company=company,
            estimate_value=f"upstream={upstream_share:.2f}, downstream={downstream_share:.2f}",
            estimate_type="analyst_estimate",
            reasoning="Normalized segment mix to align cross-company comparability where reporting taxonomy differs.",
            source_urls=(row.get("source_links") or []),
            confidence=row.get("profile_confidence", "Low"),
            model_version="2.0",
        )

        records.append(
            OperatingMixRecord(
                company_name=company,
                ticker=ticker,
                currency_raw=currency_raw,
                fx_to_usd=round(fx_to_usd, 6) if fx_to_usd is not None else None,
                upstream_volume=round(upstream_volume, 2),
                downstream_volume=round(downstream_volume, 2),
                upstream_revenue=round(upstream_revenue, 2) if upstream_revenue is not None else None,
                downstream_revenue=round(downstream_revenue, 2) if downstream_revenue is not None else None,
                total_revenue=round(total_revenue, 2) if total_revenue is not None else None,
                upstream_revenue_usd=round(upstream_revenue_usd, 2) if upstream_revenue_usd is not None else None,
                downstream_revenue_usd=round(downstream_revenue_usd, 2) if downstream_revenue_usd is not None else None,
                total_revenue_usd=round(total_revenue_usd, 2) if total_revenue_usd is not None else None,
                upstream_share_pct=round(upstream_share * 100, 2),
                downstream_share_pct=round(downstream_share * 100, 2),
                data_flag=data_flag,
                notes=notes,
            )
        )

    out = pd.DataFrame([r.model_dump() for r in records])
    logger.info("Built operating mix rows: %s", len(out))
    return out
