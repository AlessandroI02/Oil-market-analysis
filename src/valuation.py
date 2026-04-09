from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

from src.assumptions_registry import AssumptionsRegistry, MissingDataLogger
from src.fetch_diagnostics import FetchDiagnostics
from src.market_data_providers import fetch_yahoo_quote_snapshot
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
    if currency in {"USD", "US$", "$"}:
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
    fx = first_valid(price["fx"]) if not price.empty else None
    return fx


def _to_usd(value: float | None, fx_to_usd: float | None) -> float | None:
    if value is None:
        return None
    if fx_to_usd is None:
        return None
    return float(value) * float(fx_to_usd)


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return float(num) / float(den)


def _extract_yf_payload(ticker: str, fetch_diagnostics: FetchDiagnostics | None) -> tuple[dict[str, Any], dict[str, Any], str]:
    info: dict[str, Any] = {}
    fast_info: dict[str, Any] = {}
    provider = "none"

    if yf is None:
        return info, fast_info, provider

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        provider = "yfinance_info"
        if fetch_diagnostics:
            fetch_diagnostics.log_attempt(
                dataset="valuation_snapshot",
                identifier=ticker,
                provider="yfinance_info",
                status="success" if info else "insufficient",
                source_url=f"https://finance.yahoo.com/quote/{ticker}",
                message=f"info_keys={len(info)}",
                row_count=1 if info else 0,
                non_null_count=sum(1 for v in info.values() if v is not None),
            )
    except Exception as exc:
        if fetch_diagnostics:
            fetch_diagnostics.log_attempt(
                dataset="valuation_snapshot",
                identifier=ticker,
                provider="yfinance_info",
                status="failure",
                source_url=f"https://finance.yahoo.com/quote/{ticker}",
                message=str(exc),
                row_count=0,
                non_null_count=0,
            )

    try:
        tk2 = yf.Ticker(ticker)
        fi = tk2.fast_info
        fast_info = dict(fi) if fi is not None else {}
        if fetch_diagnostics:
            fetch_diagnostics.log_attempt(
                dataset="valuation_snapshot",
                identifier=ticker,
                provider="yfinance_fast_info",
                status="success" if fast_info else "insufficient",
                source_url=f"https://finance.yahoo.com/quote/{ticker}",
                message=f"fast_info_keys={len(fast_info)}",
                row_count=1 if fast_info else 0,
                non_null_count=sum(1 for v in fast_info.values() if v is not None),
            )
    except Exception as exc:
        if fetch_diagnostics:
            fetch_diagnostics.log_attempt(
                dataset="valuation_snapshot",
                identifier=ticker,
                provider="yfinance_fast_info",
                status="failure",
                source_url=f"https://finance.yahoo.com/quote/{ticker}",
                message=str(exc),
                row_count=0,
                non_null_count=0,
            )

    return info, fast_info, provider


def build_valuation_table(
    included_df: pd.DataFrame,
    source_logger: SourceLogger,
    assumptions_registry: AssumptionsRegistry,
    missing_logger: MissingDataLogger,
    fetch_diagnostics: FetchDiagnostics | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in included_df.iterrows():
        company = row["company_name"]
        ticker = row["ticker"]
        source_url = f"https://finance.yahoo.com/quote/{ticker}"

        info, fast_info, provider = _extract_yf_payload(ticker, fetch_diagnostics)
        fallback_item: dict[str, Any] = {}

        if not info and not fast_info:
            fallback_item, fallback_provider = fetch_yahoo_quote_snapshot(ticker, fetch_diagnostics)
            if fallback_item:
                provider = fallback_provider

        currency = str(
            info.get("financialCurrency")
            or info.get("currency")
            or fallback_item.get("currency")
            or "USD"
        )
        fx_to_usd = _latest_fx_to_usd(currency)

        if fx_to_usd is None and currency != "USD":
            missing_logger.add(
                company=company,
                field_name="fx_to_usd",
                reason=f"Could not fetch FX conversion for currency {currency}",
                attempted_sources=["https://finance.yahoo.com/currencies"],
                severity="medium",
            )

        market_cap_raw = (
            info.get("marketCap")
            or fast_info.get("market_cap")
            or fallback_item.get("marketCap")
        )
        enterprise_value_raw = info.get("enterpriseValue") or fallback_item.get("enterpriseValue")
        ebitda_raw = info.get("ebitda") or fallback_item.get("ebitda")
        total_debt_raw = info.get("totalDebt")
        total_cash_raw = info.get("totalCash")
        free_cash_flow_raw = info.get("freeCashflow")
        trailing_pe = info.get("trailingPE") or fallback_item.get("trailingPE")
        forward_pe = info.get("forwardPE") or fallback_item.get("forwardPE")
        dividend_yield = info.get("dividendYield")

        shares = info.get("sharesOutstanding") or fallback_item.get("sharesOutstanding")
        price = fast_info.get("last_price") or fallback_item.get("regularMarketPrice")
        market_cap_estimated = False
        if market_cap_raw is None and shares is not None and price is not None:
            try:
                market_cap_raw = float(shares) * float(price)
                market_cap_estimated = True
            except Exception:
                market_cap_raw = None

        market_cap = _to_usd(float(market_cap_raw), fx_to_usd) if market_cap_raw is not None else None
        enterprise_value = _to_usd(float(enterprise_value_raw), fx_to_usd) if enterprise_value_raw is not None else None
        ebitda = _to_usd(float(ebitda_raw), fx_to_usd) if ebitda_raw is not None else None
        total_debt = _to_usd(float(total_debt_raw), fx_to_usd) if total_debt_raw is not None else None
        total_cash = _to_usd(float(total_cash_raw), fx_to_usd) if total_cash_raw is not None else None
        free_cash_flow = _to_usd(float(free_cash_flow_raw), fx_to_usd) if free_cash_flow_raw is not None else None

        net_debt = (total_debt - total_cash) if (total_debt is not None and total_cash is not None) else None
        if enterprise_value is None and market_cap is not None and net_debt is not None:
            enterprise_value = market_cap + net_debt

        ev_ebitda = _safe_div(enterprise_value, ebitda)
        leverage_ratio = _safe_div(net_debt, ebitda)

        fcf_yield_pct = _safe_div(free_cash_flow, market_cap)
        if fcf_yield_pct is not None:
            fcf_yield_pct *= 100

        div_yield_pct = (float(dividend_yield) * 100) if dividend_yield is not None else None

        buyback_yield_pct = None
        payout_ratio = info.get("payoutRatio")
        if payout_ratio is not None and div_yield_pct is not None:
            buyback_yield_pct = max(float(payout_ratio) * 100 - div_yield_pct, 0.0)
            assumptions_registry.add(
                field_name="buyback_yield_pct",
                company=company,
                estimate_value=f"{buyback_yield_pct:.2f}",
                estimate_type="proxy_estimate",
                reasoning="Estimated from payout ratio net of dividend yield where direct buyback yield not disclosed.",
                source_urls=[source_url],
                confidence="Low",
                model_version="2.1",
            )

        payout_framework = "Dividend + buyback" if buyback_yield_pct is not None and buyback_yield_pct > 0 else "Dividend-led"

        data_flag = "exact"
        if market_cap is None:
            data_flag = "missing"
        elif market_cap_estimated:
            data_flag = "estimated"

        if market_cap_estimated:
            assumptions_registry.add(
                field_name="market_cap_usd",
                company=company,
                estimate_value=f"{market_cap:.2f}" if market_cap is not None else "",
                estimate_type="proxy_estimate",
                reasoning="Market cap estimated from shares outstanding and latest price because direct market-cap field unavailable.",
                source_urls=[source_url],
                confidence="Low",
                model_version="2.1",
            )

        row_out = {
            "company_name": company,
            "ticker": ticker,
            "currency_raw": currency,
            "fx_to_usd": fx_to_usd,
            "market_cap_usd": market_cap,
            "enterprise_value_usd": enterprise_value,
            "net_debt_usd": net_debt,
            "ev_ebitda": ev_ebitda,
            "pe_ratio": float(trailing_pe) if trailing_pe is not None else (float(forward_pe) if forward_pe is not None else None),
            "forward_pe": float(forward_pe) if forward_pe is not None else None,
            "fcf_yield_pct": fcf_yield_pct,
            "dividend_yield_pct": div_yield_pct,
            "buyback_yield_pct": buyback_yield_pct,
            "leverage_ratio": leverage_ratio,
            "payout_framework": payout_framework,
            "source_url": source_url,
            "source_tier": "Tier 3",
            "provider_used": provider,
            "data_flag": data_flag,
        }

        if market_cap is None:
            missing_logger.add(
                company=company,
                field_name="market_cap_usd",
                reason="Market cap unavailable from configured public providers",
                attempted_sources=[source_url, "https://query1.finance.yahoo.com/v7/finance/quote"],
                severity="high",
            )

        rows.append(row_out)

        source_logger.add(
            company=company,
            field="valuation_snapshot",
            source_url=source_url,
            source_tier="Tier 3",
            evidence_flag="exact" if data_flag == "exact" else ("estimated" if data_flag == "estimated" else "missing"),
            comments=f"Provider={provider}",
        )

        if fetch_diagnostics:
            fetch_diagnostics.log_provider_usage(
                dataset="valuation_snapshot",
                identifier=ticker,
                selected_provider=provider,
                fallback_used=(provider == "yahoo_quote_api"),
                status="success" if market_cap is not None else "failure",
                notes=f"data_flag={data_flag}",
            )

    out = pd.DataFrame(rows)

    if not out.empty:
        for col in ["ev_ebitda", "pe_ratio", "fcf_yield_pct", "dividend_yield_pct", "leverage_ratio"]:
            series = pd.to_numeric(out[col], errors="coerce")
            out[f"{col}_percentile"] = series.rank(pct=True)

        out["valuation_score"] = (
            (1 - out["ev_ebitda_percentile"].fillna(0.5)) * 0.30
            + (1 - out["pe_ratio_percentile"].fillna(0.5)) * 0.25
            + out["fcf_yield_pct_percentile"].fillna(0.5) * 0.20
            + out["dividend_yield_pct_percentile"].fillna(0.5) * 0.10
            + (1 - out["leverage_ratio_percentile"].fillna(0.5)) * 0.15
        ) * 100

        peer_median_ev_ebitda = pd.to_numeric(out["ev_ebitda"], errors="coerce").median()
        out["valuation_vs_peers"] = out["ev_ebitda"].apply(
            lambda x: "Discount"
            if pd.notna(x) and pd.notna(peer_median_ev_ebitda) and x < peer_median_ev_ebitda
            else "Premium"
        )

    logger.info("Built valuation table rows: %s", len(out))
    return out
