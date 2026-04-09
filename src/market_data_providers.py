from __future__ import annotations

import io
import logging
from datetime import date
from typing import Callable

import pandas as pd
import requests

from src.fetch_diagnostics import FetchDiagnostics
from src.utils.yfinance_utils import download_history, flatten_history_to_price_frame

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 20


def _normalize_price_frame(df: pd.DataFrame, value_col: str = "price") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", value_col])

    out = df.copy()
    if "date" not in out.columns:
        if out.index.name:
            out = out.reset_index().rename(columns={out.index.name: "date"})
        else:
            out = out.reset_index().rename(columns={out.columns[0]: "date"})

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out = out.dropna(subset=["date"]).sort_values("date")

    if value_col not in out.columns:
        numeric_cols = [c for c in out.columns if c != "date" and pd.api.types.is_numeric_dtype(out[c])]
        if numeric_cols:
            out = out.rename(columns={numeric_cols[0]: value_col})
        else:
            out[value_col] = pd.NA

    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    return out[["date", value_col]].drop_duplicates(subset=["date"])


def _fetch_yfinance_series(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    hist = download_history(symbol, start_date, end_date, interval="1d")
    if hist.empty:
        return pd.DataFrame(columns=["date", "price"])
    return _normalize_price_frame(flatten_history_to_price_frame(hist, value_col="price"), value_col="price")


def _fetch_stooq_series(stooq_symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/"
    params = {"s": stooq_symbol.lower(), "i": "d"}
    response = requests.get(url, params=params, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()
    text = response.text.strip()
    if not text or "No data" in text:
        return pd.DataFrame(columns=["date", "price"])

    raw = pd.read_csv(io.StringIO(text))
    if raw.empty:
        return pd.DataFrame(columns=["date", "price"])

    col_map = {c.lower(): c for c in raw.columns}
    date_col = col_map.get("date")
    close_col = col_map.get("close")
    if not date_col or not close_col:
        return pd.DataFrame(columns=["date", "price"])

    frame = raw.rename(columns={date_col: "date", close_col: "price"})[["date", "price"]]
    frame = _normalize_price_frame(frame, value_col="price")
    mask = (frame["date"].dt.date >= start_date) & (frame["date"].dt.date <= end_date)
    return frame.loc[mask].reset_index(drop=True)


def _fetch_fred_series(series_id: str, start_date: date, end_date: date) -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {
        "id": series_id,
        "cosd": start_date.isoformat(),
        "coed": end_date.isoformat(),
    }
    response = requests.get(url, params=params, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()
    text = response.text.strip()
    if not text:
        return pd.DataFrame(columns=["date", "price"])

    raw = pd.read_csv(io.StringIO(text))
    if raw.empty or "DATE" not in raw.columns:
        return pd.DataFrame(columns=["date", "price"])

    value_col = [c for c in raw.columns if c != "DATE"]
    if not value_col:
        return pd.DataFrame(columns=["date", "price"])

    frame = raw.rename(columns={"DATE": "date", value_col[0]: "price"})[["date", "price"]]
    return _normalize_price_frame(frame, value_col="price")


def _stooq_symbol_for_ticker(ticker: str) -> str | None:
    t = str(ticker).strip()
    if not t:
        return None

    explicit = {
        "SPY": "spy.us",
        "XLE": "xle.us",
        "^TNX": "us10y",
        "DX-Y.NYB": "usdidx",
    }
    if t in explicit:
        return explicit[t]

    suffix_map = {
        ".L": ".uk",
        ".PA": ".fr",
        ".AS": ".nl",
        ".MI": ".it",
        ".TO": ".ca",
        ".AX": ".au",
        ".HK": ".hk",
        ".SA": ".br",
        ".ST": ".se",
    }

    for suffix, mapped in suffix_map.items():
        if t.upper().endswith(suffix):
            base = t[: -len(suffix)]
            return f"{base.lower()}{mapped}"

    if t.startswith("^"):
        return None

    if "." not in t:
        return f"{t.lower()}.us"

    return t.lower().replace(".", "")


def fetch_with_provider_chain(
    dataset: str,
    identifier: str,
    start_date: date,
    end_date: date,
    diagnostics: FetchDiagnostics | None,
    providers: list[dict[str, object]],
    min_points: int = 2,
) -> tuple[pd.DataFrame, str, str, bool]:
    first_provider = providers[0]["name"] if providers else "none"
    for idx, provider in enumerate(providers):
        provider_name = str(provider["name"])
        source_url = str(provider.get("source_url", ""))
        fetcher = provider.get("fetcher")
        if not callable(fetcher):
            continue

        try:
            result = fetcher(start_date, end_date)
            result = _normalize_price_frame(result, value_col="price")
            non_null = int(pd.to_numeric(result.get("price", pd.Series(dtype=float)), errors="coerce").notna().sum())
            rows = len(result)
            status = "success" if non_null >= min_points else "insufficient"
            msg = f"rows={rows} non_null={non_null}"
            if diagnostics:
                diagnostics.log_attempt(
                    dataset=dataset,
                    identifier=identifier,
                    provider=provider_name,
                    status=status,
                    source_url=source_url,
                    message=msg,
                    row_count=rows,
                    non_null_count=non_null,
                )
            if status == "success":
                if diagnostics:
                    diagnostics.log_provider_usage(
                        dataset=dataset,
                        identifier=identifier,
                        selected_provider=provider_name,
                        fallback_used=(idx > 0),
                        status="success",
                        notes=f"selected with {rows} rows and {non_null} non-null points",
                    )
                return result, provider_name, source_url, (idx > 0)
        except Exception as exc:
            if diagnostics:
                diagnostics.log_attempt(
                    dataset=dataset,
                    identifier=identifier,
                    provider=provider_name,
                    status="failure",
                    source_url=source_url,
                    message=str(exc),
                    row_count=0,
                    non_null_count=0,
                )
            logger.debug("Provider %s failed for %s/%s: %s", provider_name, dataset, identifier, exc)

    if diagnostics:
        diagnostics.log_provider_usage(
            dataset=dataset,
            identifier=identifier,
            selected_provider="none",
            fallback_used=False,
            status="failure",
            notes="all providers failed or were insufficient",
        )

    return pd.DataFrame(columns=["date", "price"]), "none", "", False


def fetch_crude_series(
    benchmark: str,
    start_date: date,
    end_date: date,
    diagnostics: FetchDiagnostics | None = None,
) -> tuple[pd.DataFrame, str, str, bool]:
    key = benchmark.lower().strip()
    config = {
        "brent": {
            "yf": "BZ=F",
            "stooq": "bz.f",
            "fred": "DCOILBRENTEU",
            "yf_url": "https://finance.yahoo.com/quote/BZ%3DF",
        },
        "wti": {
            "yf": "CL=F",
            "stooq": "cl.f",
            "fred": "DCOILWTICO",
            "yf_url": "https://finance.yahoo.com/quote/CL%3DF",
        },
    }
    if key not in config:
        return pd.DataFrame(columns=["date", "price"]), "none", "", False

    conf = config[key]
    providers = [
        {
            "name": "yfinance",
            "source_url": conf["yf_url"],
            "fetcher": lambda s, e: _fetch_yfinance_series(conf["yf"], s, e),
        },
        {
            "name": "stooq",
            "source_url": f"https://stooq.com/q/d/l/?s={conf['stooq']}&i=d",
            "fetcher": lambda s, e: _fetch_stooq_series(conf["stooq"], s, e),
        },
        {
            "name": "fred",
            "source_url": f"https://fred.stlouisfed.org/series/{conf['fred']}",
            "fetcher": lambda s, e: _fetch_fred_series(conf["fred"], s, e),
        },
    ]
    return fetch_with_provider_chain(
        dataset="crude_benchmark",
        identifier=key,
        start_date=start_date,
        end_date=end_date,
        diagnostics=diagnostics,
        providers=providers,
        min_points=2,
    )


def fetch_fuel_series(
    fuel_kind: str,
    start_date: date,
    end_date: date,
    diagnostics: FetchDiagnostics | None = None,
) -> tuple[pd.DataFrame, str, str, bool]:
    key = fuel_kind.lower().strip()
    config = {
        "gasoline": {
            "yf": "RB=F",
            "stooq": "rb.f",
            "fred": "GASREGW",
            "yf_url": "https://finance.yahoo.com/quote/RB%3DF",
        },
        "diesel": {
            "yf": "HO=F",
            "stooq": "ho.f",
            "fred": "GASDESW",
            "yf_url": "https://finance.yahoo.com/quote/HO%3DF",
        },
    }
    if key not in config:
        return pd.DataFrame(columns=["date", "price"]), "none", "", False

    conf = config[key]
    providers = [
        {
            "name": "yfinance",
            "source_url": conf["yf_url"],
            "fetcher": lambda s, e: _fetch_yfinance_series(conf["yf"], s, e),
        },
        {
            "name": "stooq",
            "source_url": f"https://stooq.com/q/d/l/?s={conf['stooq']}&i=d",
            "fetcher": lambda s, e: _fetch_stooq_series(conf["stooq"], s, e),
        },
        {
            "name": "fred",
            "source_url": f"https://fred.stlouisfed.org/series/{conf['fred']}",
            "fetcher": lambda s, e: _fetch_fred_series(conf["fred"], s, e),
        },
    ]
    return fetch_with_provider_chain(
        dataset="fuel_benchmark",
        identifier=key,
        start_date=start_date,
        end_date=end_date,
        diagnostics=diagnostics,
        providers=providers,
        min_points=2,
    )


def fetch_equity_series(
    ticker: str,
    start_date: date,
    end_date: date,
    diagnostics: FetchDiagnostics | None = None,
) -> tuple[pd.DataFrame, str, str, bool]:
    stooq_symbol = _stooq_symbol_for_ticker(ticker)
    providers: list[dict[str, object]] = [
        {
            "name": "yfinance",
            "source_url": f"https://finance.yahoo.com/quote/{ticker}",
            "fetcher": lambda s, e: _fetch_yfinance_series(ticker, s, e),
        }
    ]
    if stooq_symbol:
        providers.append(
            {
                "name": "stooq",
                "source_url": f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d",
                "fetcher": lambda s, e: _fetch_stooq_series(stooq_symbol, s, e),
            }
        )

    return fetch_with_provider_chain(
        dataset="equity_price",
        identifier=ticker,
        start_date=start_date,
        end_date=end_date,
        diagnostics=diagnostics,
        providers=providers,
        min_points=2,
    )


def fetch_factor_series(
    factor_name: str,
    start_date: date,
    end_date: date,
    diagnostics: FetchDiagnostics | None = None,
) -> tuple[pd.DataFrame, str, str, bool]:
    cfg = {
        "market": {
            "symbol": "SPY",
            "stooq": "spy.us",
            "fred": None,
        },
        "energy": {
            "symbol": "XLE",
            "stooq": "xle.us",
            "fred": None,
        },
        "rates": {
            "symbol": "^TNX",
            "stooq": "us10y",
            "fred": "DGS10",
        },
        "fx": {
            "symbol": "DX-Y.NYB",
            "stooq": "usdidx",
            "fred": "DTWEXBGS",
        },
    }
    if factor_name not in cfg:
        return pd.DataFrame(columns=["date", "price"]), "none", "", False

    conf = cfg[factor_name]
    providers: list[dict[str, object]] = [
        {
            "name": "yfinance",
            "source_url": f"https://finance.yahoo.com/quote/{conf['symbol']}",
            "fetcher": lambda s, e: _fetch_yfinance_series(conf["symbol"], s, e),
        }
    ]
    if conf.get("stooq"):
        providers.append(
            {
                "name": "stooq",
                "source_url": f"https://stooq.com/q/d/l/?s={conf['stooq']}&i=d",
                "fetcher": lambda s, e: _fetch_stooq_series(str(conf["stooq"]), s, e),
            }
        )
    if conf.get("fred"):
        providers.append(
            {
                "name": "fred",
                "source_url": f"https://fred.stlouisfed.org/series/{conf['fred']}",
                "fetcher": lambda s, e: _fetch_fred_series(str(conf["fred"]), s, e),
            }
        )

    return fetch_with_provider_chain(
        dataset="factor_price",
        identifier=factor_name,
        start_date=start_date,
        end_date=end_date,
        diagnostics=diagnostics,
        providers=providers,
        min_points=2,
    )


def fetch_generic_market_symbol(
    symbol: str,
    start_date: date,
    end_date: date,
    diagnostics: FetchDiagnostics | None = None,
    dataset: str = "generic_market",
) -> tuple[pd.DataFrame, str, str, bool]:
    stooq_symbol = _stooq_symbol_for_ticker(symbol)
    providers: list[dict[str, object]] = [
        {
            "name": "yfinance",
            "source_url": f"https://finance.yahoo.com/quote/{symbol}",
            "fetcher": lambda s, e: _fetch_yfinance_series(symbol, s, e),
        }
    ]
    if stooq_symbol:
        providers.append(
            {
                "name": "stooq",
                "source_url": f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d",
                "fetcher": lambda s, e: _fetch_stooq_series(stooq_symbol, s, e),
            }
        )

    return fetch_with_provider_chain(
        dataset=dataset,
        identifier=symbol,
        start_date=start_date,
        end_date=end_date,
        diagnostics=diagnostics,
        providers=providers,
        min_points=2,
    )


def fetch_yahoo_quote_snapshot(ticker: str, diagnostics: FetchDiagnostics | None = None) -> tuple[dict[str, object], str]:
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    try:
        response = requests.get(url, params={"symbols": ticker}, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        result = payload.get("quoteResponse", {}).get("result", [])
        item = result[0] if result else {}
        if item:
            if diagnostics:
                diagnostics.log_attempt(
                    dataset="valuation_snapshot",
                    identifier=ticker,
                    provider="yahoo_quote_api",
                    status="success",
                    source_url=f"{url}?symbols={ticker}",
                    message="quote snapshot retrieved",
                    row_count=1,
                    non_null_count=sum(1 for v in item.values() if v is not None),
                )
                diagnostics.log_provider_usage(
                    dataset="valuation_snapshot",
                    identifier=ticker,
                    selected_provider="yahoo_quote_api",
                    fallback_used=True,
                    status="success",
                    notes="fallback quote endpoint",
                )
            return item, "yahoo_quote_api"
        if diagnostics:
            diagnostics.log_attempt(
                dataset="valuation_snapshot",
                identifier=ticker,
                provider="yahoo_quote_api",
                status="insufficient",
                source_url=f"{url}?symbols={ticker}",
                message="empty quote result",
                row_count=0,
                non_null_count=0,
            )
    except Exception as exc:
        if diagnostics:
            diagnostics.log_attempt(
                dataset="valuation_snapshot",
                identifier=ticker,
                provider="yahoo_quote_api",
                status="failure",
                source_url=f"{url}?symbols={ticker}",
                message=str(exc),
                row_count=0,
                non_null_count=0,
            )

    return {}, "none"


def fetch_yahoo_calendar_events(ticker: str, diagnostics: FetchDiagnostics | None = None) -> list[pd.Timestamp]:
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
    params = {"modules": "calendarEvents"}
    try:
        response = requests.get(url, params=params, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        result = payload.get("quoteSummary", {}).get("result", [])
        if not result:
            if diagnostics:
                diagnostics.log_attempt(
                    dataset="earnings_calendar",
                    identifier=ticker,
                    provider="yahoo_calendar_api",
                    status="insufficient",
                    source_url=f"{url}?modules=calendarEvents",
                    message="no calendar events in response",
                    row_count=0,
                    non_null_count=0,
                )
            return []

        events = result[0].get("calendarEvents", {}).get("earnings", {}).get("earningsDate", [])
        dates: list[pd.Timestamp] = []
        for e in events:
            raw = e.get("raw") if isinstance(e, dict) else None
            if raw is not None:
                dt = pd.to_datetime(raw, unit="s", errors="coerce")
                if pd.notna(dt):
                    dates.append(pd.Timestamp(dt).tz_localize(None))

        if diagnostics:
            diagnostics.log_attempt(
                dataset="earnings_calendar",
                identifier=ticker,
                provider="yahoo_calendar_api",
                status="success" if dates else "insufficient",
                source_url=f"{url}?modules=calendarEvents",
                message=f"dates={len(dates)}",
                row_count=len(dates),
                non_null_count=len(dates),
            )
            if dates:
                diagnostics.log_provider_usage(
                    dataset="earnings_calendar",
                    identifier=ticker,
                    selected_provider="yahoo_calendar_api",
                    fallback_used=True,
                    status="success",
                    notes="calendar events fallback",
                )

        return dates
    except Exception as exc:
        if diagnostics:
            diagnostics.log_attempt(
                dataset="earnings_calendar",
                identifier=ticker,
                provider="yahoo_calendar_api",
                status="failure",
                source_url=f"{url}?modules=calendarEvents",
                message=str(exc),
                row_count=0,
                non_null_count=0,
            )
        return []
