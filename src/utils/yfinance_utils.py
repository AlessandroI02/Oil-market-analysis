from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

logger = logging.getLogger(__name__)


def configure_yfinance_cache(cache_dir: Path) -> bool:
    """Configure yfinance cache path to a writable project-local directory."""
    if yf is None:
        return False

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(cache_dir))
        try:
            from yfinance import cache as yf_cache

            if hasattr(yf_cache, "set_cache_location"):
                yf_cache.set_cache_location(str(cache_dir))
        except Exception:
            # Best effort: set_tz_cache_location above is available in public API.
            pass

        logger.info("Configured yfinance cache directory: %s", cache_dir)
        return True
    except Exception as exc:
        logger.warning("Failed to configure yfinance cache directory %s: %s", cache_dir, exc)
        return False


def _pick_price_series(hist: pd.DataFrame) -> Optional[pd.Series]:
    """Pick a single price series from a yfinance history dataframe."""
    if hist is None or hist.empty:
        return None

    # MultiIndex columns usually appear as (field, ticker)
    if isinstance(hist.columns, pd.MultiIndex):
        for preferred in ("Close", "Adj Close"):
            matches = [col for col in hist.columns if str(col[0]).strip().lower() == preferred.lower()]
            if matches:
                series = hist[matches[0]]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                return series

        numeric_cols = [col for col in hist.columns if pd.api.types.is_numeric_dtype(hist[col])]
        if numeric_cols:
            series = hist[numeric_cols[0]]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            return series
        return None

    for preferred in ("Close", "Adj Close"):
        if preferred in hist.columns:
            series = hist[preferred]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            return series

    numeric_cols = [col for col in hist.columns if pd.api.types.is_numeric_dtype(hist[col])]
    if numeric_cols:
        series = hist[numeric_cols[0]]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        return series

    return None


def flatten_history_to_price_frame(hist: pd.DataFrame, value_col: str = "close") -> pd.DataFrame:
    """Normalize history output to: date, <value_col> with tz-naive datetime."""
    series = _pick_price_series(hist)
    if series is None:
        return pd.DataFrame(columns=["date", value_col])

    out = series.to_frame(name=value_col)
    idx = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx
    out = out[~out.index.isna()].sort_index()
    out.index.name = "date"

    out = out.reset_index()[["date", value_col]]
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    return out


def download_history(symbol: str, start_date: date, end_date: date, interval: str = "1d") -> pd.DataFrame:
    """Download price history with yfinance using multiple internal paths."""
    if yf is None:
        return pd.DataFrame()

    # Path 1: yf.download
    try:
        hist = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )
        if hist is not None and not hist.empty:
            return hist
    except Exception as exc:
        logger.debug("yf.download failed for %s: %s", symbol, exc)

    # Path 2: Ticker.history
    try:
        tk = yf.Ticker(symbol)
        hist2 = tk.history(start=start_date, end=end_date, interval=interval, auto_adjust=False)
        if hist2 is not None and not hist2.empty:
            return hist2
    except Exception as exc:
        logger.debug("Ticker.history failed for %s: %s", symbol, exc)

    return pd.DataFrame()


def first_valid(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[0])
