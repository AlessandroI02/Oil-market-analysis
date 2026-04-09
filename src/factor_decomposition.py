from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from src.assumptions_registry import MissingDataLogger
from src.fetch_diagnostics import FetchDiagnostics
from src.market_data_providers import fetch_factor_series
from src.source_logger import SourceLogger

logger = logging.getLogger(__name__)


def _load_weekly_factor(
    factor_name: str,
    start_date: date,
    end_date: date,
    frequency: str,
    value_name: str,
    source_logger: SourceLogger | None,
    missing_logger: MissingDataLogger | None,
    fetch_diagnostics: FetchDiagnostics | None,
) -> pd.DataFrame:
    daily, provider, source_url, fallback_used = fetch_factor_series(
        factor_name=factor_name,
        start_date=start_date,
        end_date=end_date,
        diagnostics=fetch_diagnostics,
    )
    if daily.empty:
        if missing_logger:
            missing_logger.add(
                company="GLOBAL",
                field_name=f"factor_{factor_name}",
                reason=f"No data for factor {factor_name} from configured providers",
                attempted_sources=[
                    f"https://finance.yahoo.com/quote/{factor_name}",
                    "https://stooq.com/",
                    "https://fred.stlouisfed.org/",
                ],
                severity="medium",
            )
        return pd.DataFrame(columns=["date", value_name])

    if source_logger:
        source_logger.add(
            company="GLOBAL",
            field=f"factor_{factor_name}",
            source_url=source_url or f"https://finance.yahoo.com/quote/{factor_name}",
            source_tier="Tier 2" if provider == "fred" else "Tier 3",
            evidence_flag="exact",
            comments=f"Provider={provider}; fallback_used={fallback_used}",
        )

    return (
        daily.rename(columns={"price": value_name})
        .assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce").dt.tz_localize(None))
        .dropna(subset=["date"])
        .set_index("date")
        .resample(frequency)
        .last()
        .reset_index()
    )


def _regression_betas(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> dict[str, float]:
    sub = df[[y_col] + x_cols].dropna()
    if len(sub) < max(8, len(x_cols) + 2):
        return {f"beta_{c}": np.nan for c in x_cols} | {"alpha": np.nan, "r2": np.nan}

    y = sub[y_col].to_numpy(dtype=float)
    x = sub[x_cols].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])

    coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ coeffs

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    out = {"alpha": float(coeffs[0]), "r2": float(r2)}
    for i, c in enumerate(x_cols, start=1):
        out[f"beta_{c}"] = float(coeffs[i])
    return out


def build_factor_decomposition(
    equity_tracker_df: pd.DataFrame,
    crude_tracker_df: pd.DataFrame,
    start_date: date,
    end_date: date,
    frequency: str,
    source_logger: SourceLogger | None = None,
    missing_logger: MissingDataLogger | None = None,
    fetch_diagnostics: FetchDiagnostics | None = None,
) -> pd.DataFrame:
    if equity_tracker_df is None or equity_tracker_df.empty:
        return pd.DataFrame(
            columns=[
                "company_name",
                "ticker",
                "beta_brent_ret",
                "beta_market_ret",
                "beta_energy_ret",
                "beta_rates_ret",
                "beta_fx_ret",
                "alpha",
                "r2",
                "idiosyncratic_residual_pct",
            ]
        )

    market = _load_weekly_factor("market", start_date, end_date, frequency, "market", source_logger, missing_logger, fetch_diagnostics)
    energy = _load_weekly_factor("energy", start_date, end_date, frequency, "energy", source_logger, missing_logger, fetch_diagnostics)
    rates = _load_weekly_factor("rates", start_date, end_date, frequency, "rates", source_logger, missing_logger, fetch_diagnostics)
    fx = _load_weekly_factor("fx", start_date, end_date, frequency, "fx", source_logger, missing_logger, fetch_diagnostics)

    if crude_tracker_df is None or crude_tracker_df.empty or "date" not in crude_tracker_df.columns:
        factor = pd.DataFrame(columns=["date", "brent"])
    else:
        base_cols = ["date", "brent_price"] if "brent_price" in crude_tracker_df.columns else ["date"]
        factor = crude_tracker_df[base_cols].copy()
        if "brent_price" not in factor.columns:
            factor["brent_price"] = np.nan
    factor = factor.rename(columns={"brent_price": "brent"})
    factor = factor.merge(market, on="date", how="left")
    factor = factor.merge(energy, on="date", how="left")
    factor = factor.merge(rates, on="date", how="left")
    factor = factor.merge(fx, on="date", how="left")

    for col in ["brent", "market", "energy", "rates", "fx"]:
        factor[f"{col}_ret"] = pd.to_numeric(factor[col], errors="coerce").pct_change()

    rows: list[dict[str, object]] = []
    x_cols = ["brent_ret", "market_ret", "energy_ret", "rates_ret", "fx_ret"]

    required_equity_cols = {"company_name", "ticker", "date", "share_price"}
    if not required_equity_cols.issubset(set(equity_tracker_df.columns)):
        return pd.DataFrame(
            columns=[
                "company_name",
                "ticker",
                "beta_brent_ret",
                "beta_market_ret",
                "beta_energy_ret",
                "beta_rates_ret",
                "beta_fx_ret",
                "alpha",
                "r2",
                "idiosyncratic_residual_pct",
            ]
        )

    for (company, ticker), g in equity_tracker_df.groupby(["company_name", "ticker"]):
        g2 = g[["date", "share_price"]].copy()
        g2["equity_ret"] = pd.to_numeric(g2["share_price"], errors="coerce").pct_change()

        merged = g2.merge(factor[["date"] + x_cols], on="date", how="left")
        betas = _regression_betas(merged, "equity_ret", x_cols)

        actual_cum = (1 + merged["equity_ret"].dropna()).prod() - 1 if merged["equity_ret"].notna().any() else np.nan
        model_ret_series = pd.Series(betas.get("alpha", 0.0), index=merged.index, dtype=float)
        for c in x_cols:
            beta = betas.get(f"beta_{c}", np.nan)
            if pd.notna(beta):
                model_ret_series = model_ret_series + (beta * merged[c].fillna(0))
        model_cum = (1 + model_ret_series.dropna()).prod() - 1
        residual = (actual_cum - model_cum) * 100 if pd.notna(actual_cum) and pd.notna(model_cum) else np.nan

        rows.append(
            {
                "company_name": company,
                "ticker": ticker,
                "beta_brent_ret": betas.get("beta_brent_ret"),
                "beta_market_ret": betas.get("beta_market_ret"),
                "beta_energy_ret": betas.get("beta_energy_ret"),
                "beta_rates_ret": betas.get("beta_rates_ret"),
                "beta_fx_ret": betas.get("beta_fx_ret"),
                "alpha": betas.get("alpha"),
                "r2": betas.get("r2"),
                "idiosyncratic_residual_pct": residual,
            }
        )

    out = pd.DataFrame(rows)
    logger.info("Built factor decomposition rows: %s", len(out))
    return out
