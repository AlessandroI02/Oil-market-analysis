from __future__ import annotations

from datetime import date
from math import erf, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.market_data_providers import (
    fetch_crude_series,
    fetch_equity_series,
    fetch_factor_series,
)
from src.storage_paths import write_csv


def _normal_two_tailed_pvalue(t_stat: float) -> float:
    z = abs(float(t_stat))
    return float(max(0.0, min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))))))


def _fetch_daily_prices(
    included_df: pd.DataFrame,
    start_date: date,
    end_date: date,
    raw_market_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    equity_prices: dict[str, pd.DataFrame] = {}
    for ticker in included_df["ticker"].astype(str).tolist():
        frame, provider, _, _ = fetch_equity_series(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            diagnostics=None,
        )
        frame = frame.rename(columns={"price": ticker})
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date")
        if ticker in frame.columns and frame[ticker].notna().any():
            equity_prices[ticker] = frame[["date", ticker]]
            raw_frame = frame.copy()
            raw_frame["provider"] = provider
            write_csv(raw_frame, raw_market_dir / f"{ticker}_daily_raw.csv")

    if equity_prices:
        merged = None
        for ticker, frame in equity_prices.items():
            merged = frame if merged is None else merged.merge(frame, on="date", how="outer")
        equity_price_matrix = merged.sort_values("date").reset_index(drop=True)
    else:
        equity_price_matrix = pd.DataFrame(columns=["date"])

    brent, brent_provider, _, _ = fetch_crude_series("brent", start_date, end_date, diagnostics=None)
    xle, xle_provider, _, _ = fetch_factor_series("energy", start_date, end_date, diagnostics=None)
    spy, spy_provider, _, _ = fetch_factor_series("market", start_date, end_date, diagnostics=None)
    rates, rates_provider, _, _ = fetch_factor_series("rates", start_date, end_date, diagnostics=None)
    dxy, dxy_provider, _, _ = fetch_factor_series("fx", start_date, end_date, diagnostics=None)

    benchmark_price = (
        brent.rename(columns={"price": "brent_price"})[["date", "brent_price"]]
        .merge(xle.rename(columns={"price": "xle_price"})[["date", "xle_price"]], on="date", how="outer")
        .merge(spy.rename(columns={"price": "spy_price"})[["date", "spy_price"]], on="date", how="outer")
        .merge(rates.rename(columns={"price": "rates_price"})[["date", "rates_price"]], on="date", how="outer")
        .merge(dxy.rename(columns={"price": "dxy_price"})[["date", "dxy_price"]], on="date", how="outer")
        .sort_values("date")
    )

    bench_raw = benchmark_price.copy()
    bench_raw["brent_provider"] = brent_provider
    bench_raw["xle_provider"] = xle_provider
    bench_raw["spy_provider"] = spy_provider
    bench_raw["rates_provider"] = rates_provider
    bench_raw["dxy_provider"] = dxy_provider
    write_csv(bench_raw, raw_market_dir / "benchmark_daily_raw.csv")

    return equity_price_matrix, benchmark_price


def _returns_from_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame(columns=["date"])
    out = price_df.copy()
    out = out.sort_values("date")
    for c in out.columns:
        if c == "date":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").pct_change()
    return out


def _weekly_returns_from_prices(price_df: pd.DataFrame, freq: str = "W-FRI") -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame(columns=["date"])
    tmp = price_df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp.dropna(subset=["date"]).set_index("date").sort_index()
    weekly_px = tmp.resample(freq).last().reset_index()
    return _returns_from_prices(weekly_px)


def _cumulative_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    if returns_df is None or returns_df.empty:
        return pd.DataFrame(columns=["date"])
    out = returns_df.copy()
    for c in out.columns:
        if c == "date":
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        out[c] = (1.0 + s.fillna(0.0)).cumprod() - 1.0
    return out


def _ols_with_stats(y: pd.Series, x_df: pd.DataFrame) -> dict[str, Any]:
    df = pd.concat([y.rename("y"), x_df], axis=1).dropna()
    if len(df) < 15:
        return {"ok": False}

    yv = df["y"].to_numpy(dtype=float)
    xv = df[x_df.columns].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(xv)), xv])
    n, k = x.shape
    if n <= k + 1:
        return {"ok": False}

    xtx = x.T @ x
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return {"ok": False}

    beta = xtx_inv @ (x.T @ yv)
    y_hat = x @ beta
    resid = yv - y_hat
    rss = float(np.sum(resid**2))
    tss = float(np.sum((yv - np.mean(yv)) ** 2))
    sigma2 = rss / max(n - k, 1)
    var_beta = sigma2 * xtx_inv
    se = np.sqrt(np.maximum(np.diag(var_beta), 1e-12))
    t_stat = beta / se
    pvals = np.array([_normal_two_tailed_pvalue(v) for v in t_stat])
    r2 = 1.0 - (rss / tss if tss > 0 else 0.0)
    adj_r2 = 1.0 - (1.0 - r2) * ((n - 1) / max(n - k, 1))
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))

    out = {
        "ok": True,
        "sample_size": int(n),
        "r_squared": float(r2),
        "adj_r_squared": float(adj_r2),
        "rmse": rmse,
        "mae": mae,
        "coef_intercept": float(beta[0]),
        "t_intercept": float(t_stat[0]),
        "p_intercept": float(pvals[0]),
    }
    for idx, col in enumerate(x_df.columns, start=1):
        out[f"coef_{col}"] = float(beta[idx])
        out[f"t_{col}"] = float(t_stat[idx])
        out[f"p_{col}"] = float(pvals[idx])
    return out


def _rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    beta = cov / var.replace(0, np.nan)
    return beta


def _kmeans_simple(features: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
    if len(features) == 0:
        return np.array([])
    rng = np.random.default_rng(42)
    idx = rng.choice(len(features), size=min(k, len(features)), replace=False)
    centroids = features[idx]
    labels = np.zeros(len(features), dtype=int)

    for _ in range(max_iter):
        dists = np.sqrt(((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(len(centroids)):
            pts = features[labels == j]
            if len(pts) > 0:
                centroids[j] = pts.mean(axis=0)
    return labels


def build_market_math_exports(
    included_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    route_risks_df: pd.DataFrame,
    event_days_df: pd.DataFrame,
    start_date: date,
    end_date: date,
    raw_market_dir: Path,
    interim_market_dir: Path,
) -> dict[str, pd.DataFrame]:
    raw_market_dir.mkdir(parents=True, exist_ok=True)
    interim_market_dir.mkdir(parents=True, exist_ok=True)

    equity_price_matrix, benchmark_price = _fetch_daily_prices(
        included_df=included_df,
        start_date=start_date,
        end_date=end_date,
        raw_market_dir=raw_market_dir,
    )
    write_csv(equity_price_matrix, interim_market_dir / "equity_daily_prices_matrix.csv")
    write_csv(benchmark_price, interim_market_dir / "benchmark_daily_prices.csv")

    returns_daily = _returns_from_prices(equity_price_matrix)
    returns_weekly = _weekly_returns_from_prices(equity_price_matrix, freq="W-FRI")
    cumulative_returns = _cumulative_returns(returns_daily)
    benchmark_returns = _returns_from_prices(
        benchmark_price.rename(
            columns={
                "brent_price": "brent",
                "xle_price": "xle",
                "spy_price": "spy",
                "rates_price": "rates",
                "dxy_price": "dxy",
            }
        )
    ).rename(
        columns={
            "brent": "brent_ret",
            "brent_price": "brent_ret",
            "xle": "xle_ret",
            "xle_price": "xle_ret",
            "spy": "spy_ret",
            "spy_price": "spy_ret",
            "rates": "rates_ret",
            "rates_price": "rates_ret",
            "dxy": "dxy_ret",
            "dxy_price": "dxy_ret",
        }
    )
    for req_col in ["date", "brent_ret", "xle_ret", "spy_ret", "rates_ret", "dxy_ret"]:
        if req_col not in benchmark_returns.columns:
            benchmark_returns[req_col] = np.nan
    benchmark_returns = benchmark_returns[["date", "brent_ret", "xle_ret", "spy_ret", "rates_ret", "dxy_ret"]]

    join_for_corr = returns_daily.merge(
        benchmark_returns[["date", "brent_ret", "xle_ret", "spy_ret", "rates_ret", "dxy_ret"]],
        on="date",
        how="left",
    )

    corr_cols = [c for c in join_for_corr.columns if c != "date"]
    correlation_matrix = join_for_corr[corr_cols].corr()
    correlation_matrix = correlation_matrix.reset_index().rename(columns={"index": "series"})

    rolling_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    regression_diag_rows: list[dict[str, Any]] = []
    rolling_beta_rows: list[dict[str, Any]] = []
    event_detail_rows: list[dict[str, Any]] = []

    event_tbl = event_days_df.copy() if event_days_df is not None else pd.DataFrame()
    if not event_tbl.empty:
        event_tbl["event_date"] = pd.to_datetime(event_tbl.get("date"), errors="coerce")
        event_tbl = event_tbl.dropna(subset=["event_date"]).sort_values("event_date")
        event_tbl["direction"] = event_tbl.get("direction", "up").astype(str)
    else:
        auto_events = benchmark_returns[["date", "brent_ret"]].copy() if "brent_ret" in benchmark_returns.columns else pd.DataFrame()
        if not auto_events.empty:
            auto_events = auto_events[pd.to_numeric(auto_events["brent_ret"], errors="coerce").abs() >= 0.02]
            auto_events["direction"] = auto_events["brent_ret"].apply(lambda x: "up" if x >= 0 else "down")
            auto_events["candidate_catalyst"] = "Price action episode"
            auto_events = auto_events.rename(columns={"date": "event_date"})
            event_tbl = auto_events

    for ticker in [c for c in returns_daily.columns if c != "date"]:
        sub = join_for_corr[["date", ticker, "brent_ret", "xle_ret", "spy_ret", "rates_ret", "dxy_ret"]].copy()
        sub = sub.rename(columns={ticker: "stock_ret"})
        sub = sub.dropna(subset=["stock_ret"])

        for window in [20, 60, 90]:
            if len(sub) >= window:
                s = sub[["date", "stock_ret", "brent_ret"]].copy()
                s[f"rolling_corr_{window}d"] = s["stock_ret"].rolling(window).corr(s["brent_ret"])
                for _, rr in s.dropna(subset=[f"rolling_corr_{window}d"]).iterrows():
                    rolling_rows.append(
                        {
                            "date": pd.to_datetime(rr["date"]).date(),
                            "ticker": ticker,
                            "window": f"{window}d",
                            "rolling_corr": float(rr[f"rolling_corr_{window}d"]),
                        }
                    )

        crisis = sub[sub["brent_ret"].abs() > 0.02]
        if len(crisis) >= 4:
            crisis_corr = float(crisis["stock_ret"].corr(crisis["brent_ret"]))
            rolling_rows.append(
                {
                    "date": sub["date"].max().date(),
                    "ticker": ticker,
                    "window": "crisis_only",
                    "rolling_corr": crisis_corr,
                }
            )

        gt2 = sub[sub["brent_ret"].abs() > 0.02]
        if len(gt2) >= 4:
            gt2_corr = float(gt2["stock_ret"].corr(gt2["xle_ret"]))
            rolling_rows.append(
                {
                    "date": sub["date"].max().date(),
                    "ticker": ticker,
                    "window": "oil_gt_2pct",
                    "rolling_corr": gt2_corr,
                }
            )

        ols = _ols_with_stats(
            y=sub["stock_ret"],
            x_df=sub[["brent_ret", "xle_ret", "spy_ret", "rates_ret", "dxy_ret"]],
        )
        if ols.get("ok"):
            regression_rows.append(
                {
                    "ticker": ticker,
                    "window_start": pd.to_datetime(sub["date"].min()).date().isoformat(),
                    "window_end": pd.to_datetime(sub["date"].max()).date().isoformat(),
                    **{k: v for k, v in ols.items() if k.startswith("coef_") or k.startswith("t_") or k.startswith("p_")},
                }
            )
            regression_diag_rows.append(
                {
                    "ticker": ticker,
                    "sample_size": ols["sample_size"],
                    "window_start": pd.to_datetime(sub["date"].min()).date().isoformat(),
                    "window_end": pd.to_datetime(sub["date"].max()).date().isoformat(),
                    "r_squared": ols["r_squared"],
                    "adj_r_squared": ols["adj_r_squared"],
                    "rmse": ols["rmse"],
                    "mae": ols["mae"],
                }
            )

        oil20 = _rolling_beta(sub["stock_ret"], sub["brent_ret"], 20)
        oil60 = _rolling_beta(sub["stock_ret"], sub["brent_ret"], 60)
        oil90 = _rolling_beta(sub["stock_ret"], sub["brent_ret"], 90)
        sec20 = _rolling_beta(sub["stock_ret"], sub["xle_ret"], 20)
        sec60 = _rolling_beta(sub["stock_ret"], sub["xle_ret"], 60)
        sec90 = _rolling_beta(sub["stock_ret"], sub["xle_ret"], 90)
        mkt20 = _rolling_beta(sub["stock_ret"], sub["spy_ret"], 20)
        mkt60 = _rolling_beta(sub["stock_ret"], sub["spy_ret"], 60)
        mkt90 = _rolling_beta(sub["stock_ret"], sub["spy_ret"], 90)

        for _, rr in sub.iterrows():
            idx = rr.name
            rolling_beta_rows.append(
                {
                    "date": pd.to_datetime(rr["date"]).date(),
                    "ticker": ticker,
                    "rolling_oil_beta_20d": _safe_value(oil20.loc[idx] if idx in oil20.index else np.nan),
                    "rolling_oil_beta_60d": _safe_value(oil60.loc[idx] if idx in oil60.index else np.nan),
                    "rolling_oil_beta_90d": _safe_value(oil90.loc[idx] if idx in oil90.index else np.nan),
                    "rolling_sector_beta_20d": _safe_value(sec20.loc[idx] if idx in sec20.index else np.nan),
                    "rolling_sector_beta_60d": _safe_value(sec60.loc[idx] if idx in sec60.index else np.nan),
                    "rolling_sector_beta_90d": _safe_value(sec90.loc[idx] if idx in sec90.index else np.nan),
                    "rolling_market_beta_20d": _safe_value(mkt20.loc[idx] if idx in mkt20.index else np.nan),
                    "rolling_market_beta_60d": _safe_value(mkt60.loc[idx] if idx in mkt60.index else np.nan),
                    "rolling_market_beta_90d": _safe_value(mkt90.loc[idx] if idx in mkt90.index else np.nan),
                }
            )

        if not event_tbl.empty:
            sub_idx = sub.set_index(pd.to_datetime(sub["date"]).dt.date)
            for _, ev in event_tbl.iterrows():
                ev_date = ev["event_date"].date()
                if ev_date not in sub_idx.index:
                    continue
                event_row = sub_idx.loc[ev_date]
                if isinstance(event_row, pd.DataFrame):
                    event_row = event_row.iloc[-1]
                day0 = float(event_row["stock_ret"])
                day0_bmk = float(event_row.get("xle_ret", np.nan))

                windows = {}
                for horizon in [1, 3, 5]:
                    future = sub_idx[sub_idx.index > ev_date].head(horizon)
                    fut_stock = future["stock_ret"].dropna()
                    fut_bmk = future["xle_ret"].dropna()
                    stock_win = float((1.0 + fut_stock).prod() - 1.0) if not fut_stock.empty else np.nan
                    bmk_win = float((1.0 + fut_bmk).prod() - 1.0) if not fut_bmk.empty else np.nan
                    windows[horizon] = (stock_win, bmk_win)

                abnormal_day0 = day0 - day0_bmk if pd.notna(day0_bmk) else np.nan
                abnormal_1 = windows[1][0] - windows[1][1] if pd.notna(windows[1][0]) and pd.notna(windows[1][1]) else np.nan
                abnormal_3 = windows[3][0] - windows[3][1] if pd.notna(windows[3][0]) and pd.notna(windows[3][1]) else np.nan
                abnormal_5 = windows[5][0] - windows[5][1] if pd.notna(windows[5][0]) and pd.notna(windows[5][1]) else np.nan

                event_detail_rows.append(
                    {
                        "event_date": ev_date.isoformat(),
                        "ticker": ticker,
                        "direction": ev.get("direction", "up"),
                        "candidate_catalyst": ev.get("candidate_catalyst", ""),
                        "day0_return": day0,
                        "window_1d_return": windows[1][0],
                        "window_3d_return": windows[3][0],
                        "window_5d_return": windows[5][0],
                        "abnormal_day0": abnormal_day0,
                        "abnormal_1d": abnormal_1,
                        "abnormal_3d": abnormal_3,
                        "abnormal_5d": abnormal_5,
                        "cumulative_abnormal_return": abnormal_5,
                    }
                )

    rolling_correlations = pd.DataFrame(rolling_rows)
    regression_coefficients = pd.DataFrame(regression_rows)
    regression_diagnostics = pd.DataFrame(regression_diag_rows)
    rolling_betas = pd.DataFrame(rolling_beta_rows)
    event_study_detail = pd.DataFrame(event_detail_rows)

    if event_study_detail.empty:
        event_study_summary = pd.DataFrame(
            columns=[
                "ticker",
                "event_count",
                "event_day_hit_rate_up_oil",
                "event_day_hit_rate_down_oil",
                "avg_abnormal_return_oil_up",
                "avg_abnormal_return_oil_down",
                "average_episode_response",
            ]
        )
    else:
        summary_rows: list[dict[str, Any]] = []
        for ticker, grp in event_study_detail.groupby("ticker"):
            up = grp[grp["direction"].astype(str).str.lower() == "up"]
            down = grp[grp["direction"].astype(str).str.lower() == "down"]
            summary_rows.append(
                {
                    "ticker": ticker,
                    "event_count": int(len(grp)),
                    "event_day_hit_rate_up_oil": float((up["abnormal_day0"] > 0).mean()) if not up.empty else np.nan,
                    "event_day_hit_rate_down_oil": float((down["abnormal_day0"] < 0).mean()) if not down.empty else np.nan,
                    "avg_abnormal_return_oil_up": float(up["abnormal_day0"].mean()) if not up.empty else np.nan,
                    "avg_abnormal_return_oil_down": float(down["abnormal_day0"].mean()) if not down.empty else np.nan,
                    "average_episode_response": float(grp["cumulative_abnormal_return"].mean()) if grp["cumulative_abnormal_return"].notna().any() else np.nan,
                }
            )
        event_study_summary = pd.DataFrame(summary_rows)

    metrics = []
    exposure_map = exposure_df.set_index("ticker")["combined_exposure_pct"].to_dict() if exposure_df is not None and not exposure_df.empty and "combined_exposure_pct" in exposure_df.columns else {}
    route_map = route_risks_df.set_index("ticker")["qualitative_route_risk"].to_dict() if route_risks_df is not None and not route_risks_df.empty and "qualitative_route_risk" in route_risks_df.columns else {}
    reg_map = regression_coefficients.set_index("ticker") if not regression_coefficients.empty else pd.DataFrame()
    for ticker in [c for c in returns_daily.columns if c != "date"]:
        joined = join_for_corr[["date", ticker, "xle_ret", "brent_ret"]].rename(columns={ticker: "stock_ret"})
        downside = joined[joined["brent_ret"] < 0]["stock_ret"].std()
        rel_vs_sector = (joined["stock_ret"] - joined["xle_ret"]).mean()
        event_row = event_study_summary[event_study_summary["ticker"] == ticker]
        event_outperf = event_row["avg_abnormal_return_oil_up"].iloc[0] if not event_row.empty else np.nan
        oil_beta = reg_map.loc[ticker]["coef_brent_ret"] if not reg_map.empty and ticker in reg_map.index and "coef_brent_ret" in reg_map.columns else np.nan
        metrics.append(
            {
                "ticker": ticker,
                "oil_beta": oil_beta,
                "route_exposure": exposure_map.get(ticker, np.nan),
                "downside_volatility": downside,
                "event_outperformance": event_outperf,
                "relative_vs_sector": rel_vs_sector,
                "route_risk_label": route_map.get(ticker, ""),
            }
        )
    metric_df = pd.DataFrame(metrics)
    metric_cols = ["oil_beta", "route_exposure", "downside_volatility", "event_outperformance", "relative_vs_sector"]
    required_metric_cols = ["ticker", "route_risk_label", *metric_cols]
    for col in required_metric_cols:
        if col not in metric_df.columns:
            metric_df[col] = pd.Series(dtype=float if col in metric_cols else object)

    z_rows: list[dict[str, Any]] = []
    p_rows: list[dict[str, Any]] = []
    for col in metric_cols:
        s = pd.to_numeric(metric_df[col], errors="coerce")
        mean = s.mean()
        std = s.std()
        z = (s - mean) / (std if std and not pd.isna(std) and std != 0 else 1.0)
        pct = s.rank(pct=True)
        for idx, ticker in enumerate(metric_df["ticker"]):
            z_rows.append({"ticker": ticker, "metric": col, "value": s.iloc[idx], "zscore": z.iloc[idx]})
            p_rows.append({"ticker": ticker, "metric": col, "value": s.iloc[idx], "percentile": pct.iloc[idx]})

    metric_zscores = pd.DataFrame(z_rows, columns=["ticker", "metric", "value", "zscore"])
    metric_percentiles = pd.DataFrame(p_rows, columns=["ticker", "metric", "value", "percentile"])

    cluster_features = (
        metric_zscores.pivot(index="ticker", columns="metric", values="zscore")
        .reset_index()
        .fillna(0.0)
    )
    if not cluster_features.empty:
        feature_cols = [c for c in cluster_features.columns if c != "ticker"]
        x = cluster_features[feature_cols].to_numpy(dtype=float)
        k = min(4, len(cluster_features))
        labels = _kmeans_simple(x, k=max(1, k))
        cluster_features["cluster_id"] = labels + 1
        cluster_features["cluster_label"] = cluster_features["cluster_id"].map(lambda x: f"cluster_{x}")
        cluster_assignments = cluster_features
    else:
        cluster_assignments = pd.DataFrame(columns=["ticker", "cluster_id", "cluster_label"])

    write_csv(returns_daily, interim_market_dir / "returns_daily_matrix.csv")
    write_csv(returns_weekly, interim_market_dir / "returns_weekly_matrix.csv")
    write_csv(cumulative_returns, interim_market_dir / "cumulative_returns_matrix.csv")
    write_csv(benchmark_returns, interim_market_dir / "benchmark_returns_matrix.csv")

    return {
        "returns_daily_matrix": returns_daily,
        "returns_weekly_matrix": returns_weekly,
        "cumulative_returns_matrix": cumulative_returns,
        "benchmark_returns_matrix": benchmark_returns,
        "correlation_matrix": correlation_matrix,
        "rolling_correlations": rolling_correlations,
        "regression_coefficients": regression_coefficients,
        "regression_diagnostics": regression_diagnostics,
        "rolling_betas": rolling_betas,
        "event_study_summary": event_study_summary,
        "event_study_detail": event_study_detail,
        "cluster_assignments": cluster_assignments,
        "metric_zscores": metric_zscores,
        "metric_percentiles": metric_percentiles,
        "equity_daily_prices_matrix": equity_price_matrix,
        "benchmark_daily_prices": benchmark_price,
    }


def _safe_value(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None
