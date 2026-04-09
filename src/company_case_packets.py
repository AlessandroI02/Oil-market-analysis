from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from src.storage_paths import write_json


def _latest_value(df: pd.DataFrame, value_col: str) -> float | None:
    if df is None or df.empty or value_col not in df.columns:
        return None
    s = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _source_summary_for_company(source_log_df: pd.DataFrame, company: str) -> str:
    if source_log_df is None or source_log_df.empty:
        return ""
    src = source_log_df[source_log_df.get("company") == company].copy()
    if src.empty:
        return ""
    domains = src.get("source_url", pd.Series(dtype=str)).astype(str).map(lambda u: urlparse(u).netloc.lower())
    counts = domains[domains != ""].value_counts().head(5)
    return ", ".join([f"{d}:{c}" for d, c in counts.items()])


def _ticker_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    out = df[df["ticker"] == ticker].copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values("date")
    return out


def _stock_returns_from_weekly(weekly_df: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    if weekly_df is None or weekly_df.empty or "share_price" not in weekly_df.columns:
        return None, None, None
    prices = pd.to_numeric(weekly_df["share_price"], errors="coerce")
    latest = prices.dropna().iloc[-1] if prices.notna().any() else None
    if latest is None:
        return None, None, None

    ret_1w = None
    ret_1m = None
    if len(prices.dropna()) >= 2:
        prev = prices.dropna().iloc[-2]
        if prev != 0:
            ret_1w = float((latest / prev - 1.0) * 100.0)
    if len(prices.dropna()) >= 5:
        prev4 = prices.dropna().iloc[-5]
        if prev4 != 0:
            ret_1m = float((latest / prev4 - 1.0) * 100.0)
    return float(latest), ret_1w, ret_1m


def build_company_case_packets(
    included_df: pd.DataFrame,
    archetypes_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    route_risks_df: pd.DataFrame,
    crude_tracker_df: pd.DataFrame,
    fuel_tracker_df: pd.DataFrame,
    equity_tracker_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    regression_coefficients_df: pd.DataFrame,
    rolling_betas_df: pd.DataFrame,
    event_study_summary_df: pd.DataFrame,
    historical_analogues_df: pd.DataFrame,
    catalyst_primary_df: pd.DataFrame,
    market_constraints_df: pd.DataFrame,
    confidence_framework_df: pd.DataFrame,
    regime_state_df: pd.DataFrame,
    source_log_df: pd.DataFrame,
    output_dir: Path,
    as_of_date: date,
    root_dir: Path | None = None,
) -> pd.DataFrame:
    if included_df is None or included_df.empty:
        return pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)

    archetypes = archetypes_df.set_index("ticker") if archetypes_df is not None and not archetypes_df.empty and "ticker" in archetypes_df.columns else pd.DataFrame()
    exposure = exposure_df.set_index("ticker") if exposure_df is not None and not exposure_df.empty and "ticker" in exposure_df.columns else pd.DataFrame()
    route = route_risks_df.set_index("ticker") if route_risks_df is not None and not route_risks_df.empty and "ticker" in route_risks_df.columns else pd.DataFrame()
    factor = factor_df.set_index("ticker") if factor_df is not None and not factor_df.empty and "ticker" in factor_df.columns else pd.DataFrame()
    regress = regression_coefficients_df.set_index("ticker") if regression_coefficients_df is not None and not regression_coefficients_df.empty and "ticker" in regression_coefficients_df.columns else pd.DataFrame()
    event_summary = event_study_summary_df.set_index("ticker") if event_study_summary_df is not None and not event_study_summary_df.empty and "ticker" in event_study_summary_df.columns else pd.DataFrame()
    constraints = market_constraints_df.set_index("ticker") if market_constraints_df is not None and not market_constraints_df.empty and "ticker" in market_constraints_df.columns else pd.DataFrame()
    confidence = confidence_framework_df.set_index("ticker") if confidence_framework_df is not None and not confidence_framework_df.empty and "ticker" in confidence_framework_df.columns else pd.DataFrame()

    regime_label = str(regime_state_df.iloc[0].get("regime_label", "normal")) if regime_state_df is not None and not regime_state_df.empty else "normal"
    regime_summary = str(regime_state_df.iloc[0].get("regime_summary", "")) if regime_state_df is not None and not regime_state_df.empty else ""

    latest_brent = _latest_value(crude_tracker_df.sort_values("date") if crude_tracker_df is not None and not crude_tracker_df.empty else pd.DataFrame(), "brent_price")
    latest_wti = _latest_value(crude_tracker_df.sort_values("date") if crude_tracker_df is not None and not crude_tracker_df.empty else pd.DataFrame(), "wti_price")

    analogue_summary = ""
    if historical_analogues_df is not None and not historical_analogues_df.empty:
        last = historical_analogues_df.tail(1).iloc[0]
        brent_ret = _safe_float(last.get("brent_return_pct"))
        peer_ret = _safe_float(last.get("peer_median_return_pct"))
        brent_text = f"{brent_ret:.1f}%" if brent_ret is not None else "n/a"
        peer_text = f"{peer_ret:.1f}%" if peer_ret is not None else "n/a"
        analogue_summary = (
            f"{last.get('period', 'latest analogue')}: Brent {brent_text} "
            f"vs peer median {peer_text}."
        )

    index_rows: list[dict[str, Any]] = []
    for _, row in included_df.iterrows():
        ticker = str(row["ticker"])
        company = str(row["company_name"])

        arche = archetypes.loc[ticker] if not archetypes.empty and ticker in archetypes.index else pd.Series(dtype=object)
        exp = exposure.loc[ticker] if not exposure.empty and ticker in exposure.index else pd.Series(dtype=object)
        rr = route.loc[ticker] if not route.empty and ticker in route.index else pd.Series(dtype=object)
        fac = factor.loc[ticker] if not factor.empty and ticker in factor.index else pd.Series(dtype=object)
        reg = regress.loc[ticker] if not regress.empty and ticker in regress.index else pd.Series(dtype=object)
        evt = event_summary.loc[ticker] if not event_summary.empty and ticker in event_summary.index else pd.Series(dtype=object)
        con = constraints.loc[ticker] if not constraints.empty and ticker in constraints.index else pd.Series(dtype=object)
        conf = confidence.loc[ticker] if not confidence.empty and ticker in confidence.index else pd.Series(dtype=object)

        fuel_series = _ticker_frame(fuel_tracker_df, ticker)
        latest_fuel_proxy = _latest_value(fuel_series, "blended_combined_fuels_price")

        equity_series = _ticker_frame(equity_tracker_df, ticker)
        stock_price_latest, stock_return_1w, stock_return_1m = _stock_returns_from_weekly(equity_series)

        if catalyst_primary_df is not None and not catalyst_primary_df.empty and "ticker" in catalyst_primary_df.columns:
            next_cats = (
                catalyst_primary_df[catalyst_primary_df["ticker"] == ticker]
                .sort_values("event_date")
                .head(5)
            )
            next_relevant_catalysts = [
                f"{pd.to_datetime(r['event_date'], errors='coerce').date().isoformat()} {r['event']}"
                for _, r in next_cats.iterrows()
            ]
        else:
            next_relevant_catalysts = []

        rolling_t = rolling_betas_df[rolling_betas_df.get("ticker") == ticker].copy() if rolling_betas_df is not None and not rolling_betas_df.empty else pd.DataFrame()
        rolling_t = rolling_t.sort_values("date") if not rolling_t.empty and "date" in rolling_t.columns else rolling_t
        rb20 = _latest_value(rolling_t, "rolling_oil_beta_20d")
        rb60 = _latest_value(rolling_t, "rolling_oil_beta_60d")
        rb90 = _latest_value(rolling_t, "rolling_oil_beta_90d")

        source_summary = str(conf.get("source_summary", "")) if isinstance(conf, pd.Series) and conf.get("source_summary", "") else _source_summary_for_company(source_log_df, company)
        packet_conf = str(conf.get("packet_confidence", "Medium")) if isinstance(conf, pd.Series) else "Medium"
        data_conf = conf.get("data_confidence", "Medium")
        source_conf = conf.get("source_confidence", "Medium")
        route_conf = conf.get("route_confidence", "Medium")
        event_conf = conf.get("event_confidence", "Medium")
        regime_conf = conf.get("regime_confidence", "Medium")
        input_data_conf = conf.get("input_data_confidence", data_conf)
        route_model_conf = conf.get("route_model_confidence", route_conf)
        event_model_conf = conf.get("event_model_confidence", event_conf)
        regime_model_conf = conf.get("regime_model_confidence", regime_conf)
        downstream_ready = bool(conf.get("downstream_readiness_flag", False)) if isinstance(conf, pd.Series) else False
        downstream_readiness_reason = str(conf.get("downstream_readiness_reason", "")) if isinstance(conf, pd.Series) else ""
        publishable_flag = bool(conf.get("publishable_flag", False)) if isinstance(conf, pd.Series) else False
        rating_status = str(conf.get("rating_status", "unrated")) if isinstance(conf, pd.Series) else "unrated"
        final_rating_confidence = str(conf.get("final_rating_confidence", "Low")) if isinstance(conf, pd.Series) else "Low"
        rating_gate_reason = str(conf.get("rating_gate_reason", "")) if isinstance(conf, pd.Series) else ""
        publishable_gate_reason = str(conf.get("publishable_gate_reason", "")) if isinstance(conf, pd.Series) else ""
        packet_high_eligibility = bool(conf.get("packet_high_eligibility", False)) if isinstance(conf, pd.Series) else False
        packet_high_eligibility_reasons = str(conf.get("packet_high_eligibility_reasons", "")) if isinstance(conf, pd.Series) else ""
        component_low_count = int(_safe_float(conf.get("component_low_count")) or 0) if isinstance(conf, pd.Series) else 0
        contradiction_count = int(_safe_float(conf.get("confidence_contradiction_count")) or 0) if isinstance(conf, pd.Series) else 0
        confidence_summary = (
            f"data={data_conf}, source={source_conf}, route={route_conf}, "
            f"event={event_conf}, regime={regime_conf}, packet={packet_conf}, "
            f"rating_status={rating_status}, publishable={publishable_flag}, downstream_ready={downstream_ready}, "
            f"component_low_count={component_low_count}, contradiction_count={contradiction_count}"
        )

        packet = {
            "as_of_date": as_of_date.isoformat(),
            "company_name": company,
            "ticker": ticker,
            "bucket_classification": row.get("bucket_classification", ""),
            "archetype": arche.get("archetype", ""),
            "inclusion_confidence": arche.get("inclusion_confidence", row.get("confidence", "Medium")),
            "input_data_confidence": input_data_conf,
            "route_model_confidence": route_model_conf,
            "event_model_confidence": event_model_conf,
            "regime_model_confidence": regime_model_conf,
            "data_confidence": data_conf,
            "source_confidence": source_conf,
            "route_confidence": route_conf,
            "event_confidence": event_conf,
            "regime_confidence": regime_conf,
            "packet_confidence": packet_conf,
            "company_packet_confidence": conf.get("company_packet_confidence", packet_conf),
            "packet_high_eligibility": packet_high_eligibility,
            "packet_high_eligibility_reasons": packet_high_eligibility_reasons,
            "downstream_readiness_flag": downstream_ready,
            "downstream_readiness_reason": downstream_readiness_reason,
            "publishable_flag": publishable_flag,
            "publishable_gate_reason": publishable_gate_reason,
            "rating_status": rating_status,
            "final_rating_confidence": final_rating_confidence,
            "rating_gate_reason": rating_gate_reason,
            "regime_label": regime_label,
            "regime_summary": regime_summary,
            "latest_brent_price": latest_brent,
            "latest_wti_price": latest_wti,
            "latest_fuel_proxy": latest_fuel_proxy,
            "stock_price_latest": stock_price_latest,
            "stock_return_1w": stock_return_1w,
            "stock_return_1m": stock_return_1m,
            "oil_beta": _safe_float(reg.get("coef_brent_ret", fac.get("beta_brent_ret"))),
            "sector_beta": _safe_float(reg.get("coef_xle_ret", fac.get("beta_energy_ret"))),
            "market_beta": _safe_float(reg.get("coef_spy_ret", fac.get("beta_market_ret"))),
            "rates_beta": _safe_float(reg.get("coef_rates_ret")),
            "dxy_beta": _safe_float(reg.get("coef_dxy_ret")),
            "rolling_oil_beta_20d": rb20,
            "rolling_oil_beta_60d": rb60,
            "rolling_oil_beta_90d": rb90,
            "route_exposure_central": _safe_float(exp.get("combined_exposure_pct")),
            "route_exposure_low": _safe_float(exp.get("exposure_low_pct")),
            "route_exposure_high": _safe_float(exp.get("exposure_high_pct")),
            "hormuz_share_pct": _safe_float(rr.get("hormuz_share_pct")),
            "bab_el_mandeb_share_pct": _safe_float(rr.get("bab_el_mandeb_share_pct")),
            "suez_share_pct": _safe_float(rr.get("suez_share_pct")),
            "non_chokepoint_share_pct": _safe_float(rr.get("non_chokepoint_share_pct")),
            "route_risk_label": rr.get("qualitative_route_risk", ""),
            "rerouting_flexibility": rr.get("rerouting_flexibility", ""),
            "pipeline_bypass_optionality": rr.get("pipeline_bypass_optionality", ""),
            "event_day_hit_rate_up_oil": _safe_float(evt.get("event_day_hit_rate_up_oil")),
            "event_day_hit_rate_down_oil": _safe_float(evt.get("event_day_hit_rate_down_oil")),
            "avg_abnormal_return_oil_up": _safe_float(evt.get("avg_abnormal_return_oil_up")),
            "avg_abnormal_return_oil_down": _safe_float(evt.get("avg_abnormal_return_oil_down")),
            "analogue_summary": analogue_summary,
            "next_relevant_catalysts": next_relevant_catalysts,
            "market_constraint_summary": str(con.get("market_constraint_summary", con.get("market_regime_impact_note", ""))),
            "suggested_discount_rate_uplift_bps": _safe_float(con.get("suggested_discount_rate_uplift_bps")),
            "suggested_discount_rate_uplift_range_bps": con.get("suggested_discount_rate_uplift_range_bps", ""),
            "suggested_risk_premium_bucket": con.get("suggested_risk_premium_bucket", ""),
            "suggested_beta_adjustment": _safe_float(con.get("suggested_beta_adjustment")),
            "suggested_beta_adjustment_range": con.get("suggested_beta_adjustment_range", ""),
            "suggested_scenario_probability_shift": con.get("suggested_scenario_probability_shift", con.get("scenario_probability_shift", "")),
            "constraint_confidence": con.get("constraint_confidence", ""),
            "confidence_summary": confidence_summary,
            "key_notes": (
                f"{str(rr.get('disruption_notes', ''))} "
                f"Repo 1 note: this packet is an upstream market-intelligence overlay, not a final investment call."
            ).strip(),
            "source_summary": source_summary,
        }

        packet_path = output_dir / f"{ticker}.json"
        write_json(packet, packet_path)
        packet_path_index = str(packet_path)
        if root_dir is not None:
            try:
                packet_path_index = str(packet_path.relative_to(root_dir))
            except Exception:
                packet_path_index = str(packet_path)
        index_rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "company_name": company,
                "ticker": ticker,
                "bucket_classification": packet["bucket_classification"],
                "packet_confidence": packet["packet_confidence"],
                "downstream_readiness_flag": downstream_ready,
                "downstream_readiness_reason": downstream_readiness_reason,
                "publishable_flag": publishable_flag,
                "publishable_gate_reason": publishable_gate_reason,
                "rating_status": rating_status,
                "final_rating_confidence": final_rating_confidence,
                "regime_label": packet["regime_label"],
                "route_risk_label": packet["route_risk_label"],
                "suggested_discount_rate_uplift_bps": packet["suggested_discount_rate_uplift_bps"],
                "packet_path": packet_path_index,
            }
        )

    index_df = pd.DataFrame(index_rows).sort_values(["bucket_classification", "ticker"]).reset_index(drop=True)
    return index_df
