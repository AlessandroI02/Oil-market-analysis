from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd


def _to_score(value: float, low: float, high: float) -> float:
    if pd.isna(value):
        return 0.0
    if high <= low:
        return 50.0
    scaled = (value - low) / (high - low) * 100.0
    return float(np.clip(scaled, 0.0, 100.0))


def _label_confidence(total_score: float, label: str) -> str:
    if label in {"mixed / transition regime", "normal"}:
        if total_score >= 70:
            return "Medium"
        return "Low"
    if total_score >= 80:
        return "High"
    if total_score >= 60:
        return "Medium"
    return "Low"


def _pick_label(
    scores: dict[str, float],
    dominant_episode_type: str,
) -> str:
    volatility = scores["oil_volatility"]
    direction = scores["oil_direction"]
    move_freq = scores["large_move_frequency"]
    geopolitical = scores["geopolitical_news"]
    factor = scores["factor_dislocation"]
    route = scores["route_risk"]
    episode = scores["episode_intensity"]

    dominant = dominant_episode_type.lower()
    if "sanction" in dominant and geopolitical >= 50:
        return "sanctions-driven regime"
    if route >= 60 and geopolitical >= 60:
        return "route-disruption regime"
    if geopolitical >= 65 and direction >= 50:
        return "geopolitical supply-shock regime"
    if volatility >= 60 and move_freq >= 55 and direction >= 50:
        return "stressed oil regime"
    if factor >= 60 and direction < 45:
        return "macro risk-off regime"
    if episode >= 50 and (40 <= direction <= 60):
        return "mixed / transition regime"
    return "normal"


def _regime_notes(
    label: str,
    scores: dict[str, float],
    dominant_episode_type: str,
) -> str:
    notes = [
        f"vol={scores['oil_volatility']:.1f}",
        f"direction={scores['oil_direction']:.1f}",
        f"large-move={scores['large_move_frequency']:.1f}",
        f"geo-news={scores['geopolitical_news']:.1f}",
        f"factor={scores['factor_dislocation']:.1f}",
        f"route={scores['route_risk']:.1f}",
        f"episode={scores['episode_intensity']:.1f}",
    ]
    dominant = dominant_episode_type if dominant_episode_type else "none"
    return f"{label} classified from score blend ({', '.join(notes)}); dominant episode type={dominant}."


def _episode_intensity(
    event_episodes_df: pd.DataFrame,
    max_date: pd.Timestamp | None = None,
) -> tuple[float, float, str]:
    if event_episodes_df is None or event_episodes_df.empty:
        return 0.0, 0.0, "none"

    episodes = event_episodes_df.copy()
    episodes["end_date"] = pd.to_datetime(episodes.get("end_date"), errors="coerce")
    episodes = episodes.dropna(subset=["end_date"])
    if max_date is not None:
        episodes = episodes[episodes["end_date"] <= max_date]
    if episodes.empty:
        return 0.0, 0.0, "none"

    recent = episodes.sort_values("end_date").tail(8)
    abs_move = pd.to_numeric(recent.get("mean_abs_move"), errors="coerce").fillna(0.0).mean()
    peak = pd.to_numeric(recent.get("peak_move"), errors="coerce").fillna(0.0).max()
    top_type = str(recent.get("episode_type", pd.Series(dtype=str)).mode().iloc[0]) if "episode_type" in recent.columns else "none"
    return float(abs_move), float(peak), top_type


def _route_risk_score(route_risks_df: pd.DataFrame) -> float:
    if route_risks_df is None or route_risks_df.empty:
        return 0.0
    risk = route_risks_df.get("qualitative_route_risk", pd.Series(dtype=str)).astype(str).str.lower()
    high = (risk == "high").mean() if len(risk) else 0.0
    medium = (risk == "medium").mean() if len(risk) else 0.0
    return float(np.clip((high * 100) + (medium * 45), 0.0, 100.0))


def _build_history_row(
    benchmark_returns_df: pd.DataFrame,
    route_score: float,
    event_episodes_df: pd.DataFrame,
    idx: int,
) -> dict[str, Any]:
    hist = benchmark_returns_df.iloc[: idx + 1].copy()
    window = hist.tail(20)
    brent_ret = pd.to_numeric(window.get("brent_ret"), errors="coerce").dropna()
    xle_ret = pd.to_numeric(window.get("xle_ret"), errors="coerce")
    spy_ret = pd.to_numeric(window.get("spy_ret"), errors="coerce")

    oil_vol = _to_score(float(brent_ret.std()) if not brent_ret.empty else 0.0, 0.004, 0.03)
    oil_dir = _to_score(float(brent_ret.sum()) if not brent_ret.empty else 0.0, -0.10, 0.10)
    move_freq = _to_score(float((brent_ret.abs() > 0.02).mean()) if not brent_ret.empty else 0.0, 0.05, 0.45)
    geo_abs, geo_peak, dominant_type = _episode_intensity(event_episodes_df, max_date=pd.to_datetime(hist.iloc[-1]["date"]))
    geo_news = _to_score(geo_abs + (geo_peak * 0.3), 0.2, 4.5)

    disloc = (xle_ret - spy_ret).abs()
    factor_dis = _to_score(float(disloc.mean()) if disloc.notna().any() else 0.0, 0.004, 0.03)
    episode_score = _to_score((geo_abs * 0.7) + (geo_peak * 0.3), 0.2, 4.0)

    scores = {
        "oil_volatility": oil_vol,
        "oil_direction": oil_dir,
        "large_move_frequency": move_freq,
        "geopolitical_news": geo_news,
        "factor_dislocation": factor_dis,
        "route_risk": route_score,
        "episode_intensity": episode_score,
    }
    total = float(np.mean(list(scores.values())))
    label = _pick_label(scores, dominant_type)
    confidence = _label_confidence(total, label)
    return {
        "date": pd.to_datetime(hist.iloc[-1]["date"]).date(),
        "regime_label": label,
        "regime_confidence": confidence,
        "regime_score": round(total, 4),
    }


def build_regime_state(
    crude_tracker_df: pd.DataFrame,
    benchmark_returns_df: pd.DataFrame,
    route_risks_df: pd.DataFrame,
    event_episodes_df: pd.DataFrame,
    as_of_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if benchmark_returns_df is None or benchmark_returns_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    bmk = benchmark_returns_df.copy()
    bmk["date"] = pd.to_datetime(bmk["date"], errors="coerce")
    bmk = bmk.dropna(subset=["date"]).sort_values("date")
    if bmk.empty:
        return pd.DataFrame(), pd.DataFrame()

    if as_of_date is not None:
        bmk = bmk[bmk["date"].dt.date <= as_of_date]
    if bmk.empty:
        return pd.DataFrame(), pd.DataFrame()

    window = bmk.tail(20)
    brent_ret = pd.to_numeric(window.get("brent_ret"), errors="coerce").dropna()
    xle_ret = pd.to_numeric(window.get("xle_ret"), errors="coerce")
    spy_ret = pd.to_numeric(window.get("spy_ret"), errors="coerce")

    oil_vol = _to_score(float(brent_ret.std()) if not brent_ret.empty else 0.0, 0.004, 0.03)
    oil_dir = _to_score(float(brent_ret.sum()) if not brent_ret.empty else 0.0, -0.10, 0.10)
    move_freq = _to_score(float((brent_ret.abs() > 0.02).mean()) if not brent_ret.empty else 0.0, 0.05, 0.45)
    geo_abs, geo_peak, dominant_type = _episode_intensity(event_episodes_df)
    geo_news = _to_score(geo_abs + (geo_peak * 0.3), 0.2, 4.5)

    disloc = (xle_ret - spy_ret).abs()
    factor_dis = _to_score(float(disloc.mean()) if disloc.notna().any() else 0.0, 0.004, 0.03)
    route_score = _route_risk_score(route_risks_df)
    episode_score = _to_score((geo_abs * 0.7) + (geo_peak * 0.3), 0.2, 4.0)

    scores = {
        "oil_volatility": oil_vol,
        "oil_direction": oil_dir,
        "large_move_frequency": move_freq,
        "geopolitical_news": geo_news,
        "factor_dislocation": factor_dis,
        "route_risk": route_score,
        "episode_intensity": episode_score,
    }
    total_score = float(np.mean(list(scores.values())))
    label = _pick_label(scores, dominant_type)
    confidence = _label_confidence(total_score, label)
    notes = _regime_notes(label, scores, dominant_type)

    latest_crude = crude_tracker_df.copy() if crude_tracker_df is not None else pd.DataFrame()
    latest_brent = np.nan
    if not latest_crude.empty and "brent_price" in latest_crude.columns:
        latest_crude = latest_crude.sort_values("date")
        latest_brent = pd.to_numeric(latest_crude["brent_price"], errors="coerce").dropna().iloc[-1] if pd.to_numeric(latest_crude["brent_price"], errors="coerce").notna().any() else np.nan

    state = pd.DataFrame(
        [
            {
                "as_of_date": (as_of_date or bmk["date"].max().date()).isoformat(),
                "regime_label": label,
                "regime_confidence": confidence,
                "regime_notes": notes,
                "regime_summary": notes,
                "latest_brent_price": latest_brent,
                "score_oil_volatility": round(scores["oil_volatility"], 4),
                "score_oil_direction": round(scores["oil_direction"], 4),
                "score_large_move_frequency": round(scores["large_move_frequency"], 4),
                "score_geopolitical_news": round(scores["geopolitical_news"], 4),
                "score_factor_dislocation": round(scores["factor_dislocation"], 4),
                "score_route_risk": round(scores["route_risk"], 4),
                "score_episode_intensity": round(scores["episode_intensity"], 4),
                "regime_score_total": round(total_score, 4),
            }
        ]
    )

    history_rows = [
        _build_history_row(
            benchmark_returns_df=bmk,
            route_score=route_score,
            event_episodes_df=event_episodes_df,
            idx=idx,
        )
        for idx in range(len(bmk))
    ]
    history = pd.DataFrame(history_rows)
    return state, history
