from __future__ import annotations

from datetime import timedelta

import pandas as pd


def _confidence_rank(label: str) -> int:
    mapping = {"Low": 1, "Medium": 2, "High": 3}
    return mapping.get(str(label), 1)


def _confidence_label(value: float) -> str:
    if value >= 2.5:
        return "High"
    if value >= 1.5:
        return "Medium"
    return "Low"


def _tier_rank(value: str) -> int:
    return {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3, "Tier 4": 4}.get(str(value), 4)


def _episode_type_from_row(row: pd.Series) -> str:
    candidate = str(row.get("candidate_catalyst", "")).strip()
    if candidate and candidate.lower() != "unattributed":
        return candidate
    theme = str(row.get("theme", "")).strip()
    return theme if theme else "Unattributed"


def cluster_event_episodes(
    events_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    gap_days: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events_df is None or events_df.empty:
        episode_cols = [
            "episode_id",
            "start_date",
            "end_date",
            "episode_type",
            "summary",
            "involved_event_days",
            "cumulative_brent_move",
            "cumulative_wti_move",
            "mean_abs_move",
            "peak_move",
            "attribution_confidence",
            "top_articles",
            "episode_source_summary",
        ]
        return pd.DataFrame(columns=episode_cols), pd.DataFrame(columns=["episode_id"])

    events = events_df.copy()
    events["event_date"] = pd.to_datetime(events.get("date"), errors="coerce").dt.date
    events = events.dropna(subset=["event_date"]).sort_values("event_date").reset_index(drop=True)
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    events["episode_type"] = events.apply(_episode_type_from_row, axis=1)

    episode_numbers: list[int] = []
    current_episode = 0
    prev_date = None
    prev_type = None
    for _, row in events.iterrows():
        dt = row["event_date"]
        typ = row["episode_type"]
        start_new = False
        if prev_date is None:
            start_new = True
        else:
            if dt - prev_date > timedelta(days=gap_days):
                start_new = True
            elif typ != prev_type and dt - prev_date > timedelta(days=1):
                start_new = True
        if start_new:
            current_episode += 1
        episode_numbers.append(current_episode)
        prev_date = dt
        prev_type = typ

    events["episode_num"] = episode_numbers
    events["episode_id"] = events["episode_num"].map(lambda x: f"EP{x:03d}")

    articles = articles_df.copy() if articles_df is not None else pd.DataFrame()
    if not articles.empty:
        articles["event_date"] = pd.to_datetime(articles.get("event_date"), errors="coerce").dt.date
        articles = articles.merge(events[["event_date", "episode_id"]], on="event_date", how="left")
        articles["episode_id"] = articles["episode_id"].fillna("UNASSIGNED")
    else:
        articles = pd.DataFrame(columns=["episode_id"])

    rows: list[dict[str, object]] = []
    for episode_id, grp in events.groupby("episode_id", as_index=False):
        grp = grp.sort_values("event_date")
        start = grp["event_date"].iloc[0]
        end = grp["event_date"].iloc[-1]
        etype = str(grp["episode_type"].mode().iloc[0]) if "episode_type" in grp.columns else "Unattributed"
        brent_move = pd.to_numeric(grp.get("brent_daily_move_pct"), errors="coerce").fillna(0.0)
        wti_move = pd.to_numeric(grp.get("wti_daily_move_pct"), errors="coerce").fillna(0.0)
        conf_series = grp.get("attribution_confidence", pd.Series(dtype=str)).astype(str)
        conf = _confidence_label(conf_series.map(_confidence_rank).mean() if not conf_series.empty else 1.0)

        involved_days = ", ".join(d.isoformat() for d in grp["event_date"].tolist())
        episode_articles = articles[articles.get("episode_id") == episode_id].copy() if not articles.empty else pd.DataFrame()

        if not episode_articles.empty:
            episode_articles["tier_rank"] = episode_articles.get("source_quality_tier", pd.Series(dtype=str)).astype(str).map(_tier_rank).fillna(4)
            sort_cols: list[str] = []
            ascending: list[bool] = []
            if "tier_rank" in episode_articles.columns:
                sort_cols.append("tier_rank")
                ascending.append(True)
            if "relevance_score" in episode_articles.columns:
                sort_cols.append("relevance_score")
                ascending.append(False)
            if "article_date" in episode_articles.columns:
                sort_cols.append("article_date")
                ascending.append(False)
            if sort_cols:
                episode_articles = episode_articles.sort_values(sort_cols, ascending=ascending)
            dedupe_cols = [c for c in ["headline", "publication"] if c in episode_articles.columns]
            if dedupe_cols:
                episode_articles = episode_articles.drop_duplicates(subset=dedupe_cols, keep="first")
            top = episode_articles.head(3).copy()
            top_headlines = top.apply(
                lambda r: f"{r.get('publication', '')}: {r.get('headline', '')}",
                axis=1,
            ).tolist()
            top_articles = " | ".join(top_headlines)
            domains = episode_articles.get("domain", pd.Series(dtype=str)).astype(str)
            domain_counts = domains[domains != ""].value_counts()
            tier_counts = episode_articles.get("source_quality_tier", pd.Series(dtype=str)).astype(str).value_counts()
            domain_summary = ", ".join([f"{idx}:{val}" for idx, val in domain_counts.head(4).items()])
            tier_summary = ", ".join([f"{idx}:{val}" for idx, val in tier_counts.head(3).items()])
            source_summary = "; ".join([s for s in [domain_summary, tier_summary] if s])
        else:
            top_articles = ""
            source_summary = ""

        summary = (
            f"{etype} episode from {start.isoformat()} to {end.isoformat()} "
            f"with {len(grp)} event days and cumulative Brent move {brent_move.sum():.2f}%."
        )

        rows.append(
            {
                "episode_id": episode_id,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "episode_type": etype,
                "summary": summary,
                "involved_event_days": involved_days,
                "cumulative_brent_move": float(brent_move.sum()),
                "cumulative_wti_move": float(wti_move.sum()),
                "mean_abs_move": float(brent_move.abs().mean()),
                "peak_move": float(brent_move.abs().max()),
                "attribution_confidence": conf,
                "top_articles": top_articles,
                "episode_source_summary": source_summary,
            }
        )

    episodes_df = pd.DataFrame(rows).sort_values("start_date").reset_index(drop=True)
    return episodes_df, articles
