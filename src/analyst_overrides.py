from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass
class OverridePackage:
    raw: dict[str, Any]
    companies: dict[str, dict[str, Any]]


def load_overrides(path: Path) -> OverridePackage:
    if not path.exists():
        return OverridePackage(raw={}, companies={})
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    companies = raw.get("companies", {}) or {}
    return OverridePackage(raw=raw, companies=companies)


def apply_universe_overrides(universe_df: pd.DataFrame, overrides: OverridePackage) -> pd.DataFrame:
    if universe_df.empty:
        return universe_df

    out = universe_df.copy()
    out["override_applied"] = False
    out["override_notes"] = ""

    for idx, row in out.iterrows():
        ticker = str(row.get("ticker"))
        ov = overrides.companies.get(ticker)
        if not ov:
            continue

        changed = False

        include_flag = ov.get("include_flag")
        if include_flag is False:
            out.at[idx, "bucket_classification"] = "rejected"
            out.at[idx, "reason_excluded"] = "Analyst override exclude"
            changed = True

        bucket_override = ov.get("bucket_override")
        if bucket_override in {"primary", "secondary", "rejected"}:
            out.at[idx, "bucket_classification"] = bucket_override
            changed = True

        conf_override = ov.get("confidence_override")
        if conf_override in {"High", "Medium", "Low"}:
            out.at[idx, "confidence"] = conf_override
            changed = True

        notes = ov.get("analyst_notes") or ""
        if notes:
            out.at[idx, "override_notes"] = str(notes)
            changed = True

        out.at[idx, "override_applied"] = changed

    return out


def apply_profile_overrides(profiles_df: pd.DataFrame, overrides: OverridePackage) -> pd.DataFrame:
    if profiles_df.empty:
        return profiles_df

    out = profiles_df.copy()
    out["override_applied"] = False
    out["override_notes"] = ""

    for idx, row in out.iterrows():
        ticker = str(row.get("ticker"))
        ov = overrides.companies.get(ticker)
        if not ov:
            continue

        changed = False

        route_ov = ov.get("route_weight_overrides") or {}
        retail_ov = ov.get("retail_weight_overrides") or {}
        conf_override = ov.get("confidence_override")
        notes = ov.get("analyst_notes") or ""

        if route_ov:
            for key, val in route_ov.items():
                if key in {"production_region_weights", "refinery_region_weights", "destination_market_weights"} and isinstance(val, dict):
                    out.at[idx, key] = val
                    changed = True

        if retail_ov and isinstance(retail_ov, dict):
            out.at[idx, "retail_country_weights"] = retail_ov
            changed = True

        if conf_override in {"High", "Medium", "Low"}:
            out.at[idx, "profile_confidence"] = conf_override
            changed = True

        if notes:
            out.at[idx, "override_notes"] = str(notes)
            changed = True

        out.at[idx, "override_applied"] = changed

    return out


def build_overrides_log(overrides: OverridePackage) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, payload in overrides.companies.items():
        rows.append(
            {
                "ticker": ticker,
                "override_payload": str(payload),
                "analyst_notes": payload.get("analyst_notes", ""),
            }
        )
    return pd.DataFrame(rows)
