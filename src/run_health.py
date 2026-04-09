from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _numeric_non_null(df: pd.DataFrame, col: str) -> int:
    if df is None or df.empty or col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").notna().sum())


def _non_null_ratio(df: pd.DataFrame, cols: list[str]) -> float:
    if df is None or df.empty or not cols:
        return 0.0
    available = [c for c in cols if c in df.columns]
    if not available:
        return 0.0
    block = df[available].apply(pd.to_numeric, errors="coerce")
    denom = len(block) * len(available)
    if denom == 0:
        return 0.0
    return float(block.notna().sum().sum() / denom)


def _coverage_ratio(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return 0.0
    total = len(df)
    if total == 0:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").notna().sum() / total)


def _append_check(
    rows: list[dict[str, Any]],
    check_name: str,
    category: str,
    severity: str,
    passed: bool,
    observed: Any,
    threshold: Any,
    message: str,
) -> None:
    rows.append(
        {
            "check_name": check_name,
            "category": category,
            "severity": severity,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "threshold": threshold,
            "message": message,
        }
    )


def assess_run_health(
    datasets: dict[str, pd.DataFrame],
    quality_cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    crude = datasets.get("Crude_Tracker", pd.DataFrame())
    fuel = datasets.get("Fuel_Tracker", pd.DataFrame())
    equity = datasets.get("Equity_Tracker", pd.DataFrame())
    valuation = datasets.get("Valuation", pd.DataFrame())
    factor = datasets.get("Factor_Decomposition", pd.DataFrame())
    analogues = datasets.get("Historical_Analogues", pd.DataFrame())
    core_rank = datasets.get("Core_Ranking", pd.DataFrame())

    min_price_points = int(quality_cfg.get("minimum_required_price_points", 4))
    min_equity_names = int(quality_cfg.get("minimum_required_equity_names", 4))
    min_val_cov = float(quality_cfg.get("minimum_required_valuation_coverage", 0.35))
    min_factor_cov = float(quality_cfg.get("minimum_required_factor_coverage", 0.35))
    min_section_ratio = float(quality_cfg.get("minimum_non_null_ratio_for_section", 0.25))

    brent_points = _numeric_non_null(crude, "brent_price")
    wti_points = _numeric_non_null(crude, "wti_price")
    crude_ratio = _non_null_ratio(crude, ["brent_price", "wti_price"])
    fuel_ratio = _non_null_ratio(fuel, ["blended_petrol_price", "blended_diesel_price", "blended_combined_fuels_price"])
    equity_names = int(equity["ticker"].dropna().nunique()) if (not equity.empty and "ticker" in equity.columns) else 0
    valuation_cov = max(_coverage_ratio(valuation, "market_cap_usd"), _coverage_ratio(valuation, "valuation_score"))
    factor_cov = _coverage_ratio(factor, "beta_brent_ret")
    analogue_cov = _coverage_ratio(analogues, "brent_return_pct")

    ranking_default_share = (
        pd.to_numeric(core_rank.get("score_default_share"), errors="coerce").mean()
        if (not core_rank.empty and "score_default_share" in core_rank.columns)
        else 1.0
    )

    checks: list[dict[str, Any]] = []

    _append_check(
        checks,
        "brent_price_points",
        "market_backbone",
        "critical",
        brent_points >= min_price_points,
        brent_points,
        min_price_points,
        "Brent weekly series has enough non-null observations.",
    )
    _append_check(
        checks,
        "wti_price_points",
        "market_backbone",
        "major",
        wti_points >= min_price_points,
        wti_points,
        min_price_points,
        "WTI weekly series has enough non-null observations.",
    )
    _append_check(
        checks,
        "crude_non_null_ratio",
        "market_backbone",
        "critical",
        crude_ratio >= min_section_ratio,
        round(crude_ratio, 4),
        min_section_ratio,
        "Crude tracker has meaningful population on benchmark columns.",
    )
    _append_check(
        checks,
        "fuel_non_null_ratio",
        "market_backbone",
        "critical",
        fuel_ratio >= min_section_ratio,
        round(fuel_ratio, 4),
        min_section_ratio,
        "Fuel tracker has meaningful population on blended price columns.",
    )
    _append_check(
        checks,
        "equity_name_coverage",
        "market_backbone",
        "critical",
        equity_names >= min_equity_names,
        equity_names,
        min_equity_names,
        "Equity tracker contains enough covered names.",
    )
    _append_check(
        checks,
        "valuation_coverage",
        "analytical_depth",
        "major",
        valuation_cov >= min_val_cov,
        round(valuation_cov, 4),
        min_val_cov,
        "Valuation table has minimum core-field coverage.",
    )
    _append_check(
        checks,
        "factor_coverage",
        "analytical_depth",
        "major",
        factor_cov >= min_factor_cov,
        round(factor_cov, 4),
        min_factor_cov,
        "Factor decomposition has enough populated observations.",
    )
    _append_check(
        checks,
        "analogue_coverage",
        "analytical_depth",
        "major",
        analogue_cov >= min_section_ratio,
        round(analogue_cov, 4),
        min_section_ratio,
        "Historical analogue section has usable return data.",
    )
    _append_check(
        checks,
        "ranking_default_share",
        "ranking_integrity",
        "critical",
        ranking_default_share <= 0.45,
        round(float(ranking_default_share), 4),
        0.45,
        "Core ranking is not dominated by neutral/default filler components.",
    )

    checks_df = pd.DataFrame(checks)
    failed = checks_df[checks_df["status"] == "FAIL"]

    critical_fails = failed[failed["severity"] == "critical"]
    major_fails = failed[failed["severity"] == "major"]

    if not critical_fails.empty:
        run_status = "INVALID"
    elif not major_fails.empty:
        run_status = "DEGRADED"
    else:
        run_status = "VALID"

    summary = {
        "run_status": run_status,
        "generated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        "critical_fail_count": int(len(critical_fails)),
        "major_fail_count": int(len(major_fails)),
        "check_count": int(len(checks_df)),
        "invalid_reasons": critical_fails["check_name"].tolist(),
        "degraded_reasons": major_fails["check_name"].tolist(),
        "metrics": {
            "brent_points": brent_points,
            "wti_points": wti_points,
            "crude_non_null_ratio": crude_ratio,
            "fuel_non_null_ratio": fuel_ratio,
            "equity_name_coverage": equity_names,
            "valuation_coverage": valuation_cov,
            "factor_coverage": factor_cov,
            "analogue_coverage": analogue_cov,
            "ranking_default_share": float(ranking_default_share) if pd.notna(ranking_default_share) else None,
        },
    }

    checks_df.insert(0, "run_status", run_status)
    return summary, checks_df


def export_run_health(debug_dir: Path, summary: dict[str, Any], checks_df: pd.DataFrame) -> dict[str, Path]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    csv_path = debug_dir / "run_health.csv"
    json_path = debug_dir / "run_health.json"

    checks_df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return {"run_health_csv": csv_path, "run_health_json": json_path}
