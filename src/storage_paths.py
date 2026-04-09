from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_storage_layout(root: Path | None = None) -> dict[str, Path]:
    base = root or project_root()

    paths = {
        "data_raw_market": base / "data" / "raw" / "market",
        "data_raw_news": base / "data" / "raw" / "news",
        "data_raw_reference": base / "data" / "raw" / "reference",
        "data_interim_market": base / "data" / "interim" / "market",
        "data_interim_news": base / "data" / "interim" / "news",
        "data_interim_reference": base / "data" / "interim" / "reference",
        "data_cache": base / "data" / "cache",
        "output_packets": base / "outputs" / "packets" / "company_case_packets",
        "schemas": base / "schemas",
        "docs": base / "docs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path
