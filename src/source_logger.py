from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models import SourceLogEntry


class SourceLogger:
    """Collects data lineage references for all populated fields."""

    def __init__(self) -> None:
        self._entries: list[SourceLogEntry] = []

    def add(
        self,
        company: str,
        field: str,
        source_url: str,
        evidence_flag: str,
        source_tier: Optional[str] = None,
        source_title: Optional[str] = None,
        provider_used: Optional[str] = None,
        comments: Optional[str] = None,
    ) -> None:
        self._entries.append(
            SourceLogEntry(
                company=company,
                field=field,
                source_url=source_url,
                source_title=source_title,
                source_tier=source_tier,
                provider_used=provider_used,
                access_date=datetime.now(UTC).replace(tzinfo=None),
                evidence_flag=evidence_flag,
                comments=comments,
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame(
                columns=[
                    "company",
                    "field",
                    "source_url",
                    "source_title",
                    "source_tier",
                    "provider_used",
                    "access_date",
                    "evidence_flag",
                    "comments",
                ]
            )
        return pd.DataFrame([entry.model_dump() for entry in self._entries])

    def export_csv(self, path: Path) -> Path:
        df = self.to_dataframe()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

