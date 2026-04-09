from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.models import AssumptionRecord, MissingDataRecord


class AssumptionsRegistry:
    """Stores every estimated value and rationale for auditability."""

    def __init__(self) -> None:
        self._entries: list[AssumptionRecord] = []

    def add(
        self,
        field_name: str,
        company: str,
        estimate_value: str,
        estimate_type: str,
        reasoning: str,
        source_urls: Iterable[str],
        confidence: str,
        model_version: str | None = None,
    ) -> None:
        record = AssumptionRecord(
            field_name=field_name,
            company=company,
            estimate_value=estimate_value,
            estimate_type=estimate_type,
            reasoning=reasoning,
            source_urls=list(source_urls),
            confidence=confidence,
            model_version=model_version,
            timestamp=datetime.now(UTC).replace(tzinfo=None),
        )
        self._entries.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame(
                columns=[
                    "field_name",
                    "company",
                    "estimate_value",
                    "estimate_type",
                    "reasoning",
                    "source_urls",
                    "confidence",
                    "model_version",
                    "timestamp",
                ]
            )
        return pd.DataFrame([entry.model_dump() for entry in self._entries])

    def export_csv(self, path: Path) -> Path:
        df = self.to_dataframe()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path


class MissingDataLogger:
    """Captures missing fields and failed source attempts without failing the full run."""

    def __init__(self) -> None:
        self._entries: list[MissingDataRecord] = []

    def add(
        self,
        company: str,
        field_name: str,
        reason: str,
        attempted_sources: Iterable[str],
        severity: str | None = None,
    ) -> None:
        self._entries.append(
            MissingDataRecord(
                company=company,
                field_name=field_name,
                reason=reason,
                attempted_sources=list(attempted_sources),
                severity=severity,
                timestamp=datetime.now(UTC).replace(tzinfo=None),
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame(
                columns=["company", "field_name", "reason", "attempted_sources", "severity", "timestamp"]
            )
        return pd.DataFrame([entry.model_dump() for entry in self._entries])

    def export_csv(self, path: Path) -> Path:
        df = self.to_dataframe()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

