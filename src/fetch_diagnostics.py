from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


@dataclass
class FetchAttempt:
    dataset: str
    identifier: str
    provider: str
    status: str
    source_url: str
    message: str
    row_count: int
    non_null_count: int
    attempted_at: datetime


class FetchDiagnostics:
    """Structured diagnostics for multi-provider market-data fetches."""

    def __init__(self) -> None:
        self._attempts: list[FetchAttempt] = []
        self._provider_usage: list[dict[str, object]] = []

    def log_attempt(
        self,
        dataset: str,
        identifier: str,
        provider: str,
        status: str,
        source_url: str,
        message: str,
        row_count: int = 0,
        non_null_count: int = 0,
    ) -> None:
        self._attempts.append(
            FetchAttempt(
                dataset=dataset,
                identifier=identifier,
                provider=provider,
                status=status,
                source_url=source_url,
                message=message,
                row_count=int(row_count),
                non_null_count=int(non_null_count),
                attempted_at=datetime.now(UTC).replace(tzinfo=None),
            )
        )

    def log_provider_usage(
        self,
        dataset: str,
        identifier: str,
        selected_provider: str,
        fallback_used: bool,
        status: str,
        notes: str = "",
    ) -> None:
        self._provider_usage.append(
            {
                "dataset": dataset,
                "identifier": identifier,
                "selected_provider": selected_provider,
                "fallback_used": bool(fallback_used),
                "status": status,
                "notes": notes,
                "recorded_at": datetime.now(UTC).replace(tzinfo=None),
            }
        )

    def attempts_df(self) -> pd.DataFrame:
        if not self._attempts:
            return pd.DataFrame(
                columns=[
                    "dataset",
                    "identifier",
                    "provider",
                    "status",
                    "source_url",
                    "message",
                    "row_count",
                    "non_null_count",
                    "attempted_at",
                ]
            )
        return pd.DataFrame([a.__dict__ for a in self._attempts])

    def provider_usage_df(self) -> pd.DataFrame:
        if not self._provider_usage:
            return pd.DataFrame(
                columns=[
                    "dataset",
                    "identifier",
                    "selected_provider",
                    "fallback_used",
                    "status",
                    "notes",
                    "recorded_at",
                ]
            )
        return pd.DataFrame(self._provider_usage)

    def export(self, debug_dir: Path) -> dict[str, Path]:
        debug_dir.mkdir(parents=True, exist_ok=True)
        attempts_path = debug_dir / "fetch_attempts.csv"
        usage_path = debug_dir / "provider_usage.csv"
        self.attempts_df().to_csv(attempts_path, index=False)
        self.provider_usage_df().to_csv(usage_path, index=False)
        return {"fetch_attempts": attempts_path, "provider_usage": usage_path}
