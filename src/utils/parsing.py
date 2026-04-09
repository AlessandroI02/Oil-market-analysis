from __future__ import annotations

from typing import Iterable


def compact_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split())


def parse_links(raw: Iterable[str] | str | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        if not raw.strip():
            return []
        return [raw.strip()]
    return [item.strip() for item in raw if item and item.strip()]
