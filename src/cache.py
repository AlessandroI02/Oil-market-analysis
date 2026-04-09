from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional


class DiskCache:
    """Simple file-backed cache for API responses and intermediate objects."""

    def __init__(self, cache_dir: Path, rebuild: bool = False) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rebuild = rebuild

    def _path_for_key(self, key: str, suffix: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.{suffix}"

    def get_json(self, key: str, max_age_hours: Optional[int] = None) -> Optional[dict[str, Any]]:
        if self.rebuild:
            return None

        path = self._path_for_key(key, "json")
        if not path.exists():
            return None
        if max_age_hours is not None:
            age_limit = datetime.utcnow() - timedelta(hours=max_age_hours)
            if datetime.utcfromtimestamp(path.stat().st_mtime) < age_limit:
                return None

        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def set_json(self, key: str, payload: dict[str, Any]) -> Path:
        path = self._path_for_key(key, "json")
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return path

    def get_pickle(self, key: str, max_age_hours: Optional[int] = None) -> Any:
        if self.rebuild:
            return None

        path = self._path_for_key(key, "pkl")
        if not path.exists():
            return None
        if max_age_hours is not None:
            age_limit = datetime.utcnow() - timedelta(hours=max_age_hours)
            if datetime.utcfromtimestamp(path.stat().st_mtime) < age_limit:
                return None
        with path.open("rb") as fh:
            return pickle.load(fh)

    def set_pickle(self, key: str, payload: Any) -> Path:
        path = self._path_for_key(key, "pkl")
        with path.open("wb") as fh:
            pickle.dump(payload, fh)
        return path
