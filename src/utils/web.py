from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from src.cache import DiskCache

logger = logging.getLogger(__name__)


class WebClient:
    """HTTP client with retry and optional cache integration."""

    def __init__(
        self,
        cache: Optional[DiskCache] = None,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        retry_wait_seconds: int = 2,
        user_agent: str = "hormuz-research-bot/1.0",
    ) -> None:
        self.cache = cache
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds
        self.headers = {"User-Agent": user_agent}

    def _cache_key(self, url: str, params: Optional[dict[str, Any]]) -> str:
        payload = {"url": url, "params": params or {}}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return f"http::{digest}"

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def _request(self, url: str, params: Optional[dict[str, Any]]) -> requests.Response:
        response = requests.get(url, params=params, headers=self.headers, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response

    def get_json(self, url: str, params: Optional[dict[str, Any]] = None, ttl_hours: int = 24) -> dict[str, Any]:
        key = self._cache_key(url, params)
        if self.cache:
            cached = self.cache.get_json(key, max_age_hours=ttl_hours)
            if cached is not None:
                return cached

        response = self._request(url, params)
        payload = response.json()
        if self.cache:
            self.cache.set_json(key, payload)
        return payload

    def get_text(self, url: str, params: Optional[dict[str, Any]] = None) -> str:
        response = self._request(url, params)
        return response.text
