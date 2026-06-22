"""
Tiny API client for the AML Platform API.

Usage:
  from api_client import AMLApi
  api = AMLApi("http://localhost:8050", api_key="dev-key")
  api.health()
"""
from __future__ import annotations

import requests
from typing import Any, Optional

class AMLApi:
    def __init__(self, base_url: str, api_key: str, timeout: int = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key}

    def health(self) -> dict[str, Any]:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def list_transactions(self, limit: int = 500) -> dict[str, Any]:
        r = requests.get(
            f"{self.base_url}/transactions",
            params={"limit": limit},
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def ingest_transactions(self, transactions: list[dict[str, Any]]) -> dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/transactions/ingest",
            json={"transactions": transactions},
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def list_cases(
            self,
            status: str | None = None,
            assigned_to: str | None = None,
            tier: str | None = None,
            entity_id: str | None = None,
            q: str | None = None,
            page: int = 1,
            page_size: int = 50,
            sort: str = "updated_at",
            order: str = "desc",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "page": page,
            "page_size": page_size,
            "sort": sort,
            "order": order,
        }
        if status:
            params["status"] = status
        if assigned_to:
            params["assigned_to"] = assigned_to
        if tier:
            params["tier"] = tier
        if entity_id:
            params["entity_id"] = entity_id
        if q:
            params["q"] = q

        r = requests.get(
            f"{self.base_url}/cases",
            params=params,
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def create_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/cases",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def get_case(self, case_id: str) -> dict[str, Any]:
        r = requests.get(
            f"{self.base_url}/cases/{case_id}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def update_case(self, case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        r = requests.patch(
            f"{self.base_url}/cases/{case_id}",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def add_case_event(self, case_id: str, event_type: str, message: str) -> dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/cases/{case_id}/events",
            json={"event_type": event_type, "message": message},
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def audit(self, limit: int = 2000) -> dict[str, Any]:
        r = requests.get(
            f"{self.base_url}/audit",
            params={"limit": limit},
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
