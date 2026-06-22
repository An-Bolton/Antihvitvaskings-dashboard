from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel

from .db import get_conn, ensure_case_tables

router = APIRouter()

API_KEY = os.getenv("AML_API_KEY", "dev-key")

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class SubmitRequest(BaseModel):
    actor: str
    comment: str | None = None


class ApproveRequest(BaseModel):
    actor: str
    comment: str | None = None


class RejectRequest(BaseModel):
    actor: str
    comment: str | None = None


def require_key(x_api_key: Optional[str]) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
