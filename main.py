"""
AML Platform API (FastAPI)

Run:
  pip install fastapi uvicorn
  export AML_DB_PATH="hvitvask.db"
  export AML_API_KEY="dev-key"   # change in prod
  uvicorn api:app --reload --port 8000

Auth:
  Send header: X-API-Key: <key>
"""
from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


DB_PATH = os.getenv("AML_DB_PATH", "hvitvask.db")
API_KEY = os.getenv("AML_API_KEY", "dev-key")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_tables() -> None:
    """
    Best-effort: ensure minimal tables exist. If your dashboard already created them,
    this is no-op.
    """
    with get_conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS transaksjoner (
              trans_id TEXT PRIMARY KEY,
              fra_konto TEXT,
              til_konto TEXT,
              belop REAL,
              valuta TEXT,
              timestamp TEXT,
              kunde_id TEXT,
              mottaker_id TEXT,
              land TEXT,
              mistenkelig INTEGER DEFAULT 0,
              mistenkelig_ml INTEGER DEFAULT 0,
              sanksjonert INTEGER DEFAULT 0,
              fuzzy_sanksjonert INTEGER DEFAULT 0,
              risikoscore REAL
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
              case_id TEXT PRIMARY KEY,
              trans_id TEXT,
              status TEXT,
              priority TEXT,
              owner TEXT,
              tags TEXT,
              note TEXT,
              created_at TEXT,
              updated_at TEXT,
              entity_key TEXT,
              submitted_by TEXT,
              submitted_at TEXT,
              approved_by TEXT,
              approved_at TEXT,
              approval_comment TEXT
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS case_events (
              event_id TEXT PRIMARY KEY,
              case_id TEXT,
              event_type TEXT,
              message TEXT,
              created_at TEXT
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS revisjonslogg (
              tidspunkt TEXT,
              handling TEXT,
              antall INTEGER
            )
            """
        )
        c.commit()


def require_key(x_api_key: Optional[str]) -> None:
    if API_KEY and (x_api_key != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


def audit(action: str, count: int = 1) -> None:
    # best-effort audit
    try:
        with get_conn() as c:
            c.execute(
                "INSERT INTO revisjonslogg (tidspunkt, handling, antall) VALUES (?, ?, ?)",
                (utc_now_iso(), action, int(count)),
            )
            c.commit()
    except Exception:
        pass


def fetchall_dict(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]


class TxIn(BaseModel):
    trans_id: Optional[str] = None
    fra_konto: Optional[str] = None
    til_konto: Optional[str] = None
    belop: Optional[float] = None
    valuta: Optional[str] = None
    timestamp: Optional[str] = None
    kunde_id: Optional[str] = None
    mottaker_id: Optional[str] = None
    land: Optional[str] = None
    mistenkelig: Optional[int] = 0
    mistenkelig_ml: Optional[int] = 0
    sanksjonert: Optional[int] = 0
    fuzzy_sanksjonert: Optional[int] = 0
    risikoscore: Optional[float] = None


class TxIngestRequest(BaseModel):
    transactions: list[TxIn] = Field(default_factory=list)


class CaseCreate(BaseModel):
    trans_id: Optional[str] = None
    status: str = "OPEN"
    priority: str = "MEDIUM"
    owner: Optional[str] = None
    tags: Optional[str] = None
    note: Optional[str] = None
    entity_key: Optional[str] = None


class CaseUpdate(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    owner: Optional[str] = None
    tags: Optional[str] = None
    note: Optional[str] = None

    # maker-checker approval flow
    submitted_by: Optional[str] = None
    submitted_at: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    approval_comment: Optional[str] = None


class CaseEventCreate(BaseModel):
    event_type: str = "NOTE"
    message: str


app = FastAPI(title="AML Platform API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    ensure_tables()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "db_path": DB_PATH, "ts_utc": utc_now_iso()}


@app.get("/transactions")
def list_transactions(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    require_key(x_api_key)
    with get_conn() as c:
        rows = c.execute(
            "SELECT * FROM transaksjoner ORDER BY COALESCE(timestamp, '') DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return {"items": fetchall_dict(rows)}


@app.post("/transactions/ingest")
def ingest_transactions(
    payload: TxIngestRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    require_key(x_api_key)
    inserted = 0
    with get_conn() as c:
        for tx in payload.transactions:
            d = tx.model_dump()
            trans_id = d.get("trans_id") or f"TX-{uuid.uuid4().hex[:12]}"
            ts = d.get("timestamp") or utc_now_iso()

            c.execute(
                """
                INSERT OR REPLACE INTO transaksjoner
                (trans_id, fra_konto, til_konto, belop, valuta, timestamp, kunde_id, mottaker_id, land,
                 mistenkelig, mistenkelig_ml, sanksjonert, fuzzy_sanksjonert, risikoscore)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trans_id,
                    d.get("fra_konto"),
                    d.get("til_konto"),
                    d.get("belop"),
                    d.get("valuta"),
                    ts,
                    d.get("kunde_id"),
                    d.get("mottaker_id"),
                    d.get("land"),
                    int(d.get("mistenkelig") or 0),
                    int(d.get("mistenkelig_ml") or 0),
                    int(d.get("sanksjonert") or 0),
                    int(d.get("fuzzy_sanksjonert") or 0),
                    d.get("risikoscore"),
                ),
            )
            inserted += 1
        c.commit()

    audit("API_TX_INGEST", inserted)
    return {"ok": True, "inserted": inserted}


@app.get("/cases")
def list_cases(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    status: Optional[str] = None,
    owner: Optional[str] = None,
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    require_key(x_api_key)
    q = "SELECT * FROM cases WHERE 1=1"
    params: list[Any] = []
    if status:
        q += " AND status=?"
        params.append(status)
    if owner:
        q += " AND owner=?"
        params.append(owner)
    q += " ORDER BY COALESCE(updated_at, created_at, '') DESC LIMIT ?"
    params.append(int(limit))

    with get_conn() as c:
        rows = c.execute(q, params).fetchall()
    return {"items": fetchall_dict(rows)}


@app.post("/cases")
def create_case(
    payload: CaseCreate,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    require_key(x_api_key)
    case_id = f"CASE-{uuid.uuid4().hex[:10]}"
    now = utc_now_iso()
    with get_conn() as c:
        c.execute(
            """
            INSERT INTO cases
              (case_id, trans_id, status, priority, owner, tags, note, created_at, updated_at, entity_key,
               submitted_by, submitted_at, approved_by, approved_at, approval_comment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
            """,
            (
                case_id,
                payload.trans_id,
                payload.status,
                payload.priority,
                payload.owner,
                payload.tags,
                payload.note,
                now,
                now,
                payload.entity_key,
            ),
        )
        c.commit()

    audit("API_CASE_CREATE", 1)
    return {"ok": True, "case_id": case_id}


@app.get("/cases/{case_id}")
def get_case(
    case_id: str,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    require_key(x_api_key)
    with get_conn() as c:
        row = c.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Case not found")
        events = c.execute(
            "SELECT * FROM case_events WHERE case_id=? ORDER BY COALESCE(created_at,'') DESC LIMIT 200",
            (case_id,),
        ).fetchall()
    return {"case": dict(row), "events": fetchall_dict(events)}


@app.patch("/cases/{case_id}")
def update_case(
    case_id: str,
    payload: CaseUpdate,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    require_key(x_api_key)

    with get_conn() as c:
        existing = c.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Case not found")

        existing_d = dict(existing)
        submitted_by = payload.submitted_by or existing_d.get("submitted_by")

        if payload.approved_by and submitted_by and (payload.approved_by == submitted_by):
            raise HTTPException(
                status_code=400,
                detail="Maker-checker violated: same user cannot approve own submission",
            )

        fields = []
        params: list[Any] = []

        for k, v in payload.model_dump(exclude_unset=True).items():
            fields.append(f"{k}=?")
            params.append(v)

        fields.append("updated_at=?")
        params.append(utc_now_iso())

        params.append(case_id)

        if fields:
            c.execute(f"UPDATE cases SET {', '.join(fields)} WHERE case_id=?", params)
            c.commit()

    audit("API_CASE_UPDATE", 1)
    return {"ok": True}


@app.post("/cases/{case_id}/events")
def add_case_event(
    case_id: str,
    payload: CaseEventCreate,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    require_key(x_api_key)
    event_id = f"EV-{uuid.uuid4().hex[:10]}"
    with get_conn() as c:
        exists = c.execute("SELECT 1 FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="Case not found")

        c.execute(
            """
            INSERT INTO case_events (event_id, case_id, event_type, message, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_id, case_id, payload.event_type, payload.message, utc_now_iso()),
        )
        c.commit()

    audit("API_CASE_EVENT", 1)
    return {"ok": True, "event_id": event_id}


@app.get("/audit")
def list_audit(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    limit: int = Query(default=2000, ge=1, le=20000),
) -> dict:
    require_key(x_api_key)
    with get_conn() as c:
        rows = c.execute(
            "SELECT tidspunkt, handling, antall FROM revisjonslogg ORDER BY COALESCE(tidspunkt,'') DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return {"items": fetchall_dict(rows)}
