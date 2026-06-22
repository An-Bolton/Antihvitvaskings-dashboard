# api/routes_ingest.py
from __future__ import annotations

import os
import uuid
from typing import Optional, Any
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

import pandas as pd

from .db import (
    get_conn,
    ensure_tx_tables,
    ensure_alert_tables,
    ensure_signal_tables,
    DB_PATH,
)
from risk_engine import analyser_transaksjoner

# Optional: dual-write into dashboard table `transaksjoner`.
try:
    from db_handler import lagre_til_db as lagre_til_dashboard_db
except Exception:
    lagre_til_dashboard_db = None

API_KEY = os.getenv("AML_API_KEY", "dev-key")

router = APIRouter()


# --- Customer signals (MVP) ---
VELOCITY_TX_THRESHOLD = 5        # 5+ tx last 24h
COUNTERPARTY_THRESHOLD = 4       # 4+ unique counterparties last 24h
STRUCTURING_TX_THRESHOLD = 3     # 3+ near-threshold tx last 7d

# "Structuring zone" = 90%.100% of medium threshold (250k by default)
STRUCTURING_TARGET = 250_000
STRUCTURING_LOW = 0.90 * STRUCTURING_TARGET
STRUCTURING_HIGH = 1.00 * STRUCTURING_TARGET


def require_key(x_api_key: Optional[str]) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class TransactionIn(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    currency: str | None = "NOK"
    timestamp: str  # ISO string
    counterparty: str | None = None
    country: str | None = None
    description: str | None = None


def _create_case_for_alert(tx_id: str, risk_score: float, risk_level: str) -> str:
    """Create a case row for a high-risk transaction."""
    case_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    title = f"Auto-case: {risk_level} ({tx_id})"
    tier = "T3"

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO cases
            (case_id, title, status, tier, risk_score, entity_id, created_at, updated_at)
            VALUES (?, ?, 'open', ?, ?, ?, ?, ?)
            """,
            (case_id, title, tier, float(risk_score), tx_id, now, now),
        )
        conn.execute(
            """
            INSERT INTO case_events (case_id, event_type, message, actor, created_at)
            VALUES (?, 'created', ?, 'system', ?)
            """,
            (case_id, f"Auto-created from high-risk alert. risk_score={risk_score:.2f}", now),
        )
        conn.commit()

    return case_id


def update_customer_signals(customer_ids: list[str]) -> None:
    """Update aggregated customer signals in SQLite for fast dashboard queries."""
    ensure_signal_tables()
    now = datetime.now(timezone.utc).isoformat()

    # avoid empty / duplicates
    ids = sorted({str(x) for x in customer_ids if x is not None and str(x).strip()})
    if not ids:
        return

    with get_conn() as conn:
        for cid in ids:
            # 24h: velocity + unique counterparties + amount
            r24 = conn.execute(
                """
                SELECT
                  COUNT(*) AS tx_24h,
                  COUNT(DISTINCT COALESCE(counterparty,'')) AS uniq_cp_24h,
                  COALESCE(SUM(amount),0) AS amt_24h,
                  MAX(timestamp) AS last_ts
                FROM transactions
                WHERE customer_id = ?
                  AND datetime(timestamp) >= datetime('now','-24 hours')
                """,
                (cid,),
            ).fetchone()

            tx_24h = int((r24[0] if r24 else 0) or 0)
            uniq_cp_24h = int((r24[1] if r24 else 0) or 0)
            amt_24h = float((r24[2] if r24 else 0.0) or 0.0)
            last_ts = (r24[3] if r24 else None)

            # 7d: total + structuring near threshold
            r7 = conn.execute(
                """
                SELECT
                  COUNT(*) AS tx_7d,
                  SUM(CASE WHEN amount BETWEEN ? AND ? THEN 1 ELSE 0 END) AS struct_7d
                FROM transactions
                WHERE customer_id = ?
                  AND datetime(timestamp) >= datetime('now','-7 days')
                """,
                (STRUCTURING_LOW, STRUCTURING_HIGH, cid),
            ).fetchone()

            tx_7d = int((r7[0] if r7 else 0) or 0)
            struct_7d = int((r7[1] if r7 else 0) or 0)

            # dormant->active (MVP): no tx in 30d window before last_ts
            dormant_30d = 0
            if last_ts:
                prev = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM transactions
                    WHERE customer_id = ?
                      AND datetime(timestamp) < datetime(?)
                      AND datetime(timestamp) >= datetime(?,'-30 days')
                    """,
                    (cid, last_ts, last_ts),
                ).fetchone()
                dormant_30d = 1 if int((prev[0] if prev else 0) or 0) == 0 else 0

            conn.execute(
                """
                INSERT INTO customer_signals
                (customer_id, tx_24h, uniq_counterparties_24h, amount_24h,
                 tx_7d, structuring_7d, last_tx_ts, dormant_30d, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                  tx_24h=excluded.tx_24h,
                  uniq_counterparties_24h=excluded.uniq_counterparties_24h,
                  amount_24h=excluded.amount_24h,
                  tx_7d=excluded.tx_7d,
                  structuring_7d=excluded.structuring_7d,
                  last_tx_ts=excluded.last_tx_ts,
                  dormant_30d=excluded.dormant_30d,
                  updated_at=excluded.updated_at
                """,
                (cid, tx_24h, uniq_cp_24h, amt_24h, tx_7d, struct_7d, last_ts, dormant_30d, now),
            )


@router.post("/transactions")
def ingest_transactions(
    payload: list[TransactionIn],
    x_api_key: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    require_key(x_api_key)
    ensure_tx_tables()
    ensure_alert_tables()

    if not payload:
        raise HTTPException(status_code=400, detail="Empty payload")

    # 1) Save raw transactions to API schema
    with get_conn() as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO transactions
            (transaction_id, customer_id, amount, currency, timestamp, counterparty, country, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    t.transaction_id,
                    t.customer_id,
                    float(t.amount),
                    t.currency or "NOK",
                    t.timestamp,
                    t.counterparty,
                    (t.country or "").upper() if t.country else None,
                    t.description,
                )
                for t in payload
            ],
        )
        conn.commit()

    # 2) Map to risk_engine expected columns (Norwegian legacy)
    df = pd.DataFrame(
        [
            {
                "trans_id": t.transaction_id,
                "fra_konto": t.customer_id,
                "til_konto": t.counterparty or "",
                "beløp": float(t.amount),
                "land": (t.country or "").upper(),
                "tidspunkt": t.timestamp,
            }
            for t in payload
        ]
    )

    scored = analyser_transaksjoner(df, db_path=DB_PATH)

    # 2b) Dual-write to dashboard schema if available
    if lagre_til_dashboard_db is not None:
        try:
            lagre_til_dashboard_db(scored)
        except Exception:
            # Don't fail ingestion if dashboard table write fails.
            pass

    now = datetime.now(timezone.utc).isoformat()
    alerts_to_insert: list[tuple] = []
    auto_cases_created = 0

    # 3) Build alerts + auto-cases
    for _, row in scored.iterrows():
        score = float(row.get("score", 0.0))
        risk_level = str(row.get("risikonivå", ""))
        reasons_json = str(row.get("reasons_json", ""))

        if score >= 0.50:
            alerts_to_insert.append(
                (
                    str(row.get("trans_id")),
                    str(row.get("fra_konto")),
                    score,
                    risk_level,
                    reasons_json,
                    now,
                    "open",
                )
            )

        if score >= 0.80:
            _create_case_for_alert(str(row.get("trans_id")), score, risk_level)
            auto_cases_created += 1

    if alerts_to_insert:
        with get_conn() as conn:
            conn.executemany(
                """
                INSERT INTO alerts
                (transaction_id, customer_id, risk_score, risk_level, reasons_json, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                alerts_to_insert,
            )
            conn.commit()

    # 4) Mark high-risk alerts as in_case
    if auto_cases_created:
        tx_ids = [str(r[0]) for r in alerts_to_insert if float(r[2]) >= 0.80]
        if tx_ids:
            q_marks = ",".join(["?"] * len(tx_ids))
            with get_conn() as conn:
                conn.execute(
                    f"UPDATE alerts SET status='in_case' WHERE transaction_id IN ({q_marks})",
                    tx_ids,
                )
                conn.commit()

    # 5) Update aggregated customer signals for fast dashboard views
    try:
        update_customer_signals([t.customer_id for t in payload])
    except Exception:
        # Never fail ingestion if signals aggregation fails.
        pass

    return {
        "ok": True,
        "ingested": len(payload),
        "alerts_created": len(alerts_to_insert),
        "auto_cases_created": auto_cases_created,
        "dashboard_written": bool(lagre_til_dashboard_db is not None),
    }
