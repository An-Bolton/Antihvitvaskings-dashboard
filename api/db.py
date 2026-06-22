import os
import sqlite3
from contextlib import contextmanager

# Prosjektrot = mappa som inneholder /api
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.getenv("AML_DB_PATH", os.path.join(BASE_DIR, "transaksjoner.db"))

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def ensure_case_tables():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            title TEXT,
            status TEXT DEFAULT 'open',
            tier TEXT DEFAULT 'T1',
            risk_score REAL DEFAULT 0.0,
            entity_id TEXT,
            created_at TEXT,
            updated_at TEXT,
            assigned_to TEXT,
            submitted_by TEXT,
            submitted_at TEXT,
            approved_by TEXT,
            approved_at TEXT,
            approval_comment TEXT
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS case_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT,
            event_type TEXT,
            message TEXT,
            actor TEXT,
            created_at TEXT
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS case_notes (
            note_id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT,
            note TEXT,
            actor TEXT,
            created_at TEXT
        )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_assigned_to ON cases(assigned_to)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_tier ON cases(tier)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_updated_at ON cases(updated_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_created_at ON cases(created_at)")

def ensure_tx_tables() -> None:
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            amount REAL NOT NULL,
            currency TEXT,
            timestamp TEXT NOT NULL,
            counterparty TEXT,
            country TEXT,
            description TEXT
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_customer ON transactions(customer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_timestamp ON transactions(timestamp)")


def ensure_signal_tables() -> None:
    """Tables used by ingest/routes for storing derived 'signals' from transactions.

    This is a lightweight event stream for flags such as sanction hits, anomaly detections,
    threshold triggers, model scores, etc.
    """
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trans_id TEXT,
                transaction_id TEXT,
                customer_id TEXT,
                signal_type TEXT NOT NULL,
                signal_score REAL,
                risk_level TEXT,
                details_json TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_trans_id ON signals(trans_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_transaction_id ON signals(transaction_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_customer_id ON signals(customer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type)")


def ensure_core_schema() -> None:
    """Convenience helper for API startup."""
    ensure_tx_tables()
    ensure_signal_tables()
    ensure_case_tables()
    ensure_alert_tables()
    ensure_v_all_transactions_view()

def ensure_alert_tables() -> None:
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            customer_id TEXT,
            risk_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            reasons_json TEXT,
            created_at TEXT,
            status TEXT DEFAULT 'open'
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_txid ON alerts(transaction_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_customer ON alerts(customer_id)")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?",
            (name,),
        ).fetchone()
        return bool(row)
    except Exception:
        return False


def ensure_v_all_transactions_view() -> None:
    """Create/refresh a unified read-only view over both legacy `transaksjoner` and API `transactions`.

    The dashboard should query ONLY `v_all_transactions` to avoid schema drift (trans_id vs transaction_id).
    """
    with get_conn() as conn:
        has_legacy = _table_exists(conn, "transaksjoner")
        has_api = _table_exists(conn, "transactions")

        # Always drop & recreate to keep it in sync with schema tweaks
        conn.execute("DROP VIEW IF EXISTS v_all_transactions")

        parts = []

        if has_legacy:
            parts.append(
                """
                SELECT
                    CAST(trans_id AS TEXT)            AS trans_id,
                    fra_konto                         AS fra_konto,
                    til_konto                         AS til_konto,
                    beløp                             AS beløp,
                    tidspunkt                         AS tidspunkt,
                    land                              AS land,
                    score                             AS score,
                    risikonivå                        AS risikonivå,
                    reasons_json                      AS reasons_json,
                    mistenkelig                       AS mistenkelig,
                    mistenkelig_ml                    AS mistenkelig_ml,
                    sanksjonert                       AS sanksjonert,
                    fuzzy_sanksjonert                 AS fuzzy_sanksjonert,
                    anomaly_score                     AS anomaly_score,
                    'transaksjoner'                   AS source_table
                FROM transaksjoner
                """.strip()
            )

        if has_api:
            # Some setups add a `trans_id` compatibility column to `transactions`.
            # We prefer it if present; otherwise fall back to transaction_id.
            parts.append(
                """
                SELECT
                    CAST(COALESCE(trans_id, transaction_id) AS TEXT) AS trans_id,
                    customer_id                        AS fra_konto,
                    COALESCE(counterparty,'')          AS til_konto,
                    amount                             AS beløp,
                    timestamp                          AS tidspunkt,
                    COALESCE(country,'')               AS land,
                    NULL                               AS score,
                    NULL                               AS risikonivå,
                    NULL                               AS reasons_json,
                    NULL                               AS mistenkelig,
                    NULL                               AS mistenkelig_ml,
                    NULL                               AS sanksjonert,
                    NULL                               AS fuzzy_sanksjonert,
                    NULL                               AS anomaly_score,
                    'transactions'                     AS source_table
                FROM transactions
                """.strip()
            )

        if not parts:
            # Create an empty view with the expected schema
            parts.append(
                """
                SELECT
                    CAST(NULL AS TEXT) AS trans_id,
                    CAST(NULL AS TEXT) AS fra_konto,
                    CAST(NULL AS TEXT) AS til_konto,
                    CAST(NULL AS REAL) AS beløp,
                    CAST(NULL AS TEXT) AS tidspunkt,
                    CAST(NULL AS TEXT) AS land,
                    CAST(NULL AS REAL) AS score,
                    CAST(NULL AS TEXT) AS risikonivå,
                    CAST(NULL AS TEXT) AS reasons_json,
                    CAST(NULL AS INTEGER) AS mistenkelig,
                    CAST(NULL AS INTEGER) AS mistenkelig_ml,
                    CAST(NULL AS INTEGER) AS sanksjonert,
                    CAST(NULL AS INTEGER) AS fuzzy_sanksjonert,
                    CAST(NULL AS REAL) AS anomaly_score,
                    CAST(NULL AS TEXT) AS source_table
                WHERE 1=0
                """.strip()
            )

        view_sql = "CREATE VIEW v_all_transactions AS\n" + "\nUNION ALL\n".join(parts)
        conn.execute(view_sql)


def ensure_core_schema() -> None:
    """Convenience: ensure all core tables + unified view exist."""
    ensure_case_tables()
    ensure_tx_tables()
    ensure_alert_tables()
    ensure_v_all_transactions_view()
