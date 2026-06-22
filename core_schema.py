import sqlite3
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def ensure_core_schema(conn: sqlite3.Connection):
    cur = conn.cursor()

    # USERS
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT,
        role TEXT,
        is_active INTEGER,
        created_at TEXT
    )
    """)

    # SESSIONS
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY,
        username TEXT,
        role TEXT,
        created_at TEXT,
        expires_at TEXT,
        revoked_at TEXT
    )
    """)

    # CASES (Maker–Checker fields)
    cur.execute("""
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
    """)

    # CASE EVENTS (human-readable)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS case_events (
        event_id TEXT PRIMARY KEY,
        case_id TEXT,
        event_type TEXT,
        message TEXT,
        created_at TEXT
    )
    """)

    # SIGNALS (Explainability)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        signal_id TEXT PRIMARY KEY,
        case_id TEXT,
        signal_type TEXT,
        description TEXT,
        score REAL,
        weight REAL DEFAULT 1.0,
        source TEXT,
        created_at TEXT
    )
    """)

    # AUDIT LOG (IMMUTABLE)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        audit_id TEXT PRIMARY KEY,
        entity_type TEXT,
        entity_id TEXT,
        action TEXT,
        actor TEXT,
        role TEXT,
        before_json TEXT,
        after_json TEXT,
        reason TEXT,
        created_at TEXT
    )
    """)

    # Indexer
    cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_case_events_case ON case_events(case_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_case ON signals(case_id)")

    conn.commit()
