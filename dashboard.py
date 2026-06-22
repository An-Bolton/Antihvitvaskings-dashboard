import difflib
import json
import os
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, date, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Auth (extreme persistence)
from auth_db import login_gate, sidebar_auth_box, has_role

st.set_page_config(page_title='Anti-hvitvaskingsdemo - av Andreas Bolton Seielstad', layout='wide', initial_sidebar_state='expanded')

# Render auth UI first (so login form is visible), then gate the rest.
sidebar_auth_box(prefix='auth')
login_gate('Login')

# ---- Sync legacy dashboard identity (Min kø / godkjenning) with DB-login ----
_ROLE_MAP = {'analyst': 'T1', 'senior': 'T2', 'compliance': 'CHECKER', 'admin': 'T2'}
try:
    _auth = st.session_state.get('auth', {})
    _username = _auth.get('username')
    _role = _auth.get('role')
    if _username:
        st.session_state['analyst_name'] = _username
    if _role:
        st.session_state['user_role'] = _ROLE_MAP.get(_role, 'T1')
except Exception:
    pass


# External modules from your project
from db_handler import hent_transaksjoner, lagre_til_db
from ml_module import legg_til_anomalikluster
from pep_checker import hent_sanksjonsdata, sjekk_mot_sanksjonsliste
from risk_engine import analyser_transaksjoner
from slack_notifier import send_slack_varsel  # transaction alerts
from utils import trygg_les_csv

# -------------------------------------------------------------------
# API integration (optional)
# -------------------------------------------------------------------
# If you run the FastAPI service under ./api (recommended), you can let the
# dashboard read/write via HTTP instead of talking to SQLite directly.
#
# Start API (your setup likely uses api/app.py):
#   export AML_DB_PATH="transaksjoner.db"   # or hvitvask.db
#   export AML_API_KEY="dev-key"
#   uvicorn api.app:app --reload --port 8052
#
# In dashboard: enable "API-mode" in the sidebar.
import traceback

_API_CLIENT_IMPORT_ERROR = ""

try:
    # Prefer local client in project root; fall back to api/api_client.py if kept there
    try:
        from api_client import AMLApi  # type: ignore
    except Exception:
        from api_client import AMLApi  # type: ignore

    _HAS_API_CLIENT = True
except Exception as e:
    AMLApi = None  # type: ignore
    _HAS_API_CLIENT = False
    _API_CLIENT_IMPORT_ERROR = f"{e}\n\n{traceback.format_exc()}"


def _init_api_settings():
    """Initialize API-related session state defaults (safe to call multiple times)."""
    if "use_api" not in st.session_state:
        st.session_state.use_api = False
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = os.getenv("AML_API_BASE_URL", "http://127.0.0.1:8052")
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("AML_API_KEY", "dev-key")

def get_api():
    """Get (and cache) API client."""
    _init_api_settings()
    if not st.session_state.use_api or not _HAS_API_CLIENT or AMLApi is None:
        return None

    cache_key = "_aml_api_client"
    client = st.session_state.get(cache_key)

    # recreate client if base_url changed
    if client is None or getattr(client, "base_url", None) != st.session_state.api_base_url:
        st.session_state[cache_key] = AMLApi(
            st.session_state.api_base_url,
            api_key=st.session_state.api_key,
        )
    return st.session_state[cache_key]


def sidebar_api_controls():
    """Render API controls in sidebar (call once early)."""
    _init_api_settings()
    with st.sidebar.expander("🔌 API-tilkobling", expanded=False):
        if not _HAS_API_CLIENT:
            st.warning("API-klient kunne ikke importeres.")
            if _API_CLIENT_IMPORT_ERROR:
                st.code(_API_CLIENT_IMPORT_ERROR)
            st.session_state.use_api = False
            return

        st.session_state.use_api = st.toggle(
            "Bruk API (i stedet for direkte DB)",
            value=bool(st.session_state.use_api),
        )

        st.session_state.api_base_url = st.text_input(
            "API base URL",
            value=st.session_state.api_base_url,
            help="Tips (Mac): bruk gjerne http://127.0.0.1:PORT i stedet for localhost.",
        )

        st.session_state.api_key = st.text_input(
            "API key",
            value=st.session_state.api_key,
            type="password",
        )

        api = get_api()
        if not api:
            return

        try:
            h = api.health()
            st.success("API OK")
        except Exception as e1:
            # Mac/IPv6-fiks: hvis du har localhost i URL, prøv automatisk 127.0.0.1
            url = (st.session_state.api_base_url or "").strip()
            if "localhost" in url:
                st.session_state.api_base_url = url.replace("localhost", "127.0.0.1")
                st.info("Bytter fra localhost til 127.0.0.1 (vanlig IPv6-fiks på Mac)…")
                st.rerun()

            st.error(f"API utilgjengelig: {e1}")


def hent_transaksjoner_via_api(limit: int = 5000) -> pd.DataFrame:
    """Hent transaksjoner via API hvis aktivert, ellers None."""
    api = get_api()
    if not api:
        return None  # type: ignore
    data = api.list_transactions(limit=limit).get("items", [])
    return pd.DataFrame(data)


def hent_transaksjoner_wrapper(limit: int = 5000) -> pd.DataFrame:
    """
    Wrapper: Returner transaksjoner enten fra API eller direkte fra DB.
    Keeps the rest of the dashboard unchanged.
    """
    df_api = hent_transaksjoner_via_api(limit=limit)
    if df_api is not None:
        return df_api

    # fallback to local DB handler
    try:
        return hent_transaksjoner()  # type: ignore
    except TypeError:
        return hent_transaksjoner(limit=limit)  # type: ignore


def lagre_transaksjoner_wrapper(df: pd.DataFrame) -> None:
    """
    Wrapper: Lagre transaksjoner via API ingest (idempotent) eller direkte til DB.
    """
    api = get_api()
    if api:
        if df is None or df.empty:
            return
        records = df.to_dict(orient="records")
        api.ingest_transactions(records)
        return

    # fallback
    lagre_til_db(df)  # type: ignore


# Optional PDF deps
def _std_risk(val) -> str:
    """Standardiser risikonivå til Lav/Medium/Høy (tåler emoji/tekst)."""
    if val is None:
        return ""
    s = str(val)
    if "Høy" in s or "High" in s:
        return "Høy"
    if "Medium" in s or "Med" in s:
        return "Medium"
    if "Lav" in s or "Low" in s:
        return "Lav"
    return ""


def _parse_iso_dt(x):
    """Parse datoer robust og normaliser til UTC (tz-aware) for å unngå tz-naive/tz-aware-feil."""
    try:
        return pd.to_datetime(x, errors="coerce", utc=True)
    except Exception:
        return pd.NaT


# SLA targets (timer) – juster som du vil
_SLA_HOURS_BY_PRIORITY = {
    "high": 24,
    "medium": 72,
    "low": 168,  # 7 dager
}

def _case_sla_enrich(df_cases: pd.DataFrame) -> pd.DataFrame:
    """Legg til alder/SLA-kolonner for case-lister."""
    if df_cases is None or df_cases.empty:
        return df_cases if df_cases is not None else pd.DataFrame()
    df = df_cases.copy()

    # created_at kan mangle/ha rare verdier -> fallback til updated_at
    created = _parse_iso_dt(df.get("created_at"))
    updated = _parse_iso_dt(df.get("updated_at"))
    base = created
    if isinstance(base, pd.Series):
        base = base.fillna(updated)
    else:
        base = updated

    now = pd.Timestamp.now(tz="UTC")
    age_hours = (now - base).dt.total_seconds() / 3600.0
    df["age_hours"] = age_hours.round(1)
    df["age_days"] = (age_hours / 24.0).round(2)

    if "priority" in df.columns:
        prio = df["priority"].astype(str).str.lower()
    else:
        prio = pd.Series(["medium"] * len(df), index=df.index)
    target_hours = prio.map(_SLA_HOURS_BY_PRIORITY).fillna(72).astype(float)
    df["sla_target_hours"] = target_hours

    df["sla_due_at"] = (base + pd.to_timedelta(target_hours, unit="h")).dt.strftime("%Y-%m-%d %H:%M:%S")

    # SLA status
    due = _parse_iso_dt(df["sla_due_at"])
    remaining_hours = (due - now).dt.total_seconds() / 3600.0
    df["sla_remaining_hours"] = remaining_hours.round(1)

    def _sla_label(rem_h):
        try:
            rem = float(rem_h)
        except Exception:
            return "🟡 Ukjent"
        if rem < 0:
            return "🔴 Brutt"
        if rem <= 12:
            return "🟠 Haster"
        if rem <= 48:
            return "🟡 Snart"
        return "🟢 OK"

    df["sla_status"] = df["sla_remaining_hours"].apply(_sla_label)

    # sort-key: brutt først, så haster, så snart, så ok
    order = {"🔴 Brutt": 0, "🟠 Hastrer": 1, "🟡 Snart": 2, "🟢 OK": 3, "🟡 Ukjent": 4}
    df["_sla_rank"] = df["sla_status"].map(order).fillna(99).astype(int)
    return df


try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm

    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


def find_matching_case_for_tx(conn, entity_key: str, tx_row: tuple, window_hours: int = 48,
                              require_same_to: bool = False, amount_tolerance: float = 0.15):
    """Smartere auto-grouping av transaksjoner inn i eksisterende sak (case).

    tx_row forventes som: (trans_id, fra_konto, til_konto, beløp, land, tidspunkt)
    """
    if not entity_key or not tx_row:
        return None

    to_acc = tx_row[2] if len(tx_row) > 2 else None
    amount = tx_row[3] if len(tx_row) > 3 else None
    ts_iso = tx_row[5] if len(tx_row) > 5 else None

    try:
        curr_ts = pd.to_datetime(ts_iso, errors="coerce")
        if pd.isna(curr_ts):
            curr_ts = None
        else:
            curr_ts = curr_ts.tz_localize(None) if getattr(curr_ts, "tzinfo", None) else curr_ts
    except Exception:
        curr_ts = None

    try:
        cand = conn.execute(
            "SELECT case_id FROM cases "
            "WHERE entity_key=? AND status IN ('open','in_review','escalated','sar_draft') "
            "ORDER BY datetime(updated_at) DESC LIMIT 25",
            (str(entity_key),)
        ).fetchall()
    except Exception:
        cand = []

    if not cand:
        return None

    def _case_last_ts(case_id: str):
        try:
            r = conn.execute(
                "SELECT MAX(datetime(t.tidspunkt)) "
                "FROM case_tx_links l JOIN transaksjoner t ON t.trans_id=l.trans_id "
                "WHERE l.case_id=?",
                (case_id,)
            ).fetchone()
            if r and r[0]:
                return pd.to_datetime(r[0], errors="coerce")
        except Exception:
            pass
        try:
            r = conn.execute(
                "SELECT datetime(t.tidspunkt) FROM cases c JOIN transaksjoner t ON t.trans_id=c.trans_id "
                "WHERE c.case_id=? LIMIT 1",
                (case_id,)
            ).fetchone()
            if r and r[0]:
                return pd.to_datetime(r[0], errors="coerce")
        except Exception:
            pass
        return None

    def _has_same_to_within(case_id: str):
        if not to_acc or not ts_iso:
            return False
        try:
            r = conn.execute(
                "SELECT 1 FROM case_tx_links l JOIN transaksjoner t ON t.trans_id=l.trans_id "
                "WHERE l.case_id=? AND t.til_konto=? "
                "AND datetime(t.tidspunkt) >= datetime(?, '-' || ? || ' hours') "
                "LIMIT 1",
                (case_id, str(to_acc), str(ts_iso), int(window_hours))
            ).fetchone()
            return bool(r)
        except Exception:
            return False

    def _amount_ok(case_id: str):
        if amount is None or not ts_iso:
            return True
        try:
            amt = float(amount)
        except Exception:
            return True
        if amt <= 0:
            return True
        try:
            r = conn.execute(
                "SELECT AVG(CAST(t.beløp AS REAL)) "
                "FROM case_tx_links l JOIN transaksjoner t ON t.trans_id=l.trans_id "
                "WHERE l.case_id=? "
                "AND datetime(t.tidspunkt) >= datetime(?, '-' || ? || ' hours')",
                (case_id, str(ts_iso), int(window_hours))
            ).fetchone()
            avg = float(r[0]) if r and r[0] is not None else None
            if not avg or avg <= 0:
                return True
            rel = abs(amt - avg) / avg
            return rel <= float(amount_tolerance)
        except Exception:
            return True

    for (cid,) in cand:
        last_ts = _case_last_ts(cid)
        if curr_ts is not None and last_ts is not None:
            try:
                last_ts = last_ts.tz_localize(None) if getattr(last_ts, "tzinfo", None) else last_ts
            except Exception:
                pass
            delta_h = (curr_ts - last_ts).total_seconds() / 3600.0
            if delta_h > float(window_hours) or delta_h < -float(window_hours):
                continue

        if require_same_to and not _has_same_to_within(cid):
            continue

        if not _amount_ok(cid):
            continue

        return cid

    return None


# =============================
#   INIT / DB UTILITIES
# =============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("AML_DB_PATH", os.path.join(BASE_DIR, "transaksjoner.db"))
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data(ttl=60)
def get_all_customer_ids() -> list[str]:
    """Fetch distinct customer IDs for dropdowns (best-effort across tables)."""
    ids: set[str] = set()
    try:
        with get_conn() as c:
            # Prefer explicit customers table if present
            for q in [
                "SELECT DISTINCT customer_id AS cid FROM customers WHERE customer_id IS NOT NULL AND customer_id != ''",
                "SELECT DISTINCT customer_id AS cid FROM transactions WHERE customer_id IS NOT NULL AND customer_id != ''",
                # legacy norwegian table might store the customer in fra_konto
                "SELECT DISTINCT fra_konto AS cid FROM transaksjoner WHERE fra_konto IS NOT NULL AND fra_konto != ''",
            ]:
                try:
                    df = pd.read_sql_query(q, c)
                    if not df.empty and "cid" in df.columns:
                        ids.update(df["cid"].astype(str).tolist())
                except Exception:
                    continue
    except Exception:
        pass
    return sorted(ids)


@st.cache_data(show_spinner=False)
def fetch_alerts(status: str = "open") -> pd.DataFrame:
    """Hent alerts fra SQLite. Returnerer tom DF hvis tabell mangler."""
    try:
        with get_conn() as conn:
            return pd.read_sql_query(
                """
                SELECT alert_id, transaction_id, customer_id, risk_score, risk_level, created_at, status, reasons_json
                FROM alerts
                WHERE status = ?
                ORDER BY alert_id DESC
                """,
                conn,
                params=(status,),
            )
    except Exception:
        return pd.DataFrame()

def update_alert_status(alert_id: int, new_status: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE alerts SET status=? WHERE alert_id=?", (new_status, int(alert_id)))
        conn.commit()


def _date_to_iso_start(d):
    # d is datetime.date
    return pd.Timestamp(d).strftime("%Y-%m-%d 00:00:00")


def _date_to_iso_end_exclusive(d):
    # exclusive end = next day 00:00:00
    return (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")


@st.cache_data(show_spinner=False)
def fetch_transactions_filtered(
        konto_query: str,
        konto_valgt: list[str],
        land_filter: list[str],
        risk_levels: list[str],
        dato_start,
        dato_slutt,
        limit: int = 501,
        offset: int = 0,
):
    """Hent transaksjoner fra DB med server-side filtering der det er mulig.
    Returnerer inntil `limit` rader (bruk limit=page_size+1 for å teste 'has_next').
    """
    where = []
    params: list = []

    # konto_query: substring match
    if konto_query:
        where.append("LOWER(fra_konto) LIKE ?")
        params.append(f"%{konto_query.strip().lower()}%")

    # konto_valgt: exact match list
    if konto_valgt:
        placeholders = ",".join(["?"] * len(konto_valgt))
        where.append(f"fra_konto IN ({placeholders})")
        params.extend([str(x) for x in konto_valgt])

    # land
    if land_filter:
        placeholders = ",".join(["?"] * len(land_filter))
        where.append(f"land IN ({placeholders})")
        params.extend([str(x) for x in land_filter])

    # dato-range (eksklusiv slutt)
    if dato_start and dato_slutt:
        where.append("tidspunkt >= ? AND tidspunkt < ?")
        params.append(_date_to_iso_start(dato_start))
        params.append(_date_to_iso_end_exclusive(dato_slutt))

    # risiko via score dersom score finnes i tabellen (best-effort)
    # Lav: <=0.5, Medium: (0.5, 0.8], Høy: >0.8
    if risk_levels and set(risk_levels) != {"Lav", "Medium", "Høy"}:
        clauses = []
        if "Lav" in risk_levels:
            clauses.append("(score <= 0.5)")
        if "Medium" in risk_levels:
            clauses.append("(score > 0.5 AND score <= 0.8)")
        if "Høy" in risk_levels:
            clauses.append("(score > 0.8)")
        if clauses:
            where.append("(" + " OR ".join(clauses) + ")")

    sql = "SELECT * FROM transaksjoner"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY datetime(tidspunkt) DESC, trans_id DESC LIMIT ? OFFSET ?"
    params.extend([int(limit), int(offset)])

    with get_conn() as conn:
        try:
            df = pd.read_sql_query(sql, conn, params=params)
        except Exception:
            # fallback: hvis score ikke finnes i DB, prøv uten risiko-delen
            sql2 = "SELECT * FROM transaksjoner"
            where2 = [w for w in where if "score" not in w]
            params2 = []
            # rebuild params without risk (simple approach: just drop risk clause params - we had none)
            # We only had params for konto/land/dato, so safe.
            params2 = params[:-2]  # drop limit/offset, will re-add
            if where2:
                sql2 += " WHERE " + " AND ".join(where2)
            sql2 += " ORDER BY datetime(tidspunkt) DESC, trans_id DESC LIMIT ? OFFSET ?"
            params2.extend([int(limit), int(offset)])
            df = pd.read_sql_query(sql2, conn, params=params2)
    return df


def ensure_tables():
    """Create tables used by the app if they don't exist, and seed checklist.
    Includes best-effort lightweight migrations for newer columns.
    """
    conn = get_conn()
    cur = conn.cursor()

    # --- Core transactions table (SQL-first expects it) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transaksjoner (
            trans_id TEXT PRIMARY KEY,
            fra_konto TEXT,
            til_konto TEXT,
            beløp REAL,
            land TEXT,
            tidspunkt TEXT,
            score REAL,
            risikonivå TEXT,
            mistenkelig INTEGER DEFAULT 0,
            mistenkelig_ml INTEGER DEFAULT 0,
            sanksjonert INTEGER DEFAULT 0,
            fuzzy_sanksjonert INTEGER DEFAULT 0,
            anomaly_cluster INTEGER
        )
    """)

    # --- Vurderinger (manuell behandling av transaksjoner) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vurderinger (
            trans_id TEXT PRIMARY KEY,
            kommentar TEXT,
            avklart INTEGER
        )
    """)

    # --- Enkle "kundetiltak" ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kundetiltak (
            kunde_id TEXT,
            risikonivå TEXT,
            tiltakstype TEXT,
            kommentar TEXT,
            dato TEXT
        )
    """)

    # --- Revisjonslogg ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS revisjonslogg (
            tidspunkt TEXT,
            handling TEXT,
            antall INTEGER
        )
    """)

    # --- KYC/EDD: kunder ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT,             -- PERSON | BUSINESS
            risk_band TEXT,        -- LOW | MEDIUM | HIGH | CRITICAL
            risk_score REAL,
            kyc_status TEXT,       -- clear | in_review | on_hold
            last_review_at TEXT,
            next_review_at TEXT,
            pep_flag INTEGER DEFAULT 0,
            edd_required INTEGER DEFAULT 0
        )
    """)

    # --- KYC/EDD: reviews ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kyc_reviews (
            review_id TEXT PRIMARY KEY,
            customer_id TEXT,
            review_type TEXT,      -- onboarding | periodic | event
            due_at TEXT,
            started_at TEXT,
            completed_at TEXT,
            outcome TEXT,          -- pass | fail | escalated
            reviewer TEXT,
            findings_json TEXT
        )
    """)

    # --- KYC/EDD: tasks (checklists) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kyc_tasks (
            task_id TEXT PRIMARY KEY,
            review_id TEXT,
            customer_id TEXT,
            task_type TEXT,
            title TEXT,
            status TEXT,           -- open | done
            assigned_to TEXT,
            created_at TEXT,
            completed_at TEXT
        )
    """)

    # --- KYC/EDD: dokumenter ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kyc_documents (
            doc_id TEXT PRIMARY KEY,
            customer_id TEXT,
            doc_type TEXT,         -- id | proof_of_address | ubo | sof
            filename TEXT,
            received_at TEXT,
            expiry_date TEXT,
            status TEXT            -- pending | verified | rejected | expired
        )
    """)

    # --- KYC/EDD: UBO-register ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kyc_ubos (
            ubo_id TEXT PRIMARY KEY,
            customer_id TEXT,
            name TEXT,
            dob TEXT,
            country TEXT,
            role TEXT,
            created_at TEXT
        )
    """)

    # --- Bank-klar sjekkliste ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS checklist (
            item_id TEXT PRIMARY KEY,
            domain TEXT,          -- Functional | Quality | Security | Operations | Compliance
            item TEXT,
            owner TEXT,
            status TEXT,          -- open | in_progress | done
            notes TEXT,
            updated_at TEXT
        )
    """)

    # --- Case management ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            trans_id TEXT,
            status TEXT,           -- open | in_review | escalated | pending_approval | approved | rejected | sar_draft | reported | closed
            priority TEXT,         -- low | medium | high
            owner TEXT,
            tags TEXT,
            note TEXT,
            entity_key TEXT,
            created_at TEXT,
            updated_at TEXT,
            submitted_by TEXT,
            submitted_at TEXT,
            approved_by TEXT,
            approved_at TEXT,
            approval_comment TEXT
        )
    """)

    # --- Case ↔ transaksjonslinking (1 case -> many tx) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS case_tx_links (
            case_id TEXT,
            trans_id TEXT,
            linked_at TEXT,
            PRIMARY KEY (case_id, trans_id)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_case_tx_trans ON case_tx_links(trans_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_case_tx_case ON case_tx_links(case_id)")

    # --- Case events (timeline/audit) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS case_events (
            event_id TEXT PRIMARY KEY,
            case_id TEXT,
            event_type TEXT,       -- create | update | note | status | decision | sar | approval
            message TEXT,
            actor TEXT,
            meta_json TEXT,
            created_at TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_case_events_case ON case_events(case_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_case_events_created ON case_events(created_at)")

    # --- Case notes (structured notes) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS case_notes (
            note_id TEXT PRIMARY KEY,
            case_id TEXT,
            note TEXT,
            author TEXT,
            created_at TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_case_notes_case ON case_notes(case_id)")

    # --- Best-effort migrations for older DBs (ignore failures) ---
    # Add missing columns on 'cases' if the DB existed before.
    try:
        cols = {r[1] for r in cur.execute("PRAGMA table_info(cases)").fetchall()}
        want = {
            "entity_key": "TEXT",
            "submitted_by": "TEXT",
            "submitted_at": "TEXT",
            "approved_by": "TEXT",
            "approved_at": "TEXT",
            "approval_comment": "TEXT",
            "tier": "TEXT",  # T1 | T2 | CHECKER
        }
        for col, typ in want.items():
            if col not in cols:
                cur.execute(f"ALTER TABLE cases ADD COLUMN {col} {typ}")
    except Exception:
        pass

    conn.commit()

    # --- Seed checklist if empty ---
    try:
        row = cur.execute("SELECT COUNT(*) FROM checklist").fetchone()
        if row and int(row[0]) == 0:
            seed_items = [
                ("Functional", "Regelbasert overvåking – prod-regler definert"),
                ("Functional", "ML-modell – treningsdata, validering, driftstrategi"),
                ("Functional", "Sanksjons/PEP-screening – full kjede m/ logging"),
                ("Functional", "KYC/KYB/EDD-checklist – maker-checker"),
                ("Quality", "Testdekning >70% kritisk logikk"),
                ("Quality", "CI: lint, typecheck, enhetstester"),
                ("Quality", "Datakvalitetsvarsler (skjevheter, mangler)"),
                ("Security", "RBAC/least-privilege implementert"),
                ("Security", "Secrets i vault (ikke i kode/env)"),
                ("Security", "Kryptering i ro/overført – verifisert"),
                ("Security", "Sikker SDLC + pentest rapport"),
                ("Operations", "Observability: metrics, logs, alerter"),
                ("Operations", "Backup/restore og DR-test"),
                ("Operations", "Skalerings- og failover-strategi"),
                ("Compliance", "ROS/DPIA med tiltaksliste"),
                ("Compliance", "Policy/prosedyrer + opplæring"),
                ("Compliance", "Internrevisjon/ekstern test planlagt"),
            ]
            now = datetime.utcnow().isoformat()
            for idx, (dom, txt) in enumerate(seed_items, start=1):
                cur.execute(
                    "INSERT OR IGNORE INTO checklist (item_id, domain, item, owner, status, notes, updated_at) VALUES (?,?,?,?,?,?,?)",
                    (f"CHK{idx:03d}", dom, txt, "", "open", "", now),
                )
            conn.commit()
    except Exception:
        pass

    conn.close()


def logg_hendelse(hva, antall=0):
    """Append a row to revisjonslogg (best-effort)."""
    try:
        conn = get_conn()
        conn.execute("INSERT INTO revisjonslogg VALUES (?, ?, ?)", (datetime.now().isoformat(), hva, int(antall)))
        conn.commit()
        conn.close()
    except Exception:
        pass


# =============================
#   CASE MANAGEMENT HELPERS
# =============================

def _new_id(prefix: str) -> str:
    return f"{prefix}{int(datetime.utcnow().timestamp() * 1000)}{np.random.randint(1000):04d}"


def ensure_case_for_trans(trans_id: str, default_owner: str = "", default_priority: str = "medium") -> str:
    """Opprett sak hvis den ikke finnes for transaksjonen. Returnerer case_id."""
    case_id = f"CASE_{trans_id}"
    now = datetime.utcnow().isoformat()
    with get_conn() as c:
        c.execute(
            """
            INSERT OR IGNORE INTO cases(case_id, trans_id, status, priority, owner, tags, note, created_at, updated_at)
            VALUES (?, ?, 'open', ?, ?, '', '', ?, ?)
            """,
            (case_id, trans_id, default_priority, default_owner, now, now),
        )
        # Logg create-event hvis den ble opprettet (best effort)
        try:
            ev_id = _new_id("EV")
            c.execute(
                """INSERT OR IGNORE INTO case_events(event_id, case_id, event_type, message, created_at)
                     VALUES (?, ?, 'create', ?, ?)""",
                (ev_id, case_id, f"Case opprettet for transaksjon {trans_id}", now),
            )
        except Exception:
            pass
    return case_id


def load_cases() -> pd.DataFrame:
    with get_conn() as c:
        return pd.read_sql_query(
            """SELECT case_id, trans_id, status, priority, owner, tags, note, created_at, updated_at
                 FROM cases ORDER BY datetime(updated_at) DESC""",
            c,
        )


def load_case_events(case_id: str) -> pd.DataFrame:
    with get_conn() as c:
        return pd.read_sql_query(
            """SELECT event_type, message, created_at
                 FROM case_events WHERE case_id=? ORDER BY datetime(created_at) DESC""",
            c, params=(case_id,)
        )


def log_case_event(
    case_id: str,
    event_type: str,
    message: str,
    actor: str = "system",
    meta: dict | None = None,
    **_,
) -> None:
    """Best effort logg til case_events (timeline/audit).

    - actor: hvem som utførte handlingen (T1/T2/checker/system)
    - meta: valgfri dict som lagres som JSON i meta_json
    """
    try:
        now = datetime.utcnow().isoformat()
        ev_id = _new_id("EV")
        meta_json = json.dumps(meta, ensure_ascii=False) if meta else None
        with get_conn() as c:
            c.execute(
                """INSERT INTO case_events(event_id, case_id, event_type, message, actor, meta_json, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (ev_id, case_id, str(event_type), str(message)[:2000], str(actor)[:200], meta_json, now),
            )
    except Exception:
        pass


def render_case_timeline(ev: pd.DataFrame, key_prefix: str = "timeline"):
    """Timeline (nyeste først) basert på case_events, med filtre og søk.

    ev forventes å ha kolonnene: event_type, message, created_at
    """
    if ev is None or getattr(ev, "empty", True):
        st.info("Ingen hendelser logget.")
        return

    # --- UI controls (safe unique keys) ---
    types_all = sorted({str(x) for x in ev.get("event_type", pd.Series([], dtype=str)).dropna().unique().tolist()})
    if not types_all:
        types_all = ["system"]

    c1, c2, c3 = st.columns([1.2, 1.6, 0.8])
    with c1:
        sel_types = st.multiselect(
            "Hendelsestyper",
            options=types_all,
            default=types_all,
            key=f"{key_prefix}_types",
        )
    with c2:
        q = st.text_input("Søk i hendelser", value="", key=f"{key_prefix}_q")
    with c3:
        limit = st.number_input("Maks", min_value=10, max_value=500, value=120, step=10, key=f"{key_prefix}_limit")

    view = ev.copy()
    if sel_types:
        view = view[view["event_type"].astype(str).isin([str(x) for x in sel_types])]
    if q and "message" in view.columns:
        ql = q.strip().lower()
        view = view[view["message"].astype(str).str.lower().str.contains(ql, na=False)]

    view = view.head(int(limit))

    icon_map = {
        "status": "🔁",
        "note": "📝",
        "system": "⚙️",
        "sar": "🧾",
        "decision": "✅",
        "escalation": "🚨",
        "update": "✏️",
        "approve": "🧷",
        "reject": "⛔",
    }

    # Pretty rendering
    for _, r in view.iterrows():
        et = str(r.get("event_type", "system"))
        msg = str(r.get("message", ""))
        ts = str(r.get("created_at", ""))

        icon = icon_map.get(et, "•")

        # Highlight "diff"-style messages if present
        if "→" in msg and et in {"status", "update"}:
            st.markdown(f"{icon} **{et}** — `{msg}`  ")
        else:
            st.markdown(f"{icon} **{et}** — {msg}  ")
        if ts:
            st.caption(ts)

    with st.expander("🔎 Rå hendelseslogg (tabell)"):
        st.dataframe(view, use_container_width=True)


@st.cache_data(show_spinner=False)
def count_cases_filtered(
        search: str,
        status_list: list[str],
        priority_list: list[str],
        owner_contains: str,
        mine_only_owner: str,
) -> int:
    where = []
    params: list = []

    if search:
        where.append("(LOWER(case_id) LIKE ? OR LOWER(trans_id) LIKE ? OR LOWER(tags) LIKE ? OR LOWER(note) LIKE ?)")
        s = f"%{search.strip().lower()}%"
        params.extend([s, s, s, s])

    if status_list:
        placeholders = ",".join(["?"] * len(status_list))
        where.append(f"status IN ({placeholders})")
        params.extend([str(x) for x in status_list])

    if priority_list:
        placeholders = ",".join(["?"] * len(priority_list))
        where.append(f"priority IN ({placeholders})")
        params.extend([str(x) for x in priority_list])

    if owner_contains:
        where.append("LOWER(owner) LIKE ?")
        params.append(f"%{owner_contains.strip().lower()}%")

    if mine_only_owner:
        where.append("owner = ?")
        params.append(mine_only_owner)

    sql = "SELECT COUNT(*) AS c FROM cases"
    if where:
        sql += " WHERE " + " AND ".join(where)

    with get_conn() as conn:
        try:
            row = pd.read_sql_query(sql, conn, params=params).iloc[0]["c"]
            return int(row)
        except Exception:
            return 0

@st.cache_data(show_spinner=False)
def fetch_cases_filtered(
        search: str,
        status_list: list[str],
        priority_list: list[str],
        owner_contains: str,
        mine_only_owner: str,
        limit: int = 201,
        offset: int = 0,
) -> pd.DataFrame:
    where = []
    params: list = []

    if search:
        where.append("(LOWER(case_id) LIKE ? OR LOWER(trans_id) LIKE ? OR LOWER(tags) LIKE ? OR LOWER(note) LIKE ?)")
        s = f"%{search.strip().lower()}%"
        params.extend([s, s, s, s])

    if status_list:
        placeholders = ",".join(["?"] * len(status_list))
        where.append(f"status IN ({placeholders})")
        params.extend([str(x) for x in status_list])

    if priority_list:
        placeholders = ",".join(["?"] * len(priority_list))
        where.append(f"priority IN ({placeholders})")
        params.extend([str(x) for x in priority_list])

    if owner_contains:
        where.append("LOWER(owner) LIKE ?")
        params.append(f"%{owner_contains.strip().lower()}%")

    if mine_only_owner:
        where.append("owner = ?")
        params.append(mine_only_owner)

    sql = """SELECT case_id, trans_id, status, priority, owner, tags, note, created_at, updated_at
             FROM cases"""
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY datetime(updated_at) DESC LIMIT ? OFFSET ?"
    params.extend([int(limit), int(offset)])

    with get_conn() as conn:
        try:
            return pd.read_sql_query(sql, conn, params=params)
        except Exception:
            return pd.DataFrame(
                columns=["case_id", "trans_id", "status", "priority", "owner", "tags", "note", "created_at",
                         "updated_at"])


def update_case(case_id: str, status: str, priority: str, owner: str, tags: str, note: str,
                event_msg: str = "Oppdatert") -> None:
    """Oppdaterer case og logger *differ* i case_events.

    Logger:
      - status-endring som event_type='status' (f.eks. open → escalated)
      - endringer i priority/owner/tags/note som event_type='update'
    """
    now = datetime.utcnow().isoformat()

    with get_conn() as c:
        # Hent før-verdier (best effort)
        before = None
        try:
            before = c.execute(
                "SELECT status, priority, owner, tags, note FROM cases WHERE case_id=?",
                (case_id,),
            ).fetchone()
        except Exception:
            before = None

        c.execute(
            """UPDATE cases
                 SET status=?, priority=?, owner=?, tags=?, note=?, updated_at=?
                 WHERE case_id=?""",
            (status, priority, owner, tags, note, now, case_id),
        )

        # Diff-logging
        try:
            if before:
                b_status, b_pri, b_owner, b_tags, b_note = before
                # status
                if (b_status or "") != (status or ""):
                    ev_id = _new_id("EV")
                    c.execute(
                        """INSERT INTO case_events(event_id, case_id, event_type, message, created_at)
                             VALUES (?, ?, 'status', ?, ?)""",
                        (ev_id, case_id, f"status: {b_status} → {status}", now),
                    )
                # priority
                if (b_pri or "") != (priority or ""):
                    ev_id = _new_id("EV")
                    c.execute(
                        """INSERT INTO case_events(event_id, case_id, event_type, message, created_at)
                             VALUES (?, ?, 'update', ?, ?)""",
                        (ev_id, case_id, f"priority: {b_pri} → {priority}", now),
                    )
                # owner
                if (b_owner or "") != (owner or ""):
                    ev_id = _new_id("EV")
                    c.execute(
                        """INSERT INTO case_events(event_id, case_id, event_type, message, created_at)
                             VALUES (?, ?, 'update', ?, ?)""",
                        (ev_id, case_id, f"owner: {b_owner} → {owner}", now),
                    )
                # tags
                if (b_tags or "") != (tags or ""):
                    ev_id = _new_id("EV")
                    c.execute(
                        """INSERT INTO case_events(event_id, case_id, event_type, message, created_at)
                             VALUES (?, ?, 'update', ?, ?)""",
                        (ev_id, case_id, "tags oppdatert", now),
                    )
                # note (ikke dump hele teksten som diff)
                if (b_note or "") != (note or ""):
                    ev_id = _new_id("EV")
                    c.execute(
                        """INSERT INTO case_events(event_id, case_id, event_type, message, created_at)
                             VALUES (?, ?, 'update', ?, ?)""",
                        (ev_id, case_id, "note oppdatert", now),
                    )
            else:
                # fallback: én generell oppdateringshendelse
                ev_id = _new_id("EV")
                c.execute(
                    """INSERT INTO case_events(event_id, case_id, event_type, message, created_at)
                         VALUES (?, ?, 'update', ?, ?)""",
                    (ev_id, case_id, event_msg, now),
                )
        except Exception:
            # best effort
            pass

        c.commit()


def submit_case_for_approval(case_id: str, submitted_by: str, comment: str = "") -> None:
    """Send sak til maker-checker. Setter status=pending_approval og logger event."""
    now = datetime.utcnow().isoformat()
    with get_conn() as c:
        c.execute(
            """UPDATE cases
                 SET status='pending_approval',
                     submitted_by=?,
                     submitted_at=?,
                     approval_comment=?,
                     updated_at=?
               WHERE case_id=?""",
            (submitted_by or "analyst", now, comment or "", now, case_id),
        )
    log_case_event(case_id, "approval", f"Sendt til godkjenning av {submitted_by or 'analyst'}",
                   actor=submitted_by or "analyst")


def approve_case(case_id: str, approved_by: str, comment: str) -> None:
    now = datetime.utcnow().isoformat()
    with get_conn() as c:
        c.execute(
            """UPDATE cases
                 SET status='approved',
                     approved_by=?,
                     approved_at=?,
                     approval_comment=?,
                     updated_at=?
               WHERE case_id=?""",
            (approved_by or "checker", now, comment or "", now, case_id),
        )
    log_case_event(case_id, "approval", f"Godkjent av {approved_by or 'checker'}", actor=approved_by or "checker",
                   meta_json=json.dumps({"comment": comment or ""}))


def reject_case(case_id: str, approved_by: str, comment: str) -> None:
    now = datetime.utcnow().isoformat()
    with get_conn() as c:
        c.execute(
            """UPDATE cases
                 SET status='rejected',
                     approved_by=?,
                     approved_at=?,
                     approval_comment=?,
                     updated_at=?
               WHERE case_id=?""",
            (approved_by or "checker", now, comment or "", now, case_id),
        )
    log_case_event(case_id, "approval", f"Avvist av {approved_by or 'checker'}", actor=approved_by or "checker",
                   meta_json=json.dumps({"comment": comment or ""}))


def add_case_note(case_id: str, note: str, author: str = "system"):
    """Legg til et notat på en sak og logg det i case_events."""
    if not note or not str(note).strip():
        return
    now = datetime.utcnow().isoformat()
    nid = _new_id("NOTE")
    with get_conn() as c:
        c.execute(
            """INSERT INTO case_notes(note_id, case_id, author, note, created_at)
                 VALUES (?, ?, ?, ?, ?)""",
            (nid, case_id, author, str(note).strip(), now),
        )
        try:
            ev_id = _new_id("EV")
            c.execute(
                """INSERT INTO case_events(event_id, case_id, event_type, message, created_at)
                     VALUES (?, ?, 'note', ?, ?)""",
                (ev_id, case_id, f"{author}: {str(note).strip()}", now),
            )
        except Exception:
            pass
        c.commit()


# =============================
#   CASE WORKFLOW (AML-style)
# =============================

CASE_STATUSES = ["open", "in_review", "escalated", "sar_draft", "reported", "closed"]
CASE_PRIORITIES = ["low", "medium", "high"]

# Tillatte overganger (kan justeres)
_ALLOWED_TRANSITIONS = {
    "open": ["in_review", "escalated"],
    "in_review": ["escalated", "sar_draft", "closed"],
    "escalated": ["in_review", "sar_draft"],
    "sar_draft": ["reported", "in_review"],
    "reported": ["closed"],
    "closed": [],
}


def allowed_next_status(current: str) -> list[str]:
    cur = (current or "open").strip()
    return [cur] + _ALLOWED_TRANSITIONS.get(cur, [])


def _ensure_case_columns():
    """Best-effort migrations for new workflow fields."""
    cols = {
        "outcome": "TEXT",
        "closed_reason": "TEXT",
        "closed_by": "TEXT",
        "closed_at": "TEXT",
        "reported_by": "TEXT",
        "reported_at": "TEXT",
        "checker": "TEXT",
        "checker_at": "TEXT",
    }
    with get_conn() as c:
        cur = c.cursor()
        existing = {r[1] for r in cur.execute("PRAGMA table_info(cases)").fetchall()}
        for name, typ in cols.items():
            if name in existing:
                continue
            try:
                cur.execute(f"ALTER TABLE cases ADD COLUMN {name} {typ}")
            except Exception:
                pass
        c.commit()


def transition_case_status(
        case_id: str,
        old_status: str,
        new_status: str,
        actor: str,
        checker: str | None = None,
        outcome: str | None = None,
        closed_reason: str | None = None,
) -> tuple[bool, str]:
    """Validér workflow og oppdater status. Returnerer (ok, msg)."""
    actor = (actor or "system").strip() or "system"
    old_s = (old_status or "open").strip()
    new_s = (new_status or old_s).strip()

    if new_s not in CASE_STATUSES:
        return False, f"Ugyldig status: {new_s}"

    allowed = _ALLOWED_TRANSITIONS.get(old_s, [])
    if new_s != old_s and new_s not in allowed:
        return False, f"Ugyldig overgang: {old_s} → {new_s}"

    # Maker-checker for reported/closed
    if new_s in {"reported", "closed"}:
        chk = (checker or "").strip()
        if not chk:
            return False, "Maker-checker: Skriv inn checker-navn for å sette status til 'reported'/'closed'."
        if chk == actor:
            return False, "Maker-checker: Checker må være en annen person enn maker/actor."

    now = datetime.utcnow().isoformat()
    _ensure_case_columns()

    with get_conn() as c:
        cur = c.cursor()

        # Oppdater basestatus + workflow-felt
        sets = ["status=?", "updated_at=?"]
        params = [new_s, now]

        if new_s == "reported":
            sets += ["reported_at=?", "reported_by=?", "checker=?", "checker_at=?"]
            params += [now, actor, checker, now]
        elif new_s == "closed":
            sets += ["closed_at=?", "closed_by=?", "checker=?", "checker_at=?", "outcome=?", "closed_reason=?"]
            params += [now, actor, checker, now, outcome or "", closed_reason or ""]
        else:
            # rydde evt. outcome/reason ved reopening? Ikke automatisk.
            pass

        params.append(case_id)
        cur.execute(f"UPDATE cases SET {', '.join(sets)} WHERE case_id=?", params)

        # Logg status-event
        try:
            ev_id = _new_id("EV")
            cur.execute(
                """INSERT INTO case_events(event_id, case_id, event_type, message, created_at)
                     VALUES (?, ?, 'status', ?, ?)""",
                (ev_id, case_id, f"Status: {old_s} → {new_s} (by {actor})", now),
            )
        except Exception:
            pass

        c.commit()

    return True, "Oppdatert"


def auto_escalate_cases(run_reason: str = "SLA brutt", bump_priority: bool = True) -> int:
    """Auto-eskaler saker som har brutt SLA.

    - Finner cases med status open/in_review
    - Beregner SLA (samme logikk som i _case_sla_enrich)
    - Setter status='escalated' når SLA er brutt
    - Logger event i case_events

    Returnerer antall saker som ble eskalert.
    """
    now_iso = datetime.utcnow().isoformat()

    with get_conn() as c:
        df = pd.read_sql_query(
            """
            SELECT
                case_id,
                title,
                status,
                tier,
                risk_score,
                assigned_to,
                created_at,
                updated_at
            FROM cases
            WHERE status = 'open'
            ORDER BY created_at DESC
            """,
            c,
        )

        if df.empty:
            return 0

        df = _case_sla_enrich(df)

        # SLA brutt -> eskaler
        overdue = df[df.get("sla_status") == "🔴 Brutt"].copy()
        if overdue.empty:
            return 0

        escalated = 0
        for _, r in overdue.iterrows():
            cid = str(r["case_id"])
            prio = str(r.get("priority") or "medium").lower()
            new_prio = "high" if bump_priority and prio != "high" else prio

            # Oppdater case
            c.execute(
                """UPDATE cases
                     SET status='escalated',
                         priority=?,
                         updated_at=?
                   WHERE case_id=? AND status IN ('open','in_review')""",
                (new_prio, now_iso, cid),
            )

            # Logg event
            try:
                ev_id = _new_id("EV")
                msg = f"Auto-eskalert ({run_reason}). SLA brutt. priority={prio}→{new_prio}"
                c.execute(
                    """INSERT INTO case_events(event_id, case_id, event_type, message, created_at)
                         VALUES (?, ?, 'status', ?, ?)""",
                    (ev_id, cid, msg, now_iso),
                )
            except Exception:
                pass

            escalated += 1

        c.commit()

    # (valgfritt) Slack-varsel på aggregert nivå
    try:
        if escalated:
            slack_send(f"🚨 Auto-eskalering: {escalated} sak(er) eskalert pga. SLA-brudd.")
    except Exception:
        pass

    return escalated


def maybe_run_auto_escalation(min_seconds: int = 60) -> None:
    """Kjør auto-eskalering maks én gang per min_seconds (for å unngå DB-spam ved rerun)."""
    if not st.session_state.get("auto_escalation_enabled", True):
        return

    last = st.session_state.get("auto_escalation_last_run_utc")
    now = datetime.utcnow()

    try:
        last_dt = datetime.fromisoformat(last) if last else None
    except Exception:
        last_dt = None

    if last_dt and (now - last_dt).total_seconds() < float(min_seconds):
        return

    n = auto_escalate_cases(run_reason="periodisk", bump_priority=True)
    st.session_state["auto_escalation_last_run_utc"] = now.isoformat()
    if n:
        st.session_state["auto_escalation_last_count"] = int(n)


def risk_reasons_from_row(row: pd.Series) -> list[str]:
    reasons = []
    try:
        bel = float(row.get("beløp", 0) or 0)
        if bel >= 50000:
            reasons.append("Høyt beløp (>= 50k)")
    except Exception:
        pass
    land = str(row.get("land", "") or "")
    if land in {"IR", "KP", "SY", "RU"}:
        reasons.append("Høyrisikoland")
    try:
        ts = pd.to_datetime(row.get("tidspunkt"), errors="coerce")
        if pd.notna(ts) and 0 <= int(ts.hour) <= 5:
            reasons.append("Nattaktivitet (00–05)")
    except Exception:
        pass
    if bool(row.get("sanksjonert", False)):
        reasons.append("Treff på sanksjonsliste")
    if bool(row.get("fuzzy_sanksjonert", False)):
        reasons.append("Fuzzy treff på sanksjonsliste")
    if bool(row.get("mistenkelig", False)):
        reasons.append("Regelbasert flagg")
    if bool(row.get("mistenkelig_ml", False)):
        reasons.append("ML/anomali flagg")
    return reasons


# =============================
#   SLACK (enkel webhook-sender)
# =============================

def slack_send(text: str) -> bool:
    """Send en enkel Slack-melding via webhook. Sett SLACK_WEBHOOK_URL i miljøet."""
    hook = os.getenv("SLACK_WEBHOOK_URL")
    if not hook:
        return False
    try:
        data = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(hook, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
        return True
    except urllib.error.URLError:
        return False


# ============================
#   STREAMLIT CONFIG
# ============================

st.set_page_config(layout="wide", page_title="Anti-hvitvasking dashboard")

# --- API controls (optional) ---
sidebar_api_controls()

# If API-mode is enabled, route DB functions through wrappers.
# This keeps most of the existing dashboard code unchanged.
if st.session_state.get("use_api"):
    hent_transaksjoner = hent_transaksjoner_wrapper  # type: ignore
    lagre_til_db = lagre_transaksjoner_wrapper  # type: ignore

st.title("Anti-hvitvasking dashboard")
ensure_tables()

# Kjør auto-eskalering (rate-limited) ved oppstart/rerun
maybe_run_auto_escalation(min_seconds=60)

# -----------------------------
#   AUTH (DB-backed login)
# -----------------------------
with st.sidebar:
    #sidebar_auth_box()
    st.divider()
    st.markdown("### 🧩 Auto-grouping (case clustering)")
    st.checkbox(
        "Aktiver auto-grouping (gjenbruk sak på samme fra_konto)",
        value=bool(st.session_state.get("autogroup_enabled", True)),
        key="autogroup_enabled",
    )
    st.number_input(
        "Tidsvindu (timer)",
        min_value=1,
        max_value=720,
        value=int(st.session_state.get("autogroup_window_hours", 48)),
        step=1,
        key="autogroup_window_hours",
    )
    st.checkbox(
        "Krev samme til_konto innenfor vinduet",
        value=bool(st.session_state.get("autogroup_require_same_to", False)),
        key="autogroup_require_same_to",
    )
    st.slider(
        "Beløps-toleranse (± relativt til snitt i saken)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("autogroup_amount_tol", 0.15)),
        step=0.01,
        key="autogroup_amount_tol",
    )

    st.divider()
    st.markdown("### 🚨 Auto-eskalering")
    st.checkbox(
        "Aktivert",
        value=bool(st.session_state.get("auto_escalation_enabled", True)),
        key="auto_escalation_enabled",
    )
    colA, colB = st.columns(2)
    with colA:
        if st.button("Kjør nå", key="btn_run_auto_escalation"):
            n = auto_escalate_cases(run_reason="manuell", bump_priority=True)
            st.session_state["auto_escalation_last_run_utc"] = datetime.utcnow().isoformat()
            if n:
                st.success(f"Eskalerte {n} sak(er).")
            else:
                st.info("Ingen saker å eskalere.")
            st.rerun()
    with colB:
        lastn = st.session_state.get("auto_escalation_last_count")
        if lastn is not None:
            st.caption(f"Sist eskalert: {lastn}")


# -----------------------------
#   SMALL UI HELPERS

# -----------------------------
def _rerun_local():
    """Kompatibel rerun for ulike Streamlit-versjoner."""
    try:
        st.rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            pass


# =============================
#   DATA / PIPELINE HELPERS
# =============================

def simuler_transaksjoner(n=100, seed: int | None = 42):
    """Demo-generator. Sett seed=None for ekte random."""
    if seed is not None:
        np.random.seed(seed)
    transaksjoner = pd.DataFrame({
        "trans_id": [f"T{i + 1:04d}" for i in range(n)],
        "fra_konto": np.random.choice(["NO9386011117947", "NO9386011117948", "NO9386011117949"], size=n),
        "til_konto": np.random.choice(
            ["DE89370400440532013000", "FR7630006000011234567890189", "GB29NWBK60161331926819"], size=n),
        "beløp": np.round(np.random.uniform(1000, 100000, size=n), 2),
        "land": np.random.choice(["Norge", "Tyskland", "Frankrike", "UK", "Kina", "USA", "IR", "KP", "SY", "RU"],
                                 size=n),
        "tidspunkt": pd.date_range(end=pd.Timestamp.today(), periods=n).to_list(),
    })
    scores = np.concatenate([
        np.random.uniform(0.0, 0.5, n // 3),
        np.random.uniform(0.5, 0.8, n // 3),
        np.random.uniform(0.8, 1.0, n - 2 * (n // 3))
    ])
    np.random.shuffle(scores)
    transaksjoner["score"] = scores
    transaksjoner["risikonivå"] = transaksjoner["score"].apply(
        lambda x: "🔺 Høy" if x > 0.8 else ("⚡ Medium" if x > 0.5 else "✅ Lav")
    )
    return transaksjoner


def last_transaksjoner():
    """Prøv DB først; hvis tom -> lag et lite demo-sett én gang og lagre til DB."""
    try:
        df = hent_transaksjoner()
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        df = simuler_transaksjoner(n=50, seed=42)
        try:
            lagre_til_db(df)
            df = hent_transaksjoner()
        except Exception:
            pass

    if "tidspunkt" in df.columns:
        df["tidspunkt"] = pd.to_datetime(df["tidspunkt"], errors="coerce")

    for col in ["mistenkelig", "mistenkelig_ml", "sanksjonert", "fuzzy_sanksjonert"]:
        if col not in df.columns:
            df[col] = False

    if "score" not in df.columns:
        n = len(df)
        s = np.concatenate([
            np.random.uniform(0.0, 0.5, n // 3),
            np.random.uniform(0.5, 0.8, n // 3),
            np.random.uniform(0.8, 1.0, n - 2 * (n // 3)),
        ])
        np.random.shuffle(s)
        df["score"] = s

    if "risikonivå" not in df.columns:
        df["risikonivå"] = df["score"].apply(
            lambda x: "🔺 Høy" if x > 0.8 else ("⚡ Medium" if x > 0.5 else "✅ Lav")
        )
    return df


def beregn_risikoscore(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn score + forklaring per transaksjon.

    - df['score']: 0..1
    - df['risikonivå']: emoji-label
    - df['_risk_band']: Lav/Medium/Høy (ren label for filtering)
    - df['reasons_json']: JSON med komponentbidrag og triggere
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    df = df.copy()

    n = len(df)
    bel_comp = np.zeros(n, dtype=float)
    land_comp = np.zeros(n, dtype=float)
    natt_comp = np.zeros(n, dtype=float)
    sanc_comp = np.zeros(n, dtype=float)
    fuzzy_comp = np.zeros(n, dtype=float)

    # Beløp-komponent (normalisert 0..1 basert på max i datasettet)
    if "beløp" in df.columns:
        bel = pd.to_numeric(df["beløp"], errors="coerce").fillna(0.0)
        max_beløp = float(bel.max() or 0.0)
        if max_beløp > 0:
            bel_comp = (bel / max_beløp).to_numpy(dtype=float)

    # Land-komponent
    høyrisikoland = {"IR", "KP", "SY", "RU"}
    if "land" in df.columns:
        land_comp = df["land"].isin(høyrisikoland).to_numpy(dtype=float) * 0.5

    # Natt-komponent
    if "tidspunkt" in df.columns:
        tids = pd.to_datetime(df["tidspunkt"], errors="coerce")
        natt = tids.dt.hour.between(0, 5).fillna(False).to_numpy(dtype=bool)
        natt_comp = natt.astype(float) * 0.3

    # Sanksjon-komponenter
    if "sanksjonert" in df.columns:
        sanc_comp = df["sanksjonert"].fillna(False).to_numpy(dtype=bool).astype(float) * 0.5
    if "fuzzy_sanksjonert" in df.columns:
        fuzzy_comp = df["fuzzy_sanksjonert"].fillna(False).to_numpy(dtype=bool).astype(float) * 0.4

    total = bel_comp + land_comp + natt_comp + sanc_comp + fuzzy_comp
    df["score"] = np.clip(total, 0, 1)

    # Label + ren band
    def _label(x: float) -> str:
        return "🔺 Høy" if x > 0.8 else ("⚡ Medium" if x > 0.5 else "✅ Lav")

    df["risikonivå"] = df["score"].apply(lambda x: _label(float(x or 0)))

    df["_risk_band"] = df["risikonivå"].astype(str).apply(
        lambda s: "Høy" if "Høy" in s else ("Medium" if "Medium" in s else "Lav")
    )

    # Forklaring per rad (komponenter + triggere)
    reasons_list = []
    for i in range(n):
        trig = {
            "high_amount_ge_50k": False,
            "high_risk_country": False,
            "night_activity_00_05": False,
            "sanction_hit": False,
            "fuzzy_sanction_hit": False,
            "rule_flag": bool(df.iloc[i].get("mistenkelig", False)),
            "ml_flag": bool(df.iloc[i].get("mistenkelig_ml", False)),
        }

        try:
            bel_val = float(df.iloc[i].get("beløp", 0) or 0)
            trig["high_amount_ge_50k"] = bel_val >= 50000
        except Exception:
            pass

        try:
            trig["high_risk_country"] = str(df.iloc[i].get("land", "") or "") in høyrisikoland
        except Exception:
            pass

        try:
            ts = pd.to_datetime(df.iloc[i].get("tidspunkt"), errors="coerce")
            trig["night_activity_00_05"] = bool(pd.notna(ts) and 0 <= int(ts.hour) <= 5)
        except Exception:
            pass

        trig["sanction_hit"] = bool(df.iloc[i].get("sanksjonert", False))
        trig["fuzzy_sanction_hit"] = bool(df.iloc[i].get("fuzzy_sanksjonert", False))

        reasons = {
            "components": {
                "amount_norm": float(bel_comp[i]),
                "high_risk_country": float(land_comp[i]),
                "night_activity": float(natt_comp[i]),
                "sanction": float(sanc_comp[i]),
                "fuzzy_sanction": float(fuzzy_comp[i]),
            },
            "triggers": trig,
            "total_before_clip": float(total[i]),
            "score": float(df.iloc[i]["score"]),
            "risk_band": str(df.iloc[i]["_risk_band"]),
        }
        reasons_list.append(json.dumps(reasons, ensure_ascii=False))

    df["reasons_json"] = reasons_list
    return df


def sankey_transaksjoner(df: pd.DataFrame):
    if not set(["fra_konto", "til_konto", "beløp"]).issubset(df.columns):
        return
    df_group = df.groupby(["fra_konto", "til_konto"])["beløp"].sum().reset_index()
    labels = list(pd.unique(df_group[["fra_konto", "til_konto"]].values.ravel()))
    mapping = {label: i for i, label in enumerate(labels)}
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels),
        link=dict(
            source=df_group["fra_konto"].map(mapping),
            target=df_group["til_konto"].map(mapping),
            value=df_group["beløp"]
        )
    )])
    st.markdown("### 🔁 Transaksjonsflyt")
    st.plotly_chart(fig, use_container_width=True)


# =============================
#   KYC/EDD SCHEDULER + RE-SCREEN
# =============================

FREQ_MONTHS = {"LOW": 36, "MEDIUM": 24, "HIGH": 12, "CRITICAL": 6}

CHECKLIST = {
    ("PERSON", "STANDARD"): [
        ("verify_identity", "Bekreft identitet (ID + liveness)"),
        ("verify_address", "Bekreft adresse (PoA < 3 mnd)"),
        ("adverse_media", "Adverse media-søk"),
        ("approve", "Maker-checker godkjenning"),
    ],
    ("PERSON", "ENHANCED"): [
        ("verify_identity", "Forsterket ID (to uavh. kilder)"),
        ("verify_address", "Oppdatert PoA"),
        ("source_of_funds", "Kartlegg midlenes opprinnelse"),
        ("adverse_media", "Utvidet adverse media + PEP"),
        ("approve", "Senior-godkjenning"),
    ],
    ("BUSINESS", "STANDARD"): [
        ("verify_identity", "KYB: registrer foretak + styre"),
        ("verify_ubo", "Identifiser og verifiser UBO"),
        ("verify_address", "Virksomhetsadresse"),
        ("adverse_media", "Adverse media på selskap/UBO"),
        ("approve", "Maker-checker"),
    ],
    ("BUSINESS", "ENHANCED"): [
        ("verify_identity", "KYB utvidet (struktur/kjede)"),
        ("verify_ubo", "EDD på UBO (ID+PEP/sanksjon)"),
        ("source_of_funds", "Midlenes/velstands opprinnelse"),
        ("adverse_media", "Utvidet adverse media"),
        ("approve", "Senior-godkjenning"),
    ],
}


def next_review_date(risk_band: str, pep: bool, edd: bool, last_review: str | None) -> str:
    band = (risk_band or "MEDIUM").upper()
    months = FREQ_MONTHS.get(band, 24)
    if pep or edd or band in {"HIGH", "CRITICAL"}:
        months = min(months, 6)
    start = datetime.fromisoformat(last_review) if last_review else datetime.utcnow()
    return (start + timedelta(days=months * 30)).date().isoformat()  # enkel mnd-approx


def checklist_for(customer_type: str, pep: bool, edd: bool, risk_band: str):
    key = (customer_type or "PERSON",
           "ENHANCED" if (pep or edd or (risk_band or "MEDIUM").upper() in {"HIGH", "CRITICAL"}) else "STANDARD")
    return CHECKLIST.get(key, CHECKLIST[("PERSON", "STANDARD")])


def create_review(conn, customer_id: str, review_type: str, due_at_iso: str,
                  customer_type: str, pep: bool, edd: bool, risk_band: str):
    rid = f"R{int(datetime.utcnow().timestamp() * 1000)}{np.random.randint(1000):04d}"
    conn.execute("INSERT INTO kyc_reviews(review_id, customer_id, review_type, due_at) VALUES (?,?,?,?)",
                 (rid, customer_id, review_type, due_at_iso))
    for task_type, title in checklist_for(customer_type, pep, edd, risk_band):
        tid = f"TASK{int(datetime.utcnow().timestamp() * 1000)}{np.random.randint(1000):04d}"
        conn.execute(
            "INSERT INTO kyc_tasks(task_id, review_id, customer_id, task_type, title, status, created_at) "
            "VALUES (?,?,?,?,?,'open',?)",
            (tid, rid, customer_id, task_type, title, datetime.utcnow().isoformat())
        )
    conn.commit()
    return rid


def plan_reviews(conn):
    rows = conn.execute(
        "SELECT customer_id, name, type, risk_band, pep_flag, edd_required, last_review_at, next_review_at, kyc_status FROM customers"
    ).fetchall()
    created = 0
    for cid, cname, ctype, band, pep, edd, last_r, next_r, status in rows:
        band = (band or "MEDIUM").upper()
        nxt = next_review_date(band, bool(pep), bool(edd), last_r)
        if not next_r or next_r != nxt:
            conn.execute("UPDATE customers SET next_review_at=? WHERE customer_id=?", (nxt, cid))
        due = date.fromisoformat(nxt) <= date.today()
        if due and status != "in_review":
            create_review(conn, cid, "periodic", nxt, ctype or "PERSON", bool(pep), bool(edd), band)
            conn.execute("UPDATE customers SET kyc_status='in_review' WHERE customer_id=?", (cid,))
            created += 1
            slack_send(f"🔐 KYC: Opprettet PERIODIC review for {cname} ({cid}), band {band}, forfall {nxt}.")
    conn.commit()
    return created


def upsert_customers_from_df(conn, df: pd.DataFrame):
    """Hvis du ikke har kundetabell fra før: lag en enkel kundeliste basert på fra_konto."""
    if df is None or df.empty or "fra_konto" not in df.columns:
        return 0
    exists = pd.read_sql_query("SELECT customer_id FROM customers", conn)
    have = set(exists["customer_id"]) if not exists.empty else set()
    rows = 0
    for acc in sorted(df["fra_konto"].dropna().astype(str).unique()):
        if acc in have:
            continue
        name = f"Kunde {acc[-4:]}"
        band = "MEDIUM"
        conn.execute(
            "INSERT OR IGNORE INTO customers(customer_id, name, type, risk_band, risk_score, kyc_status) "
            "VALUES (?,?,?,?,?,?)",
            (acc, name, "PERSON", band, 0.5, "clear")
        )
        rows += 1
    conn.commit()
    return rows


# --- Re-screen (kunder + UBO) ---

def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _build_sanction_index(sanksjonsliste: pd.DataFrame) -> set[str]:
    cand_cols = [c for c in ["name", "navn", "entity", "person", "alias", "aka"] if c in sanksjonsliste.columns]
    sanc_names = set()
    for c in cand_cols:
        sanc_names.update(s for s in sanksjonsliste[c].dropna().astype(str).tolist())
    return {_norm(s) for s in sanc_names if s}


def _entity_hits(norm_name: str, sanc_norm: set[str]) -> bool:
    if not norm_name:
        return False
    if norm_name in sanc_norm:
        return True
    close = difflib.get_close_matches(norm_name, list(sanc_norm), n=2, cutoff=0.92)
    return len(close) > 0


def rescreen_customers(conn, sanksjonsliste: pd.DataFrame):
    """Re-screen alle kunder (ikke UBO) mot sanksjons-/PEP-listen."""
    if sanksjonsliste is None or getattr(sanksjonsliste, "empty", True):
        return {"matches": 0, "checked": 0}
    sanc_norm = _build_sanction_index(sanksjonsliste)

    df_c = pd.read_sql_query("SELECT customer_id, name, type, risk_band FROM customers", conn)
    hits = 0
    for _, r in df_c.iterrows():
        cid, cname, ctype, band = r["customer_id"], r["name"], r["type"], r["risk_band"]
        if _entity_hits(_norm(cname), sanc_norm):
            hits += 1
            today = date.today().isoformat()
            conn.execute("UPDATE customers SET pep_flag=1, edd_required=1 WHERE customer_id=?", (cid,))
            create_review(conn, cid, "event", today, ctype or "PERSON", True, True, band or "HIGH")
            slack_send(f"🚨 Re-screen: Treff på kunde {cname} ({cid}) – EDD/event review opprettet (forfall {today}).")
    conn.commit()
    return {"matches": hits, "checked": len(df_c)}


def rescreen_ubos(conn, sanksjonsliste: pd.DataFrame):
    """Re-screen alle UBO-er mot sanksjons-/PEP-listen, trigge EDD på tilhørende kunde."""
    if sanksjonsliste is None or getattr(sanksjonsliste, "empty", True):
        return {"matches": 0, "checked": 0}
    sanc_norm = _build_sanction_index(sanksjonsliste)

    df_u = pd.read_sql_query("SELECT ubo_id, customer_id, name, country, role FROM kyc_ubos", conn)
    if df_u.empty:
        return {"matches": 0, "checked": 0}

    df_c = pd.read_sql_query("SELECT customer_id, name, type, risk_band FROM customers", conn).set_index("customer_id")

    hits = 0
    for _, r in df_u.iterrows():
        ubo_id, cid, uname = r["ubo_id"], r["customer_id"], r["name"]
        if not uname or cid not in df_c.index:
            continue
        if _entity_hits(_norm(uname), sanc_norm):
            hits += 1
            c = df_c.loc[cid]
            today = date.today().isoformat()
            conn.execute("UPDATE customers SET edd_required=1 WHERE customer_id=?", (cid,))
            create_review(conn, cid, "event", today, c.get("type") or "PERSON", True, True,
                          c.get("risk_band") or "HIGH")
            slack_send(
                f"🚨 Re-screen UBO: Treff på {uname} (UBO for {c.get('name')} / {cid}) – EDD/event review opprettet.")
    conn.commit()
    return {"matches": hits, "checked": len(df_u)}


def notify_upcoming_reviews(conn, days: int = 7):
    """Slack-varsel for kunder med forfall innen 'days' dager eller over due."""
    df = pd.read_sql_query(
        "SELECT customer_id, name, next_review_at, risk_band FROM customers WHERE next_review_at IS NOT NULL",
        conn
    )
    if df.empty:
        return 0
    df["days"] = (pd.to_datetime(df["next_review_at"]) - pd.Timestamp.today().normalize()).dt.days
    due_df = df[(df["days"] <= days) | (df["days"] < 0)].copy()
    count = 0
    for _, r in due_df.iterrows():
        cid, name, nxt, band, d = r["customer_id"], r["name"], r["next_review_at"], r["risk_band"], r["days"]
        when = "FORFALT" if d < 0 else f"forfaller om {int(d)} dager"
        if slack_send(f"⏰ KYC: {name} ({cid}) – {when} (band {band}). Forfallsdato: {nxt}"):
            count += 1
    return count


# =============================
#   DEMO RESET
# =============================

def reset_demo(n_trans: int = 300, seed: int | None = None):
    """Tøm relevante tabeller og fyll databasen med friske demo-data."""
    gen = simuler_transaksjoner(n=n_trans, seed=seed)

    with get_conn() as c:
        cur = c.cursor()
        for tbl in ["transaksjoner", "customers", "kyc_reviews", "kyc_tasks", "kyc_documents", "kyc_ubos",
                    "vurderinger", "kundetiltak"]:
            try:
                cur.execute(f"DELETE FROM {tbl}")
            except Exception:
                pass
        c.commit()

        lagre_til_db(gen)
        upsert_customers_from_df(c, gen)

        # Demo-UBO for noen kunder
        try:
            custs = pd.read_sql_query("SELECT customer_id, name FROM customers LIMIT 5", c)
            for _, row in custs.iterrows():
                ubo_id = f"UBO{int(datetime.utcnow().timestamp() * 1000)}{np.random.randint(1000):04d}"
                c.execute(
                    "INSERT INTO kyc_ubos (ubo_id, customer_id, name, role, country, created_at) VALUES (?,?,?,?,?,?)",
                    (ubo_id, row["customer_id"], f"UBO for {row['name']}", "UBO", "NO", datetime.utcnow().isoformat())
                )
        except Exception:
            pass

        # Sett litt variert risikobånd og planlegg reviews
        try:
            c.execute("UPDATE customers SET risk_band='HIGH' WHERE rowid % 5 = 0")
            c.execute("UPDATE customers SET risk_band='LOW' WHERE rowid % 5 = 1")
            c.execute("UPDATE customers SET risk_band='MEDIUM' WHERE risk_band IS NULL OR risk_band=''")
            c.commit()
            plan_reviews(c)
        except Exception:
            pass

    st.session_state.df = last_transaksjoner()
    st.session_state.df = beregn_risikoscore(st.session_state.df)
    logg_hendelse("Demo reset", antall=len(gen))
    return gen


# =============================
#   PDF EXPORT HELPERS (reportlab)
# =============================

def _pdf_start(title: str):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    c.setTitle(title)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, height - 20 * mm, title)
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, height - 26 * mm, f"Generert: {datetime.utcnow().isoformat()}Z")
    c.line(20 * mm, height - 28 * mm, width - 20 * mm, height - 28 * mm)
    return c, buf, width, height


def _pdf_multiline(c, x_mm: float, y_start_mm: float, text: str, line_height: float = 5):
    width, height = A4
    y = y_start_mm
    for line in text.splitlines():
        c.drawString(x_mm * mm, (height - y * mm), line)
        y += line_height
    return y


def export_kyc_review_pdf(review_id: str) -> bytes | None:
    if not REPORTLAB_OK:
        return None
    conn = get_conn()
    r = pd.read_sql_query("""
        SELECT r.*, c.name, c.customer_id, c.type, c.risk_band, c.kyc_status, c.pep_flag, c.edd_required
        FROM kyc_reviews r JOIN customers c ON c.customer_id = r.customer_id
        WHERE r.review_id=?
    """, conn, params=(review_id,))
    if r.empty:
        return None
    row = r.iloc[0]
    tasks = pd.read_sql_query("""
        SELECT task_id, task_type, title, status, created_at, completed_at
        FROM kyc_tasks WHERE review_id=? ORDER BY created_at
    """, conn, params=(review_id,))
    conn.close()

    c, buf, width, height = _pdf_start(f"KYC Review – {row['name']} ({row['customer_id']})")
    y = 35
    c.setFont("Helvetica-Bold", 11)
    y = _pdf_multiline(c, 20, y, "Sammendrag")
    c.setFont("Helvetica", 10)
    summary = [
        f"Review-ID: {row['review_id']}",
        f"Kunde: {row['name']} ({row['customer_id']})",
        f"Type: {row.get('type', '')}, Risiko: {row.get('risk_band', '')}, Status: {row.get('kyc_status', '')}",
        f"Review-type: {row.get('review_type', '')}",
        f"Forfallsdato: {row.get('due_at', '')}",
        f"Startet: {row.get('started_at', '') or '-'}  Fullført: {row.get('completed_at', '') or '-'}  UtfalI: {row.get('outcome', '') or '-'}",
        f"PEP: {'Ja' if int(row.get('pep_flag', 0)) else 'Nei'}   EDD: {'Ja' if int(row.get('edd_required', 0)) else 'Nei'}",
    ]
    y = _pdf_multiline(c, 20, y + 6, "\n".join(summary))

    y += 6
    c.setFont("Helvetica-Bold", 11)
    y = _pdf_multiline(c, 20, y, "Sjekkliste")
    c.setFont("Helvetica", 10)
    if tasks.empty:
        y = _pdf_multiline(c, 20, y + 6, "- (ingen oppgaver)")
    else:
        for _, t in tasks.iterrows():
            line = f"[{'x' if t['status'] == 'done' else ' '}] {t['title']}  ({t['task_type']})"
            y = _pdf_multiline(c, 25, y + 5, line)

    y += 8
    c.setFont("Helvetica-Bold", 11)
    y = _pdf_multiline(c, 20, y, "Funn (JSON)")
    c.setFont("Helvetica", 9)
    findings = row.get("findings_json") or "{}"
    y = _pdf_multiline(c, 20, y + 6, findings)

    c.showPage()
    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf


def export_sar_pdf(review_id: str) -> bytes | None:
    """Generer en enkel SAR-kladd basert på review + transaksjoner (siste 30 dager for kunden)."""
    if not REPORTLAB_OK:
        return None
    conn = get_conn()
    r = pd.read_sql_query("""
        SELECT r.*, c.name, c.customer_id, c.type, c.risk_band
        FROM kyc_reviews r JOIN customers c ON c.customer_id = r.customer_id
        WHERE r.review_id=?
    """, conn, params=(review_id,))
    if r.empty:
        return None
    row = r.iloc[0]
    cust_id = row["customer_id"]

    tx = pd.read_sql_query("""
        SELECT * FROM transaksjoner WHERE fra_konto=? AND tidspunkt >= DATE('now','-30 day') ORDER BY tidspunkt DESC
    """, conn, params=(cust_id,))
    tasks = pd.read_sql_query("SELECT title,status FROM kyc_tasks WHERE review_id=?", conn, params=(review_id,))
    conn.close()

    c, buf, width, height = _pdf_start(f"SAR – Mistenkelig aktivitetsrapport (kladd)")
    y = 35
    c.setFont("Helvetica-Bold", 11)
    y = _pdf_multiline(c, 20, y, "Kunde & review")
    c.setFont("Helvetica", 10)
    y = _pdf_multiline(c, 20, y + 6, f"Kunde: {row['name']} ({cust_id})  | Risk: {row.get('risk_band', '')}")
    y = _pdf_multiline(c, 20, y + 5,
                       f"Review: {row['review_id']} ({row.get('review_type', '')})  Due: {row.get('due_at', '')}")

    y += 8
    c.setFont("Helvetica-Bold", 11)
    y = _pdf_multiline(c, 20, y, "Oppsummering (redigerbar kladd)")
    c.setFont("Helvetica", 10)
    kladd = (
        "- Beskriv observasjoner som indikerer mistenkelig aktivitet.\n"
        "- Nevn sanksjons-/PEP-treff, mønstre, geografiske risikoer, nattaktivitet eller uvanlige beløp.\n"
        "- Oppsummer tiltak: KYC/EDD, dokumenter, kontakt med kunde, interna eskaleringer.\n"
        "- Vurder om transaksjoner bør stanses eller kundeforhold avsluttes."
    )
    y = _pdf_multiline(c, 20, y + 6, kladd)

    y += 8
    c.setFont("Helvetica-Bold", 11)
    y = _pdf_multiline(c, 20, y, f"Transaksjoner (siste 30 dager) – {len(tx)} stk")
    c.setFont("Helvetica", 9)
    if tx.empty:
        y = _pdf_multiline(c, 20, y + 6, "(Ingen funnet for filteret.)")
    else:
        n = 0
        for _, t in tx.head(35).iterrows():
            n += 1
            line = f"{n:02d}. {t.get('tidspunkt', '')} | {t.get('beløp', '')} | {t.get('fra_konto', '')} → {t.get('til_konto', '')} | {t.get('land', '')}"
            y = _pdf_multiline(c, 20, y + 5, line)
            if y > 260:
                c.showPage();
                y = 20

    y += 8
    c.setFont("Helvetica-Bold", 11)
    y = _pdf_multiline(c, 20, y, "Sjekkliste-status")
    c.setFont("Helvetica", 10)
    if tasks.empty:
        y = _pdf_multiline(c, 20, y + 6, "- (ingen oppgaver)")
    else:
        done = int((tasks["status"] == "done").sum());
        total = len(tasks)
        y = _pdf_multiline(c, 20, y + 6, f"Ferdigstilt: {done}/{total}")

    c.showPage();
    c.save()
    pdf = buf.getvalue();
    buf.close()
    return pdf


# =============================
#   CHECKLIST & GAP HELPERS
# =============================

def checklist_load() -> pd.DataFrame:
    with get_conn() as c:
        df = pd.read_sql_query(
            "SELECT item_id, domain, item, owner, status, notes, updated_at FROM checklist ORDER BY domain, item_id",
            c
        )
    return df


def checklist_update(item_id: str, status: str | None = None, owner: str | None = None, notes: str | None = None):
    sets, params = [], []
    if status is not None: sets.append("status=?"); params.append(status)
    if owner is not None:  sets.append("owner=?");  params.append(owner)
    if notes is not None:  sets.append("notes=?");  params.append(notes)
    sets.append("updated_at=?");
    params.append(datetime.utcnow().isoformat())
    if not sets: return
    with get_conn() as c:
        c.execute(f"UPDATE checklist SET {', '.join(sets)} WHERE item_id=?", (*params, item_id))


def gap_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["domain", "open", "in_progress", "done", "total", "progress"])
    g = df.groupby(["domain", "status"]).size().unstack(fill_value=0)
    g["total"] = g.sum(axis=1)
    g["done"] = g.get("done", 0)
    g["open"] = g.get("open", 0)
    g["in_progress"] = g.get("in_progress", 0)
    g["progress"] = (g["done"] / g["total"] * 100).round(1)
    return g.reset_index()[["domain", "open", "in_progress", "done", "total", "progress"]]


def export_checklist_pdf() -> bytes | None:
    if not REPORTLAB_OK:
        return None
    df = checklist_load()
    c, buf, width, height = _pdf_start("Bank-klar sjekkliste")
    y = 35
    c.setFont("Helvetica", 10)
    cur_dom = None
    for _, r in df.iterrows():
        if r["domain"] != cur_dom:
            cur_dom = r["domain"]
            y += 7
            c.setFont("Helvetica-Bold", 11)
            y = _pdf_multiline(c, 20, y, f"{cur_dom}")
            c.setFont("Helvetica", 9)
        line = f"[{r['status']:^11}] {r['item']}  | owner: {r['owner'] or '-'}"
        y = _pdf_multiline(c, 25, y + 5, line[:110])
        if y > 270: c.showPage(); y = 20
    c.showPage();
    c.save()
    pdf = buf.getvalue();
    buf.close()
    return pdf


def export_gap_pdf() -> bytes | None:
    if not REPORTLAB_OK:
        return None
    df = checklist_load()
    s = gap_summary(df)
    c, buf, width, height = _pdf_start("Gap-analyse – oversikt")
    y = 35;
    c.setFont("Helvetica", 10)
    for _, r in s.iterrows():
        line = f"{r['domain']:<12} | total: {int(r['total'])}  done: {int(r['done'])}  in_prog: {int(r['in_progress'])}  open: {int(r['open'])}  progress: {r['progress']}%"
        y = _pdf_multiline(c, 20, y, line)
        if y > 270: c.showPage(); y = 20
    c.showPage();
    c.save()
    pdf = buf.getvalue();
    buf.close()
    return pdf


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Robust bool-kolonne: tåler 0/1, True/False, NaN, str."""
    if df is None or df.empty or col not in df.columns:
        return pd.Series(False, index=(df.index if df is not None else []))
    s = df[col]
    if str(s.dtype) == "bool":
        return s.fillna(False)
    if str(s.dtype).startswith(("int", "float")):
        return s.fillna(0).astype(float).ne(0)
    return s.fillna("").astype(str).str.lower().isin(["1", "true", "t", "yes", "y", "ja", "j", "x"])


def ensure_vurderinger_columns():
    """Best-effort migrering for å støtte strukturert avklaring."""
    with get_conn() as c:
        cur = c.cursor()
        try:
            cols = [r[1] for r in cur.execute("PRAGMA table_info(vurderinger)").fetchall()]
        except Exception:
            return
        add = []
        if "decision" not in cols: add.append(("decision", "TEXT"))
        if "checklist_json" not in cols: add.append(("checklist_json", "TEXT"))
        if "decided_by" not in cols: add.append(("decided_by", "TEXT"))
        if "decided_at" not in cols: add.append(("decided_at", "TEXT"))
        for name, typ in add:
            try:
                cur.execute(f"ALTER TABLE vurderinger ADD COLUMN {name} {typ}")
            except Exception:
                pass
        c.commit()


def get_case_by_trans_id(trans_id: str):
    """Hent case enten direkte på cases.trans_id eller via case_tx_links."""
    if not trans_id:
        return None
    trans_id = str(trans_id)

    with get_conn() as c:
        row = c.execute(
            "SELECT case_id, trans_id, status, priority, owner, tags, note, created_at, updated_at "
            "FROM cases WHERE trans_id=? ORDER BY datetime(created_at) DESC LIMIT 1",
            (trans_id,)
        ).fetchone()

        if not row:
            # fall back: sjekk om transaksjonen er linket til en eksisterende case
            try:
                row = c.execute(
                    "SELECT c.case_id, c.trans_id, c.status, c.priority, c.owner, c.tags, c.note, c.created_at, c.updated_at "
                    "FROM cases c JOIN case_tx_links l ON l.case_id=c.case_id "
                    "WHERE l.trans_id=? "
                    "ORDER BY datetime(c.created_at) DESC LIMIT 1",
                    (trans_id,)
                ).fetchone()
            except Exception:
                row = None

    if not row:
        return None
    keys = ["case_id", "trans_id", "status", "priority", "owner", "tags", "note", "created_at", "updated_at"]
    return dict(zip(keys, row))


def create_case_for_trans(trans_id: str, priority: str = "medium", owner: str = "", tags: str = "",
                          note: str = "") -> str | None:
    """Dedup + case linking:
    - Finner eksisterende case for trans_id (direkte eller via link)
    - Hvis ikke: forsøker å gjenbruke en åpen case på samme entity (fra_konto)
    - Hvis finnes: linker transaksjonen til den casen
    - Hvis ikke: oppretter ny case og linker transaksjonen
    """
    if not trans_id:
        return None

    trans_id = str(trans_id)

    # 1) Finn eksisterende case for akkurat denne transaksjonen
    existing = get_case_by_trans_id(trans_id)
    if existing:
        return existing["case_id"]

    now = datetime.utcnow().isoformat()

    # 2) Finn entity_key (fra_konto) for dedup
    entity_key = None
    tx_row = None
    try:
        with get_conn() as c:
            tx_row = c.execute(
                "SELECT trans_id, fra_konto, til_konto, beløp, land, tidspunkt FROM transaksjoner WHERE trans_id=? LIMIT 1",
                (trans_id,)
            ).fetchone()
    except Exception:
        tx_row = None

    if tx_row:
        try:
            entity_key = str(tx_row[1]) if tx_row[1] is not None else None
        except Exception:
            entity_key = None

    def _find_open_case_for_entity(conn, ekey: str):
        if not ekey:
            return None
        # Gjenbruk kun aktive saker
        try:
            row = conn.execute(
                "SELECT case_id FROM cases "
                "WHERE (entity_key=? OR tags LIKE ?) "
                "AND status IN ('open','in_review','escalated','sar_draft') "
                "ORDER BY datetime(updated_at) DESC LIMIT 1",
                (ekey, f"%{ekey}%")
            ).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    with get_conn() as c:
        # 3) Dedup på entity_key: gjenbruk åpen case hvis mulig
        reuse_case_id = (find_matching_case_for_tx(c, entity_key, tx_row,
                                                   window_hours=int(st.session_state.get('autogroup_window_hours', 48)),
                                                   require_same_to=bool(
                                                       st.session_state.get('autogroup_require_same_to', False)),
                                                   amount_tolerance=float(
                                                       st.session_state.get('autogroup_amount_tol', 0.15))) if bool(
            st.session_state.get('autogroup_enabled', True)) else None) if entity_key else None

        if reuse_case_id:
            # link transaksjon til eksisterende case
            try:
                c.execute(
                    "INSERT OR IGNORE INTO case_tx_links(case_id, trans_id, linked_at) VALUES (?,?,?)",
                    (reuse_case_id, trans_id, now)
                )
            except Exception:
                pass
            try:
                ev_id = _new_id("EV")
                c.execute(
                    "INSERT INTO case_events(event_id, case_id, event_type, message, created_at) VALUES (?,?,?,?,?)",
                    (ev_id, reuse_case_id, "update",
                     f"Transaksjon {trans_id} linket til case (dedup på fra_konto={entity_key})", now)
                )
            except Exception:
                pass
            try:
                c.execute("UPDATE cases SET updated_at=? WHERE case_id=?", (now, reuse_case_id))
            except Exception:
                pass
            c.commit()
            return reuse_case_id

        # 4) Opprett ny case
        case_id = _new_id("CASE")
        tag_str = tags or ""
        if entity_key and entity_key not in tag_str:
            tag_str = (tag_str + " " + entity_key).strip()

        # best effort: entity_key kolonne finnes kanskje ikke (avhengig av DB)
        try:
            c.execute(
                "INSERT INTO cases(case_id, trans_id, status, priority, owner, tags, note, created_at, updated_at, entity_key) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (case_id, trans_id, "open", priority, owner, tag_str, note, now, now, entity_key)
            )
        except Exception:
            c.execute(
                "INSERT INTO cases(case_id, trans_id, status, priority, owner, tags, note, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (case_id, trans_id, "open", priority, owner, tag_str, note, now, now)
            )

        # link transaksjon til casen
        try:
            c.execute(
                "INSERT OR IGNORE INTO case_tx_links(case_id, trans_id, linked_at) VALUES (?,?,?)",
                (case_id, trans_id, now)
            )
        except Exception:
            pass

        try:
            ev_id = _new_id("EV")
            c.execute(
                "INSERT INTO case_events(event_id, case_id, event_type, message, created_at) VALUES (?,?,?,?,?)",
                (ev_id, case_id, "create",
                 f"Case opprettet for transaksjon {trans_id}" + (f" (entity={entity_key})" if entity_key else ""), now)
            )
        except Exception:
            pass

        c.commit()

    return case_id


def get_case_trans_ids(case_id: str) -> list[str]:
    """Returner alle transaksjons-IDer knyttet til en case (inkl. primary)."""
    if not case_id:
        return []
    ids: list[str] = []
    with get_conn() as c:
        try:
            row = c.execute("SELECT trans_id FROM cases WHERE case_id=? LIMIT 1", (case_id,)).fetchone()
            if row and row[0]:
                ids.append(str(row[0]))
        except Exception:
            pass
        try:
            rows = c.execute("SELECT trans_id FROM case_tx_links WHERE case_id=? ORDER BY datetime(linked_at) ASC",
                             (case_id,)).fetchall()
            ids.extend([str(r[0]) for r in rows if r and r[0] is not None])
        except Exception:
            pass
    # unique preserve order
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x);
            out.append(x)
    return out


def generate_sar_draft_text(case_id: str, headline: str = "") -> str:
    """Generer en enkel SAR-kladd (tekst) basert på case + transaksjoner."""
    if not case_id:
        return ""
    tx_ids = get_case_trans_ids(case_id)
    with get_conn() as c:
        case_row = None
        try:
            case_row = c.execute(
                "SELECT case_id, status, priority, owner, tags, note, created_at, entity_key FROM cases WHERE case_id=? LIMIT 1",
                (case_id,)).fetchone()
        except Exception:
            case_row = c.execute(
                "SELECT case_id, status, priority, owner, tags, note, created_at FROM cases WHERE case_id=? LIMIT 1",
                (case_id,)).fetchone()

        tx = pd.DataFrame()
        if tx_ids:
            q_marks = ",".join(["?"] * len(tx_ids))
            try:
                tx = pd.read_sql_query(
                    f"SELECT * FROM transaksjoner WHERE trans_id IN ({q_marks}) ORDER BY datetime(tidspunkt) DESC", c,
                    params=tx_ids)
            except Exception:
                tx = pd.DataFrame()

    h = headline.strip() or f"SAR-kladd – Case {case_id}"
    lines = [h, "=" * len(h), ""]
    if case_row:
        # defensiv unpack
        lines += [
            f"Case-ID: {case_row[0]}",
            f"Status: {case_row[1]}  | Prioritet: {case_row[2]}  | Owner: {case_row[3] or '-'}",
            f"Tags/entity: {case_row[4] or '-'}",
            f"Opprettet: {case_row[6] if len(case_row) > 6 else ''}",
            ""
        ]
        if len(case_row) >= 8 and case_row[7]:
            lines.append(f"Primær enhet (fra_konto): {case_row[7]}")
            lines.append("")

    lines += [
        "Sammendrag (redigerbar):",
        "- Beskriv hvorfor aktiviteten vurderes som mistenkelig.",
        "- Pek på konkrete røde flagg (beløp, geografi, mønster, retur, sanksjon/PEP, nattaktivitet).",
        "- Oppsummer tiltak (kontakt, dokumentasjon, eskalering).",
        "",
        f"Transaksjoner knyttet til casen ({len(tx)}):"
    ]

    if tx.empty:
        lines.append("(Ingen transaksjoner funnet i DB for denne casen.)")
    else:
        for i, r in tx.head(50).iterrows():
            lines.append(
                f"- {r.get('tidspunkt', '')} | {r.get('beløp', '')} | {r.get('fra_konto', '')} → {r.get('til_konto', '')} | {r.get('land', '')} | score={r.get('score', '')}"
            )

    lines.append("")
    lines.append("Vurdering / Konklusjon:")
    lines.append("- [ ] False positive")
    lines.append("- [ ] Overvåk videre")
    lines.append("- [ ] SAR anbefalt / sendt")
    lines.append("")
    lines.append("Signatur:")
    lines.append("- Analytiker:")
    lines.append("- Checker (hvis relevant):")
    return "\n".join(lines)


# =============================
#   VIEW HELPERS
# =============================

def _rydde_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "tidspunkt" in df.columns:
        df["tidspunkt"] = pd.to_datetime(df["tidspunkt"], errors="coerce")
    for col in ["mistenkelig", "mistenkelig_ml", "sanksjonert", "fuzzy_sanksjonert"]:
        if col not in df.columns:
            df[col] = False
    df["score"] = pd.to_numeric(df.get("score", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["risikonivå"] = df["score"].apply(
        lambda x: "🔺 Høy" if (pd.notnull(x) and x > 0.8)
        else ("⚡ Medium" if (pd.notnull(x) and x > 0.5) else "✅ Lav")
    )
    # Praktisk filter-kolonne uten emoji (Lav/Medium/Høy)
    df["risiko_nivaa"] = df["score"].apply(
        lambda x: "Høy" if (pd.notnull(x) and x > 0.8)
        else ("Medium" if (pd.notnull(x) and x > 0.5) else "Lav")
    )

    return df


# =============================
#   SEARCH HELPERS (debounce + fuzzy)
# =============================

def _norm_q(s: str) -> str:
    return (s or "").strip().lower()


def _konto_options(series: pd.Series) -> list[str]:
    # cache-friendly unique sort
    try:
        return sorted(series.dropna().astype(str).unique().tolist())
    except Exception:
        return []


def _konto_filter_options(opts: list[str], q: str, fuzzy: bool, limit: int = 250) -> list[str]:
    qn = _norm_q(q)
    if not qn:
        return opts[:limit]
    # substring hits first
    sub = [o for o in opts if qn in o.lower()]
    if sub:
        return sub[:limit]
    if not fuzzy:
        return []
    # fuzzy fallback (difflib)
    # cutoff a bit lower for account strings
    matches = difflib.get_close_matches(qn, [o.lower() for o in opts], n=limit, cutoff=0.75)
    # map back to original
    low_map = {o.lower(): o for o in opts}
    out = [low_map[m] for m in matches if m in low_map]
    return out[:limit]


# =============================
#   CACHE & GLOBALS
# =============================

@st.cache_data(show_spinner=False)
def cached_sanksjonsliste():
    return hent_sanksjonsdata()


sanksjonsliste = cached_sanksjonsliste()

if "df" not in st.session_state:
    st.session_state.df = last_transaksjoner()

st.session_state.df = beregn_risikoscore(st.session_state.df)
df = st.session_state.df

# Bootstrap customers from transactions
with get_conn() as c:
    ny_kunder = upsert_customers_from_df(c, df)
    if ny_kunder:
        logg_hendelse("Auto-upsert kunder", ny_kunder)

# =============================


# =============================
#   SCENARIO / WHAT-IF
# =============================
def render_scenario_whatif(db_path: str) -> None:
    """Read-only simulering over transaksjoner. Endrer ikke DB."""
    st.subheader(" Scenario / What-if (simulering)")
    st.caption("Simuler terskler uten å endre data. Nyttig for policy, kapasitet og tilsyn.")

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            tx = pd.read_sql("SELECT * FROM transaksjoner", conn)
    except Exception as e:
        st.error(f"Kunne ikke lese transaksjoner fra DB: {e}")
        return

    if tx.empty:
        st.info("Fant ingen transaksjoner i databasen.")
        return

    # Robust kolonne-normalisering
    for col in [
        "trans_id", "fra_konto", "til_konto", "belop", "valuta", "timestamp", "kunde_id",
        "mottaker_id", "land", "mistenkelig_ml", "sanksjonert", "fuzzy_sanksjonert", "risikoscore"
    ]:
        if col not in tx.columns:
            tx[col] = None

    # Parse/clean
    def _parse_ts(x):
        try:
            return datetime.fromisoformat(str(x))
        except Exception:
            return None

    tx["_ts"] = tx["timestamp"].apply(_parse_ts)
    tx["_belop"] = pd.to_numeric(tx["belop"], errors="coerce")
    tx["_risk"] = pd.to_numeric(tx["risikoscore"], errors="coerce")
    tx["_ml"] = pd.to_numeric(tx["mistenkelig_ml"], errors="coerce").fillna(0).astype(int)
    tx["_san"] = (
        pd.to_numeric(tx["sanksjonert"], errors="coerce").fillna(0).astype(int)
        | pd.to_numeric(tx["fuzzy_sanksjonert"], errors="coerce").fillna(0).astype(int)
    ).astype(int)

    st.markdown("### 🎛️ Parametre")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lookback_days = st.number_input("Se bakover (dager)", min_value=1, max_value=3650, value=30, step=1)
    with c2:
        risk_thr = st.slider("Risiko-terskel (risikoscore)", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
    with c3:
        amount_thr = st.number_input("Beløp-terskel", min_value=0.0, value=250000.0, step=10000.0)
    with c4:
        hi_c = st.text_input("Høyrisikoland (komma-separert)", value="IR,KP,SY")

    st.markdown("### ⚙️ Regler")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        rule_sanctions_always = st.checkbox("Sanksjon-hit → eskalér", value=True)
    with r2:
        rule_ml_escalate = st.checkbox("ML-flag → eskalér", value=True)
    with r3:
        rule_risk_escalate = st.checkbox("Risikoscore ≥ terskel → eskalér", value=True)
    with r4:
        rule_amount_escalate = st.checkbox("Beløp ≥ terskel → eskalér", value=True)

    st.markdown("### 🌪️ Burst / struktur")
    b1, b2, b3 = st.columns(3)
    with b1:
        burst_hours = st.number_input("Burst-vindu (timer)", min_value=1, max_value=720, value=24, step=1)
    with b2:
        burst_count = st.number_input("Burst terskel (# tx per fra_konto)", min_value=2, max_value=200, value=10, step=1)
    with b3:
        burst_enable = st.checkbox("Aktiver burst-regel", value=True)

    # Lookback-filter
    now = datetime.utcnow()
    cutoff = now - timedelta(days=int(lookback_days))
    tx_lb = tx.copy()
    tx_lb = tx_lb[tx_lb["_ts"].notnull()]
    tx_lb = tx_lb[tx_lb["_ts"] >= cutoff]

    if tx_lb.empty:
        st.warning("Ingen transaksjoner i valgt tidsperiode.")
        return

    # High-risk country (signal)
    hi = {x.strip().upper() for x in str(hi_c).split(",") if x.strip()}
    tx_lb["_hi_country"] = tx_lb["land"].astype(str).str.upper().isin(hi).astype(int)

    # Burst rule (approksimasjon i v1: bucket per time)
    tx_lb["_burst"] = 0
    if burst_enable:
        tx_lb["_hour_bucket"] = pd.to_datetime(tx_lb["_ts"]).dt.floor("H")
        grp = tx_lb.groupby(["fra_konto", "_hour_bucket"]).size().reset_index(name="cnt")
        hot = grp[grp["cnt"] >= int(burst_count)][["fra_konto", "_hour_bucket"]]
        tx_lb = tx_lb.merge(hot.assign(_burst=1), how="left", on=["fra_konto", "_hour_bucket"])
        tx_lb["_burst"] = tx_lb["_burst"].fillna(0).astype(int)

    # Escalation logic
    flags = pd.Series(False, index=tx_lb.index)
    reasons = []

    if rule_sanctions_always:
        m = tx_lb["_san"] == 1
        flags = flags | m
        reasons.append(("SANCTIONS_HIT", m))

    if rule_ml_escalate:
        m = tx_lb["_ml"] == 1
        flags = flags | m
        reasons.append(("ML_FLAG", m))

    if rule_risk_escalate:
        m = tx_lb["_risk"].fillna(-1) >= float(risk_thr)
        flags = flags | m
        reasons.append(("RISK_SCORE", m))

    if rule_amount_escalate:
        m = tx_lb["_belop"].fillna(-1) >= float(amount_thr)
        flags = flags | m
        reasons.append(("AMOUNT", m))

    m_country = tx_lb["_hi_country"] == 1
    reasons.append(("HIGH_RISK_COUNTRY", m_country))

    if burst_enable:
        m = tx_lb["_burst"] == 1
        flags = flags | m
        reasons.append(("BURST_ACTIVITY", m))

    tx_lb["_escalate"] = flags.astype(int)

    st.markdown("### 📌 Resultat")
    total_tx = len(tx_lb)
    escalated = int(tx_lb["_escalate"].sum())
    pct = round(100.0 * escalated / max(1, total_tx), 2)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Transaksjoner (periode)", total_tx)
    k2.metric("Ville blitt eskalert", escalated)
    k3.metric("Eskalering (%)", f"{pct}%")

    throughput_per_day = st.number_input("Kapasitet (saker per dag)", min_value=1, max_value=10000, value=50, step=5)
    est_days = round(escalated / max(1, int(throughput_per_day)), 2)
    k4.metric("Estimat backlog (dager)", est_days)

    st.markdown("### 📊 Drivers")
    driver_rows = [{"driver": name, "hits": int(mask.sum())} for name, mask in reasons]
    drivers = pd.DataFrame(driver_rows).sort_values("hits", ascending=False)
    drivers = drivers[drivers["hits"] > 0]
    if not drivers.empty:
        st.bar_chart(drivers.set_index("driver"))
    else:
        st.info("Ingen drivere trigget i valgt scenario.")

    st.markdown("### 🔎 Eksempler (topp eskalerte transaksjoner)")
    cols = ["trans_id","fra_konto","til_konto","belop","valuta","timestamp","land","risikoscore","mistenkelig_ml","sanksjonert","fuzzy_sanksjonert"]
    view = tx_lb[tx_lb["_escalate"] == 1].copy()
    view = view.sort_values(by=["_risk", "_belop"], ascending=False, na_position="last")
    st.dataframe(view[[c for c in cols if c in view.columns]].head(200), use_container_width=True)

#   TABS
# =============================

(
    tab_all_tx,
    tab_mistenkelige,
    tab_stats,
    tab_scenario,
    tab_sanksjon,
    tab_avklaring,
    tab_historikk,
    tab_upload,
    tab_kundetiltak,
    tab_kyc,
    tab_rapporter,
    tab_bankklar,
    tab_alerts,
    tab_saker,
    tab_godkjenning,
) = st.tabs([
    " Alle transaksjoner",
    " Mistenkelige",
    " Statistikk & grafer",
    " Scenario / What-if",
    "️ Sanksjonsliste",
    " Avklaring",
    " Historikk",
    " Last opp CSV",
    " Kundetiltak",
    " KYC/EDD",
    " Rapporter (MTA/ROS)",
    " Bank-klar (sjekkliste & gap)",
    " Alerts",
    " Saker",
    " Godkjenning",
])
# ---------- Tab 0: Alle transaksjoner ----------
with tab_all_tx:
    st.subheader("📋 Alle transaksjoner")


    def _rerun():
        try:
            st.rerun()
        except Exception:
            try:
                st.rerun()
            except Exception:
                pass


    # One-click demo reset
    with st.expander("🧪 One-click demo reset", expanded=False):
        antall = st.number_input("Antall demo-transaksjoner", min_value=50, max_value=20000, value=500, step=50)
        stabilt = st.checkbox("Stabilt mønster (seed=42)", value=False)
        if st.button("🔄 Tøm og fyll demo-data på nytt"):
            gen = reset_demo(n_trans=int(antall), seed=(42 if stabilt else None))
            st.success(f"Demo er resatt. Genererte {len(gen)} transaksjoner, opprettet kunder/UBO og planla reviews.")
            _rerun()

    # Manuell analyse fra lokal CSV
    # (Skjules som standard – åpne ved behov)
    with st.expander("⚙️ Avanserte CSV-innstillinger", expanded=False):
        datoformat_input = st.text_input("Datoformat (valgfritt, f.eks. %d.%m.%Y %H:%M)", value="",
                                         key="datoformat_all_tx")
        har_header = st.checkbox("Filen har kolonnenavn", value=True, key="har_header_all_tx")

    if st.button("🔁 Kjør ny analyse nå"):
        with st.spinner("Analyserer transaksjoner og sjekker sanksjonsliste..."):
            try:
                datoformat = datoformat_input.strip() or None

                if not os.path.exists("transaksjoner.csv"):
                    st.warning("📁 Finner ikke `transaksjoner.csv`. Last opp under **📂 Last opp CSV** "
                               "eller legg filen ved siden av appen.")
                    st.stop()

                ny_df = trygg_les_csv("transaksjoner.csv", datoformat=datoformat, har_header=har_header)
                ny_df = analyser_transaksjoner(ny_df)
                ny_df = legg_til_anomalikluster(ny_df)
                ny_df = sjekk_mot_sanksjonsliste(ny_df, sanksjonsliste)

                eksisterende_ids = df["trans_id"].unique() if "trans_id" in df.columns else []
                nye_trans = ny_df[~ny_df["trans_id"].isin(eksisterende_ids)]

                if not nye_trans.empty:
                    lagre_til_db(nye_trans)

                    mistenkelige_nye = nye_trans[
                        _bool_series(nye_trans, "mistenkelig") |
                        _bool_series(nye_trans, "mistenkelig_ml") |
                        _bool_series(nye_trans, "sanksjonert") |
                        _bool_series(nye_trans, "fuzzy_sanksjonert")
                        ]
                    if not mistenkelige_nye.empty:
                        send_slack_varsel(mistenkelige_nye)

                st.session_state.df = last_transaksjoner()
                st.session_state.df = beregn_risikoscore(st.session_state.df)
                with get_conn() as c:
                    upsert_customers_from_df(c, st.session_state.df)
                st.success("✅ Analyse fullført og lagret!")

            except Exception as e:
                st.error(f"❌ Feil under analyse: {e}")
                st.stop()

    # Filtrering + tabell
    df = _rydde_df(st.session_state.get("df", pd.DataFrame()))
    if df.empty:
        st.info("Ingen transaksjoner funnet.")
    else:
        st.markdown("### 🔎 Filtrering")

        # ---- "Kulere" filter UX + DEBOUNCE: alle filtre påføres først når du trykker "Bruk filtre" ----
        # NB: Ikke skriv direkte til st.session_state for keys som er knyttet til widgets (Streamlit-feil).
        # Derfor bruker vi *_live som widget-keys, og kopierer over til *_applied når brukeren trykker "Bruk filtre".

        # Init "applied" defaults
        st.session_state.setdefault("filters_all_applied", {
            "konto_query": "",
            "konto_fuzzy": False,
            "konto_valgt": [],
            "land": [],
            "risk": ["Høy", "Medium", "Lav"],
            "date_range": None,  # settes når vi kjenner min/max
        })

        applied = st.session_state["filters_all_applied"]

        # --- Dato-min/max (for date_input) ---
        min_dato = max_dato = None
        if "tidspunkt" in df.columns:
            tidspunkt_dt = pd.to_datetime(df["tidspunkt"], errors="coerce")
            if tidspunkt_dt.notna().any():
                min_dato = tidspunkt_dt.min().date()
                max_dato = tidspunkt_dt.max().date()

        # Sett default date_range hvis ikke satt ennå
        if applied.get("date_range") is None and min_dato and max_dato:
            applied["date_range"] = (min_dato, max_dato)

        st.markdown("#### Filtre (trykk **Bruk filtre** for å oppdatere)")

        # --- Live widgets ---
        f1, f2, f3 = st.columns([1.2, 1.2, 1.6])

        with f1:
            konto_query_live = st.text_input(
                "Søk i fra_konto",
                value=applied.get("konto_query", ""),
                key="konto_query_all_live",
                placeholder="f.eks. 9386 eller NO93",
            )
            konto_fuzzy_live = st.checkbox(
                "Fuzzy-søk",
                value=bool(applied.get("konto_fuzzy", False)),
                key="konto_fuzzy_all_live",
                help="Hjelper hvis du skriver litt feil – kan være litt tregere på store lister.",
            )

        # Konto-velger (options kan snevres inn av live søk)
        konto_opts_full = _konto_options(df["fra_konto"]) if "fra_konto" in df.columns else []
        konto_opts = _konto_filter_options(
            konto_opts_full,
            konto_query_live,
            bool(konto_fuzzy_live),
            limit=300
        )

        with f2:
            konto_valgt_live = st.multiselect(
                "Filtrer på fra_konto (velg)",
                options=konto_opts,
                default=[k for k in applied.get("konto_valgt", []) if k in konto_opts],
                key="konto_multi_all_live",
                help="Tips: bruk søkefeltet til venstre for å snevre inn listen før du velger.",
            )

            land_opts = sorted(df["land"].dropna().unique()) if "land" in df.columns else []
            land_live = st.multiselect(
                "Filtrer på land",
                options=land_opts,
                default=[l for l in applied.get("land", []) if l in land_opts],
                key="land_multi_all_live",
            )

        with f3:
            risk_live = st.multiselect(
                "Filtrer på risikonivå",
                options=["Høy", "Medium", "Lav"],
                default=applied.get("risk", ["Høy", "Medium", "Lav"]),
                key="risk_multi_all_live",
            )

            dato_start = dato_slutt = None
            if min_dato and max_dato:
                dr_default = applied.get("date_range") or (min_dato, max_dato)
                # Streamlit krever at default ligger innenfor min/max.
                try:
                    _s, _e = dr_default
                except Exception:
                    _s, _e = (min_dato, max_dato)
                # Sørg for date-typer
                try:
                    _s = _s.date() if hasattr(_s, "date") else _s
                    _e = _e.date() if hasattr(_e, "date") else _e
                except Exception:
                    _s, _e = (min_dato, max_dato)
                # Clamp
                if _s < min_dato: _s = min_dato
                if _s > max_dato: _s = max_dato
                if _e < min_dato: _e = min_dato
                if _e > max_dato: _e = max_dato
                if _s > _e:
                    _s, _e = (min_dato, max_dato)
                dr_default = (_s, _e)
                dato_interval_live = st.date_input(
                    "Velg datointervall",
                    value=dr_default,
                    min_value=min_dato,
                    max_value=max_dato,
                    key="date_range_all_tx_live",
                    help="Velg start og slutt. (Velg samme dato for én dag.)",
                )
                if isinstance(dato_interval_live, (tuple, list)) and len(dato_interval_live) == 2:
                    dato_start, dato_slutt = dato_interval_live
                else:
                    dato_start = dato_interval_live
                    dato_slutt = dato_interval_live

        # --- Apply / reset ---
        b1, b2, _sp = st.columns([1, 1, 6])
        with b1:
            apply_filters = st.button("✅ Bruk filtre", key="apply_all_filters_all_tx")
        with b2:
            reset_filters = st.button("🧼 Nullstill", key="reset_all_filters_all_tx")

        if reset_filters:
            st.session_state["filters_all_applied"] = {
                "konto_query": "",
                "konto_fuzzy": False,
                "konto_valgt": [],
                "land": [],
                "risk": ["Høy", "Medium", "Lav"],
                "date_range": (min_dato, max_dato) if (min_dato and max_dato) else None,
            }
            try:
                st.rerun()
            except Exception:
                pass

        if apply_filters:
            st.session_state["filters_all_applied"] = {
                "konto_query": konto_query_live.strip(),
                "konto_fuzzy": bool(konto_fuzzy_live),
                "konto_valgt": konto_valgt_live,
                "land": land_live,
                "risk": risk_live if risk_live else ["Høy", "Medium", "Lav"],
                "date_range": (dato_start, dato_slutt) if (dato_start and dato_slutt) else None,
            }
            try:
                st.rerun()
            except Exception:
                pass

        # --- Bruk "applied" verdier i resten av logikken ---
        applied = st.session_state["filters_all_applied"]
        konto_query = applied.get("konto_query", "")
        konto_valgt = applied.get("konto_valgt", [])
        land_filter = applied.get("land", [])
        risikonivå_filter = applied.get("risk", ["Høy", "Medium", "Lav"])

        dato_start = dato_slutt = None
        if applied.get("date_range"):
            dato_start, dato_slutt = applied["date_range"]


        def _build_where_sql(konto_query: str, konto_valgt: list, land_filter: list,
                             risk_filter: list, dato_start, dato_slutt):
            clauses = []
            params = []

            # customer_id-søk (tidligere fra_konto)
            if konto_query:
                clauses.append("LOWER(t.customer_id) LIKE ?")
                params.append(f"%{konto_query.lower()}%")

            if konto_valgt:
                placeholders = ",".join(["?"] * len(konto_valgt))
                clauses.append(f"t.customer_id IN ({placeholders})")
                params.extend([str(k) for k in konto_valgt])

            if land_filter:
                placeholders = ",".join(["?"] * len(land_filter))
                clauses.append(f"UPPER(t.country) IN ({placeholders})")
                params.extend([str(x).upper() for x in land_filter])

            # Risiko: kommer fra alerts-tabellen (a.risk_level)
            if risk_filter:
                risk_clauses = []
                for r in risk_filter:
                    if r == "Høy":
                        risk_clauses.append("a.risk_level LIKE '%Høy%'")
                    elif r == "Medium":
                        risk_clauses.append("a.risk_level LIKE '%Medium%'")
                    elif r == "Lav":
                        risk_clauses.append("a.risk_level LIKE '%Lav%'")
                if risk_clauses:
                    clauses.append("(" + " OR ".join(risk_clauses) + ")")

            # Dato-intervall bruker transactions.timestamp
            if dato_start and dato_slutt:
                try:
                    start_iso = pd.Timestamp(dato_start).strftime("%Y-%m-%d 00:00:00")
                    end_iso = pd.Timestamp(dato_slutt).strftime("%Y-%m-%d 23:59:59")
                    clauses.append("datetime(t.timestamp) BETWEEN datetime(?) AND datetime(?)")
                    params.extend([start_iso, end_iso])
                except Exception:
                    pass

            where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            return where_sql, params


        def fetch_tx_count(where_sql: str, params: list) -> int:
            with get_conn() as _c:
                q = f"""
                SELECT COUNT(*) AS c
                FROM transactions t
                LEFT JOIN alerts a ON a.transaction_id = t.transaction_id
                {where_sql}
                """
                return int(pd.read_sql_query(q, _c, params=params)["c"].iloc[0])


        def fetch_tx_page(where_sql: str, params: list, limit: int, offset: int) -> pd.DataFrame:
            with get_conn() as _c:
                q = f"""
                SELECT
                    t.transaction_id AS trans_id,
                    t.customer_id AS fra_konto,
                    COALESCE(t.counterparty, '') AS til_konto,
                    t.amount AS beløp,
                    COALESCE(t.country, '') AS land,
                    COALESCE(a.risk_level, '—') AS risikonivå,
                    COALESCE(a.risk_score, 0.0) AS score,
                    t.timestamp AS tidspunkt
                FROM transactions t
                LEFT JOIN alerts a ON a.transaction_id = t.transaction_id
                {where_sql}
                ORDER BY datetime(t.timestamp) DESC, t.transaction_id DESC
                LIMIT ? OFFSET ?
                """
                p = list(params) + [int(limit), int(offset)]
                out = pd.read_sql_query(q, _c, params=p)

            if "tidspunkt" in out.columns:
                out["tidspunkt"] = pd.to_datetime(out["tidspunkt"], errors="coerce")
            return out

# ---------- Tab 1: Mistenkelige ----------
with tab_mistenkelige:
    st.subheader("⚠️ Mistenkelige transaksjoner")

    df = _rydde_df(st.session_state.get("df", pd.DataFrame()))
    if df.empty:
        st.info("Ingen transaksjoner i minnet.")
    else:
        mistenkelige_df = df[
            _bool_series(df, "mistenkelig") |
            _bool_series(df, "mistenkelig_ml") |
            _bool_series(df, "sanksjonert") |
            _bool_series(df, "fuzzy_sanksjonert")
            ].copy()

        if mistenkelige_df.empty:
            st.info("Ingen mistenkelige transaksjoner funnet.")
        else:
            st.markdown("### 📋 Oversikt over flaggede transaksjoner")
            vis_kolonner = ["trans_id", "fra_konto", "til_konto", "beløp", "land", "risikonivå"]
            if "fuzzy_sanksjonert" in mistenkelige_df.columns:
                vis_kolonner.append("fuzzy_sanksjonert")
            st.dataframe(mistenkelige_df[[k for k in vis_kolonner if k in mistenkelige_df.columns]],
                         use_container_width=True)

            csv = mistenkelige_df.to_csv(index=False).encode("utf-8")
            st.download_button("📤 Last ned som CSV", data=csv, file_name="mistenkelige_transaksjoner.csv",
                               mime="text/csv")

# ---------- Tab 2: Statistikk & grafer ----------
with tab_stats:
    st.subheader("📈 Statistikk og visuelle innsikter")

    # Dual mode: global oversikt vs bruk aktive filtre (applied) fra "Alle transaksjoner"
    mode = st.radio(
        "Datagrunnlag",
        ["Global oversikt", "Bruk aktive filtre"],
        horizontal=True,
        help="Global viser hele datasettet (større risikobilde). 'Bruk aktive filtre' viser tall/grafer for filteret du har brukt i 'Alle transaksjoner'.",
        key="stats_mode",
    )

    df_all = _rydde_df(st.session_state.get("df", pd.DataFrame()))
    if df_all.empty:
        st.info("Ingen data tilgjengelig for statistikk.")
        st.stop()

    df_view = df_all

    if mode == "Bruk aktive filtre":
        applied = st.session_state.get("filters_all_applied", {}) or {}
        konto_query = (applied.get("konto_query") or "").strip()
        konto_valgt = applied.get("konto_valgt") or []
        land_filter = applied.get("land") or []
        risk_levels = applied.get("risk") or ["Høy", "Medium", "Lav"]
        date_range = applied.get("date_range")

        dato_start = dato_slutt = None
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            dato_start, dato_slutt = date_range[0], date_range[1]

        # Hent et relativt stort utsnitt for statistikk (og klipp for plots senere)
        try:
            df_view = fetch_transactions_filtered(
                konto_query=konto_query,
                konto_valgt=konto_valgt,
                land_filter=land_filter,
                risk_levels=risk_levels,
                dato_start=dato_start,
                dato_slutt=dato_slutt,
                limit=50000,
                offset=0,
            )
            df_view = _rydde_df(df_view)
        except Exception:
            # fallback: filtrer i pandas dersom DB ikke er tilgjengelig
            df_view = df_all.copy()
            if konto_valgt and "fra_konto" in df_view.columns:
                df_view = df_view[df_view["fra_konto"].astype(str).isin([str(x) for x in konto_valgt])]
            if konto_query and "fra_konto" in df_view.columns:
                df_view = df_view[df_view["fra_konto"].astype(str).str.contains(konto_query, na=False, case=False)]
            if land_filter and "land" in df_view.columns:
                df_view = df_view[df_view["land"].isin(land_filter)]
            if risk_levels and "risikonivå" in df_view.columns:
                df_view["_risk_band"] = df_view["risikonivå"].apply(_std_risk)
                df_view = df_view[df_view["_risk_band"].isin(risk_levels)]
            if dato_start and dato_slutt and "tidspunkt" in df_view.columns:
                ts = pd.to_datetime(df_view["tidspunkt"], errors="coerce")
                start_ts = pd.Timestamp(dato_start)
                end_ts = pd.Timestamp(dato_slutt) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                df_view = df_view[ts.notna() & (ts >= start_ts) & (ts <= end_ts)]

        if df_view.empty:
            st.warning("Ingen treff for aktive filtre.")
            st.stop()

    # -------------------------
    # KPIer (også litt "bank-ish")
    # -------------------------
    c1, c2, c3, c4 = st.columns(4)

    total = len(df_view)
    flagged_rules = int(df_view.get("mistenkelig", pd.Series(False, index=df_view.index)).fillna(False).sum())
    flagged_ml = int(df_view.get("mistenkelig_ml", pd.Series(False, index=df_view.index)).fillna(False).sum())
    high_risk = int((df_view.get("score", pd.Series(0, index=df_view.index)).fillna(0) > 0.8).sum())

    c1.metric("🔢 Antall", total)
    c2.metric("⚠️ Flagget (regler)", flagged_rules)
    c3.metric("🤖 Flagget (ML)", flagged_ml)
    c4.metric("🔺 Høy risiko", high_risk)

    # Risk velocity: siste 24t og 7d (basert på tidspunkt)
    if "tidspunkt" in df_view.columns:
        ts = pd.to_datetime(df_view["tidspunkt"], errors="coerce")
        now = pd.Timestamp.now(tz="UTC")
        ts_utc = ts.dt.tz_localize("UTC") if getattr(ts.dt, "tz", None) is None else ts.dt.tz_convert("UTC")

        last_24 = df_view[ts_utc >= (now - pd.Timedelta(hours=24))]
        last_7d = df_view[ts_utc >= (now - pd.Timedelta(days=7))]

        v1, v2, v3 = st.columns(3)
        v1.metric("⏱️ Siste 24t – antall", len(last_24))
        v2.metric("⏱️ Siste 24t – 🔺 høy",
                  int((last_24.get("score", pd.Series(0, index=last_24.index)).fillna(0) > 0.8).sum()))
        v3.metric("📆 Siste 7 dager – 🔺 høy",
                  int((last_7d.get("score", pd.Series(0, index=last_7d.index)).fillna(0) > 0.8).sum()))

    st.divider()

    # For plots: klipp om datasettet er veldig stort (UI/ytelse)
    df_plot = df_view.copy()
    if len(df_plot) > 20000:
        df_plot = df_plot.head(20000)

    # Land-fordeling
    if "land" in df_plot.columns:
        st.markdown("### 🌍 Antall transaksjoner per land")
        st.bar_chart(df_plot["land"].value_counts().head(30))

    # Beløp over tid
    if "tidspunkt" in df_plot.columns and "beløp" in df_plot.columns:
        st.markdown("### 📈 Beløpsutvikling over tid")
        df_sorted = df_plot.sort_values("tidspunkt")
        try:
            st.line_chart(df_sorted.set_index("tidspunkt")["beløp"].tail(250))
        except Exception:
            pass

    # Risiko per land
    if "score" in df_plot.columns and "land" in df_plot.columns:
        st.markdown("### 🧠 Risiko per land (gj.snittlig score)")
        risiko_per_land = df_plot.groupby("land")["score"].mean().sort_values(ascending=False)
        st.bar_chart(risiko_per_land.head(30))

    # Risikonivå
    if "risikonivå" in df_plot.columns:
        st.markdown("### 🚨 Fordeling av risikonivå")
        st.bar_chart(df_plot["risikonivå"].value_counts())

    # Sankey kan bli tung - vis kun for moderate datasett
    if len(df_plot) <= 5000:
        sankey_transaksjoner(df_plot)
    else:
        st.info("Sankey-graf er slått av for store datasett (over 5 000 rader) for å holde UI raskt.")

# ---------- Tab 3: 🧪 Scenario / What-if ----------
with tab_scenario:
    render_scenario_whatif(DB_PATH)

# ---------- Tab 3: PEP/Sanksjonsliste ----------
with tab_sanksjon:
    st.subheader("🛡️ PEP- og sanksjonssjekk")
    try:
        if sanksjonsliste is None or getattr(sanksjonsliste, "empty", True):
            st.warning("Ingen data i sanksjonslisten.")
        else:
            st.write(f"Totalt i sanksjonslisten: {len(sanksjonsliste)}")
            st.dataframe(sanksjonsliste, use_container_width=True)
            if "land" in sanksjonsliste.columns:
                st.markdown("### 🌍 Fordeling per land (i sanksjonslisten)")
                st.bar_chart(sanksjonsliste["land"].value_counts())
    except Exception as e:
        st.error(f"❌ Klarte ikke laste sanksjonslisten: {e}")

# ---------- Tab 4: Avklaring ----------

with tab_avklaring:
    st.subheader("✅ Avklaring – etterforskning, vurdering og beslutning")

    ensure_vurderinger_columns()
    analyst = (st.session_state.get("analyst_name") or "analytiker").strip()
    df = _rydde_df(st.session_state.get("df", pd.DataFrame()))
    if df.empty:
        st.warning("⚠️ Ingen transaksjoner å avklare.")
        st.stop()

    flag = (
            _bool_series(df, "mistenkelig") |
            _bool_series(df, "mistenkelig_ml") |
            _bool_series(df, "sanksjonert") |
            _bool_series(df, "fuzzy_sanksjonert") |
            (pd.to_numeric(df.get("score", pd.Series(0, index=df.index)), errors="coerce").fillna(0) > 0.8)
    )
    cand = df.loc[flag].copy()

    if cand.empty:
        st.info("Ingen flaggede transaksjoner akkurat nå.")
        st.stop()

    with get_conn() as c:
        v = pd.read_sql_query(
            "SELECT trans_id, kommentar, avklart, decision, checklist_json, decided_by, decided_at FROM vurderinger", c)
    if not v.empty:
        v["trans_id"] = v["trans_id"].astype(str)
        cand["trans_id"] = cand["trans_id"].astype(str)
        cand = cand.merge(v, on="trans_id", how="left")

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### 📌 Velg transaksjon")

        show_open_only = st.checkbox("Vis kun ikke-avklarte", value=True, key="avk_only_open")
        view = cand.copy()
        if show_open_only:
            view = view[~view.get("avklart", pd.Series(False, index=view.index)).fillna(False).astype(bool)]

        view["__score"] = pd.to_numeric(view.get("score", 0), errors="coerce").fillna(0)
        view["__ts"] = pd.to_datetime(view.get("tidspunkt", pd.NaT), errors="coerce")
        view = view.sort_values(["__score", "__ts"], ascending=[False, False])

        options = [
            f"{r.get('trans_id', '')} | {r.get('fra_konto', '')} → {r.get('til_konto', '')} | {r.get('beløp', '')} | {r.get('land', '')} | score {float(r.get('__score', 0)):.2f}"
            for _, r in view.head(500).iterrows()
        ]
        if not options:
            st.info("Ingen matcher filteret.")
            st.stop()

        picked = st.selectbox("Transaksjon", options, key="avk_pick_tx")
        trans_id = picked.split("|")[0].strip()

        case = get_case_by_trans_id(trans_id)
        if case:
            st.caption(
                f"🔎 Case: {case['case_id']} • status **{case['status']}** • priority **{case['priority']}** • owner **{case.get('owner') or '-'}**")
        else:
            st.caption("Ingen case knyttet til denne transaksjonen ennå.")

    with right:
        row = cand[cand["trans_id"].astype(str) == str(trans_id)].iloc[0].to_dict()
        st.markdown("### 🧾 Kontekst")
        c1, c2, c3 = st.columns(3)
        c1.metric("Beløp", f"{row.get('beløp', '')}")
        c2.metric("Land", f"{row.get('land', '')}")
        c3.metric("Score", f"{float(pd.to_numeric(row.get('score', 0), errors='coerce') or 0):.2f}")

        st.write({
            "trans_id": row.get("trans_id"),
            "fra_konto": row.get("fra_konto"),
            "til_konto": row.get("til_konto"),
            "tidspunkt": str(row.get("tidspunkt")),
            "risikonivå": row.get("risikonivå"),
            "mistenkelig": bool(row.get("mistenkelig", False)),
            "mistenkelig_ml": bool(row.get("mistenkelig_ml", False)),
            "sanksjonert": bool(row.get("sanksjonert", False)),
            "fuzzy_sanksjonert": bool(row.get("fuzzy_sanksjonert", False)),
        })

with st.expander("👤 Entity-profil (fra_konto)", expanded=False):
    acc = str(row.get("fra_konto") or "").strip()
    if not acc:
        st.info("Ingen fra_konto på denne transaksjonen.")
    else:
        dff_all = _rydde_df(st.session_state.get("df", pd.DataFrame()))
        dff_acc = dff_all[dff_all.get("fra_konto", pd.Series(dtype=str)).astype(str) == acc].copy()
        dff_acc["__score"] = pd.to_numeric(dff_acc.get("score", 0), errors="coerce").fillna(0)
        dff_acc["__bel"] = pd.to_numeric(dff_acc.get("beløp", 0), errors="coerce").fillna(0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transaksjoner", int(len(dff_acc)))
        m2.metric("Sum beløp", f"{float(dff_acc['__bel'].sum()):,.0f}" if len(dff_acc) else "0")
        m3.metric("Snitt score", f"{float(dff_acc['__score'].mean()):.2f}" if len(dff_acc) else "0.00")
        m4.metric("Max score", f"{float(dff_acc['__score'].max()):.2f}" if len(dff_acc) else "0.00")

        if "land" in dff_acc.columns and not dff_acc.empty:
            st.markdown("**Topp land**")
            st.bar_chart(dff_acc["land"].value_counts().head(10))

        st.markdown("**Siste transaksjoner (konto)**")
        cols = [c for c in ["trans_id", "tidspunkt", "til_konto", "beløp", "land", "risikonivå", "score"] if
                c in dff_acc.columns]
        if cols:
            st.dataframe(dff_acc.sort_values("tidspunkt", ascending=False).head(25)[cols], use_container_width=True)

        st.markdown("**Relaterte saker (samme fra_konto)**")
        with get_conn() as c:
            try:
                cases_rel = pd.read_sql_query(
                    "SELECT case_id, status, priority, owner, created_at, updated_at "
                    "FROM cases WHERE (entity_key=? OR tags LIKE ?) "
                    "ORDER BY datetime(updated_at) DESC LIMIT 200",
                    c, params=(acc, f"%{acc}%")
                )
            except Exception:
                cases_rel = pd.read_sql_query(
                    "SELECT case_id, status, priority, owner, created_at, updated_at "
                    "FROM cases WHERE tags LIKE ? "
                    "ORDER BY datetime(updated_at) DESC LIMIT 200",
                    c, params=(f"%{acc}%",)
                )
        if cases_rel.empty:
            st.caption("Ingen saker funnet på denne kontoen ennå.")
        else:
            st.dataframe(cases_rel, use_container_width=True)

        st.markdown("**SAR-kladd (tekst)**")
        if st.button("🧾 Generer SAR-kladd for aktiv case", key=f"sar_draft_btn_{trans_id}"):
            if not case:
                cid = create_case_for_trans(trans_id, priority="medium", owner=st.session_state.get("analyst_name", ""))
                case = get_case_by_trans_id(trans_id)
            if case:
                sar_text = generate_sar_draft_text(case["case_id"])
                st.session_state[f"sar_text_{case['case_id']}"] = sar_text
                st.success("SAR-kladd generert.")
                log_case_event(case["case_id"], "sar",
                               f"SAR-kladd generert av {st.session_state.get('analyst_name', '') or 'analytiker'}")
        if case and st.session_state.get(f"sar_text_{case['case_id']}"):
            st.download_button(
                "⬇️ Last ned SAR-kladd (txt)",
                data=st.session_state.get(f"sar_text_{case['case_id']}", ""),
                file_name=f"sar_draft_{case['case_id']}.txt",
                mime="text/plain",
                key=f"sar_dl_{case['case_id']}"
            )

        st.markdown("---")
        st.markdown("### 📜 Tidslinje (case)")
        if case:
            ev = load_case_events(case["case_id"])
            render_case_timeline(ev, key_prefix=f"tl_{case['case_id']}")
            with st.expander("🔎 Rå hendelseslogg (tabell)"):
                st.dataframe(ev, use_container_width=True)
        else:
            st.caption("Opprett eller velg en case for å se tidslinje.")

        st.markdown("### 🧠 Vurderingspunkter (strukturert)")
        existing_chk = {}
        try:
            if isinstance(row.get("checklist_json"), str) and row.get("checklist_json"):
                existing_chk = json.loads(row.get("checklist_json")) or {}
        except Exception:
            existing_chk = {}

        chk_items = [
            ("unusual_amount", "Uvanlig beløp ift. historikk"),
            ("unusual_receiver", "Uvanlig mottaker / ny mottaker"),
            ("high_risk_geo", "Høyrisiko geografi (land/region)"),
            ("structuring", "Mistenkt strukturering / oppsplitting"),
            ("return_pattern", "Retur-/sirkelmønster observert"),
            ("sanctions_pep", "Sanksjon/PEP-indikasjon"),
            ("customer_explained", "Kunde har forklart forholdet"),
            ("documentation_ok", "Dokumentasjon innhentet/verifisert"),
        ]

        checklist = {}
        for key, label in chk_items:
            checklist[key] = st.checkbox(label, value=bool(existing_chk.get(key, False)),
                                         key=f"avk_chk_{trans_id}_{key}")

        st.markdown("### 🧾 Notat og beslutning")
        prev_comment = row.get("kommentar") if isinstance(row.get("kommentar"), str) else ""
        kommentar = st.text_area("Fritekstnotat (lagres)", value=prev_comment, height=120, key=f"avk_note_{trans_id}")

        decision_map = {
            "False positive (ingen SAR)": "false_positive",
            "Overvåk videre": "monitor",
            "SAR anbefalt (kladd)": "sar_recommended",
        }
        prev_dec = row.get("decision") if isinstance(row.get("decision"), str) else ""
        inv = {v: k for k, v in decision_map.items()}
        default_label = inv.get(prev_dec, "Overvåk videre")
        decision_label = st.radio("Konklusjon", list(decision_map.keys()),
                                  index=list(decision_map.keys()).index(default_label),
                                  key=f"avk_dec_{trans_id}")
        decision = decision_map[decision_label]

        st.markdown("### 🗂️ Case-håndtering")
        pri = "medium"
        rl = _std_risk(row.get("risikonivå", ""))
        if rl == "Høy":
            pri = "high"
        elif rl == "Lav":
            pri = "low"

        colA, colB, colC = st.columns([1, 1, 1.2])
        with colA:
            if st.button("➕ Opprett case (hvis mangler)", key=f"avk_create_case_{trans_id}"):
                cid = create_case_for_trans(trans_id, priority=pri, owner=st.session_state["analyst_name"])
                if cid:
                    st.success(f"Case opprettet: {cid}")
                    st.rerun()
        with colB:
            if st.button("📎 Ta eierskap", key=f"avk_take_owner_{trans_id}"):
                c = get_case_by_trans_id(trans_id)
                if c:
                    update_case(c["case_id"], c["status"], c["priority"], st.session_state["analyst_name"],
                                c.get("tags", "") or "", c.get("note", "") or "", event_msg="Eier satt via Avklaring")
                    st.success("Eierskap oppdatert.")
                    st.rerun()
                else:
                    st.info("Opprett case først.")
        with colC:
            checker = st.text_input("Checker (kun for lukking)", value="", placeholder="Navn (annen enn analytiker)",
                                    key=f"avk_checker_{trans_id}")

        target_status = "in_review"
        if decision == "false_positive":
            target_status = "closed"
        elif decision == "sar_recommended":
            target_status = "sar_draft"

        if st.button("💾 Lagre avklaring", key=f"avk_save_{trans_id}"):
            now = datetime.utcnow().isoformat()
            analyst_name = st.session_state["analyst_name"]

            with get_conn() as c:
                c.execute(
                    """INSERT INTO vurderinger(trans_id, kommentar, avklart, decision, checklist_json, decided_by, decided_at)
                         VALUES (?,?,?,?,?,?,?)
                         ON CONFLICT(trans_id) DO UPDATE SET
                           kommentar=excluded.kommentar,
                           avklart=excluded.avklart,
                           decision=excluded.decision,
                           checklist_json=excluded.checklist_json,
                           decided_by=excluded.decided_by,
                           decided_at=excluded.decided_at""",
                    (str(trans_id), kommentar, 1 if decision == "false_positive" else 0, decision,
                     json.dumps(checklist), analyst_name, now)
                )
                c.commit()

            cinfo = get_case_by_trans_id(trans_id)
            if not cinfo:
                create_case_for_trans(trans_id, priority=pri, owner=analyst_name)
                cinfo = get_case_by_trans_id(trans_id)

            if cinfo:
                if target_status == "closed":
                    if not checker.strip() or checker.strip().lower() == analyst_name.lower():
                        st.error("For å lukke saken kreves en checker (annen enn analytiker).")
                        st.stop()
                update_case(
                    cinfo["case_id"],
                    target_status,
                    cinfo.get("priority", "medium") or "medium",
                    cinfo.get("owner", "") or analyst_name,
                    cinfo.get("tags", "") or "",
                    cinfo.get("note", "") or "",
                    event_msg=f"Avklaring: decision={decision} → status={target_status}",
                )
                log_case_event(cinfo["case_id"], "decision", f"Decision={decision} av {analyst_name}")
                if target_status == "closed":
                    add_case_note(cinfo["case_id"],
                                  f"Maker-checker: Lukket av {analyst_name}, kontrollert av {checker.strip()}.",
                                  author=analyst_name)

            st.success("Avklaring lagret.")
            st.rerun()
# ---------- Tab 5: Vurderingshistorikk ----------
with tab_historikk:
    st.subheader("📊 Tidligere vurderinger")
    try:
        conn = get_conn()
        vurderinger_df = pd.read_sql_query("SELECT * FROM vurderinger", conn)
        conn.close()
        if vurderinger_df.empty:
            st.info("Ingen vurderinger er registrert ennå.")
        else:
            vurderinger_df["avklart"] = vurderinger_df["avklart"].astype(bool)
            vurderinger_df["status"] = vurderinger_df["avklart"].apply(lambda x: "✅ Avklart" if x else "❌ Ikke avklart")
            visning = vurderinger_df[["trans_id", "kommentar", "status"]]
            st.dataframe(visning, use_container_width=True)
            csv_data = vurderinger_df.drop(columns=["status"]).to_csv(index=False).encode("utf-8")
            st.download_button(label="📥 Last ned vurderinger som CSV", data=csv_data,
                               file_name="vurderingshistorikk.csv", mime="text/csv")
    except Exception as e:
        st.error(f"❌ Klarte ikke hente vurderingshistorikk: {e}")

# ---------- Tab 6: Last opp CSV ----------
with tab_upload:
    st.subheader("📂 Last opp ny CSV-fil med transaksjoner")
    uploaded = st.file_uploader("Velg en CSV-fil", type=["csv"], key="opplasting_csv")

    datoformat_input = st.text_input("Datoformat (valgfritt, f.eks. %d.%m.%Y %H:%M)", key="datoformat_opplasting")
    har_header = st.checkbox("Filen har kolonnenavn", value=True, key="header_opplasting")

    if uploaded is not None:
        tmp_path = "midlertidig_opplasting.csv"
        try:
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())

            datoformat = datoformat_input.strip() or None
            ny_df = trygg_les_csv(tmp_path, datoformat=datoformat, har_header=har_header)
            ny_df = analyser_transaksjoner(ny_df)
            ny_df = legg_til_anomalikluster(ny_df)
            ny_df = sjekk_mot_sanksjonsliste(ny_df, sanksjonsliste)

            eksisterende_ids = df["trans_id"].unique() if "trans_id" in df.columns else []
            nye_trans = ny_df[~ny_df["trans_id"].isin(eksisterende_ids)]

            if not nye_trans.empty:
                lagre_til_db(nye_trans)
                st.session_state.df = last_transaksjoner()
                st.session_state.df = beregn_risikoscore(st.session_state.df)

                mistenkelige = nye_trans[
                    _bool_series(nye_trans, "mistenkelig") |
                    _bool_series(nye_trans, "mistenkelig_ml") |
                    _bool_series(nye_trans, "sanksjonert") |
                    _bool_series(nye_trans, "fuzzy_sanksjonert")
                    ]
                if not mistenkelige.empty:
                    send_slack_varsel(mistenkelige)

                st.success(f"✅ {len(nye_trans)} nye transaksjoner analysert og lagret.")
                logg_hendelse("Opplasting CSV", antall=len(nye_trans))
            else:
                st.info("🟡 Ingen nye transaksjoner å lagre.")

            # (valgfritt) auto-registrer UBO hvis CSV har ubo_name
            if "ubo_name" in ny_df.columns and "fra_konto" in ny_df.columns:
                c = get_conn()
                existing = pd.read_sql_query("SELECT customer_id, name FROM kyc_ubos", c)
                pairs = set(
                    zip(existing.get("customer_id", []), existing.get("name", []))) if not existing.empty else set()
                new_ubos = 0
                for cid, uboname in ny_df[["fra_konto", "ubo_name"]].dropna().astype(str).drop_duplicates().itertuples(
                        index=False):
                    if (cid, uboname) in pairs:
                        continue
                    ubo_id = f"UBO{int(datetime.utcnow().timestamp() * 1000)}{np.random.randint(1000):04d}"
                    c.execute(
                        "INSERT INTO kyc_ubos (ubo_id, customer_id, name, role, created_at) VALUES (?,?,?,?,?)",
                        (ubo_id, cid, uboname, "UBO", datetime.utcnow().isoformat())
                    )
                    new_ubos += 1
                c.commit()
                if new_ubos:
                    st.info(f"Registrerte {new_ubos} nye UBO(er) fra CSV.")

            st.dataframe(ny_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Feil under opplasting og analyse: {e}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

# ---------- Tab 7: Kundetiltak ----------
with tab_kundetiltak:
    st.subheader("🧾 Kundetiltak og oppfølging")

    st.markdown("### ➕ Registrer tiltak")
    with st.form("kundetiltak_form"):
        kunde_ids = get_all_customer_ids()
        if kunde_ids:
            kunde_id = st.selectbox(
                "Kunde-ID",
                options=kunde_ids,
                index=None,
                placeholder="Velg / søk kunde-ID…",
                key="kundetiltak_kunde_id_select",
            )
        else:
            kunde_id = st.text_input("Kunde-ID", placeholder="Skriv inn kunde-ID", key="kundetiltak_kunde_id_text")
        risikonivå = st.selectbox("Risikonivå", ["Lav", "Medium", "Høy"])
        tiltakstype = st.selectbox("Tiltakstype", ["Standard kontroll", "Løpende overvåking", "Periodisk vurdering"])
        kommentar = st.text_area("Kommentar")
        send_inn = st.form_submit_button("💾 Lagre tiltak")

        if send_inn:
            if kunde_id:
                try:
                    conn = get_conn()
                    conn.execute(
                        "INSERT INTO kundetiltak (kunde_id, risikonivå, tiltakstype, kommentar, dato) VALUES (?, ?, ?, ?, DATE('now'))",
                        (kunde_id, risikonivå, tiltakstype, kommentar),
                    )
                    conn.commit()
                    conn.close()
                    st.success("✅ Tiltak lagret.")
                    logg_hendelse("Registrer tiltak", antall=1)
                except Exception as e:
                    st.error(f"❌ Klarte ikke lagre tiltak: {e}")
            else:
                st.warning("⚠️ Kunde-ID må fylles ut.")

    st.divider()
    st.markdown("### 📊 Registrerte tiltak")
    try:
        conn = get_conn()
        tiltak_df = pd.read_sql_query("SELECT * FROM kundetiltak", conn)
        conn.close()
        if tiltak_df.empty:
            st.info("Ingen tiltak registrert ennå.")
        else:
            st.dataframe(tiltak_df, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Feil ved visning: {e}")

# ---------- Tab 8: KYC/EDD ----------
with tab_kyc:
    st.subheader("🔐 KYC/EDD – periodiske og løpende kundetiltak")


    def _rerun_local():
        try:
            st.rerun()
        except Exception:
            try:
                st.rerun()
            except Exception:
                pass


    with st.sidebar:
        st.header("KYC/EDD handlinger")

        if st.button("Planlegg KYC nå"):
            with get_conn() as c:
                made = plan_reviews(c)
            st.success(f"Opprettet {made} nye reviews.")
            _rerun_local()

        if st.button("Re-screen kunder + UBO mot sanksjons-/PEP"):
            with get_conn() as c:
                res_c = rescreen_customers(c, sanksjonsliste)
                res_u = rescreen_ubos(c, sanksjonsliste)
            st.success(
                f"Re-screen ferdig: Kunder {res_c['matches']}/{res_c['checked']} treff • UBO {res_u['matches']}/{res_u['checked']} treff.")
            _rerun_local()

        if st.button("Varsle for forfall (7 dager)"):
            with get_conn() as c:
                sent = notify_upcoming_reviews(c, days=7)
            if sent:
                st.success(f"Sendte {sent} Slack-varsel/er for kommende/forfalte reviews.")
            else:
                st.info("Ingen kunder med forfall innen 7 dager, eller SLACK_WEBHOOK_URL ikke satt.")

    left, right = st.columns(2)

    with left:
        st.markdown("### Åpne reviews")

        with get_conn() as c:
            df_open = pd.read_sql_query("""
                SELECT r.review_id, r.customer_id, c.name AS customer_name, r.review_type, r.due_at,
                       c.risk_band, c.kyc_status
                FROM kyc_reviews r
                JOIN customers c ON c.customer_id = r.customer_id
                WHERE r.completed_at IS NULL
                ORDER BY date(r.due_at) ASC, r.review_id ASC
            """, c)

        if df_open.empty:
            st.info("Ingen åpne reviews for øyeblikket.")
            selected_rid = None
        else:
            st.dataframe(df_open, use_container_width=True)
            options = [
                f"{row.customer_name} ({row.customer_id}) – {row.review_type} – due {row.due_at}  [id:{row.review_id}]"
                for _, row in df_open.iterrows()
            ]
            picked = st.selectbox("Velg review for arbeid", options)
            selected_rid = None
            if picked:
                selected_rid = picked.split("[id:")[-1].rstrip("]")

        if selected_rid:
            st.markdown("### Sjekkliste")
            with get_conn() as c:
                tasks = pd.read_sql_query(
                    "SELECT task_id, task_type, title, status, created_at, completed_at FROM kyc_tasks WHERE review_id=? ORDER BY created_at",
                    c, params=(selected_rid,)
                )
            st.dataframe(tasks, use_container_width=True)

            with st.form("complete_task_form"):
                t_done = st.text_input("Skriv inn task_id for å fullføre")
                submitted = st.form_submit_button("✅ Marker oppgave som fullført")
                if submitted:
                    if t_done:
                        with get_conn() as c:
                            c.execute("UPDATE kyc_tasks SET status='done', completed_at=? WHERE task_id=?",
                                      (date.today().isoformat(), t_done))
                        st.success("Oppgave fullført.")
                        _rerun_local()
                    else:
                        st.warning("Task-ID mangler.")

            if st.button("▶️ Start review (sett started_at)"):
                with get_conn() as c:
                    c.execute(
                        "UPDATE kyc_reviews SET started_at=? WHERE review_id=? AND (started_at IS NULL OR started_at='')",
                        (datetime.utcnow().isoformat(), selected_rid)
                    )
                st.success("Review satt til 'startet'.")
                _rerun_local()

            if st.button("🏁 Fullfør review (maker–checker)"):
                with get_conn() as c:
                    pending = pd.read_sql_query(
                        "SELECT COUNT(*) AS c FROM kyc_tasks WHERE review_id=? AND status!='done'",
                        c, params=(selected_rid,)
                    )["c"].iloc[0]
                    if int(pending) == 0:
                        c.execute(
                            "UPDATE kyc_reviews SET completed_at=?, outcome='pass' WHERE review_id=?",
                            (date.today().isoformat(), selected_rid)
                        )
                        cust = pd.read_sql_query("SELECT customer_id FROM kyc_reviews WHERE review_id=?",
                                                 c, params=(selected_rid,)).iloc[0]["customer_id"]
                        c.execute("UPDATE customers SET kyc_status='clear', last_review_at=? WHERE customer_id=?",
                                  (date.today().isoformat(), cust))
                        st.success("Review fullført. Neste periodiske review planlegges automatisk ved 'Planlegg KYC'.")
                        _rerun_local()
                    else:
                        st.warning(f"{int(pending)} oppgave(r) gjenstår – kan ikke fullføre enda.")

            st.markdown("---")
            st.markdown("### PDF-eksport")
            if REPORTLAB_OK:
                kyc_pdf = export_kyc_review_pdf(selected_rid)
                if kyc_pdf:
                    st.download_button("📄 Last ned KYC-review (PDF)", data=kyc_pdf,
                                       file_name=f"kyc_review_{selected_rid}.pdf", mime="application/pdf")
                sar_pdf = export_sar_pdf(selected_rid)
                if sar_pdf:
                    st.download_button("🚨 Last ned SAR-kladd (PDF)", data=sar_pdf,
                                       file_name=f"sar_draft_{selected_rid}.pdf", mime="application/pdf")
            else:
                st.info("For PDF-eksport: `pip install reportlab`.")

        st.markdown("---")
        st.markdown("### ➕ Legg til UBO (Ultimate Beneficial Owner)")
        with st.form("add_ubo_form"):
            ubo_cid = st.text_input("Kunde-ID (tilhører)")
            ubo_name = st.text_input("UBO-navn")
            ubo_role = st.text_input("Rolle", value="UBO")
            ubo_country = st.text_input("Land (valgfritt)", value="")
            add_ubo = st.form_submit_button("➕ Legg til UBO")
            if add_ubo:
                if ubo_cid and ubo_name:
                    ubo_id = f"UBO{int(datetime.utcnow().timestamp() * 1000)}{np.random.randint(1000):04d}"
                    with get_conn() as c:
                        c.execute(
                            "INSERT INTO kyc_ubos (ubo_id, customer_id, name, country, role, created_at) VALUES (?,?,?,?,?,?)",
                            (ubo_id, ubo_cid, ubo_name, ubo_country, ubo_role, datetime.utcnow().isoformat())
                        )
                    st.success(f"UBO '{ubo_name}' lagt til for {ubo_cid}.")
                    _rerun_local()
                else:
                    st.warning("Fyll ut både Kunde-ID og UBO-navn.")

    with right:
        st.markdown("### Forfaller snart (30 dager)")
        with get_conn() as c:
            df_soon = pd.read_sql_query("""
                SELECT customer_id, name, next_review_at, risk_band, pep_flag, edd_required
                FROM customers
                WHERE next_review_at IS NOT NULL
                ORDER BY date(next_review_at) ASC
            """, c)
        if df_soon.empty:
            st.info("Ingen kunder nær forfall.")
        else:
            df_soon["days"] = (pd.to_datetime(df_soon["next_review_at"]) - pd.Timestamp.today().normalize()).dt.days
            st.dataframe(df_soon[df_soon["days"].between(0, 30)], use_container_width=True)

        st.markdown("### Dokumentstatus")
        with get_conn() as c:
            df_docs = pd.read_sql_query("""
                SELECT customer_id, doc_type, filename, status, expiry_date
                FROM kyc_documents
                ORDER BY date(expiry_date) ASC
            """, c)
        st.dataframe(df_docs, use_container_width=True)

        st.markdown("### UBO-oversikt")
        with get_conn() as c:
            df_ubos = pd.read_sql_query("""
                SELECT ubo_id, customer_id, name, role, country, created_at
                FROM kyc_ubos
                ORDER BY datetime(created_at) DESC
            """, c)
        if df_ubos.empty:
            st.info("Ingen UBO-er registrert ennå.")
        else:
            st.dataframe(df_ubos, use_container_width=True)

# ---------- Tab 9: Rapporter (MTA/ROS) ----------
with tab_rapporter:
    st.subheader("📑 Rapporter – MTA og ROS")

    df = _rydde_df(st.session_state.get("df", pd.DataFrame()))

    st.markdown("### MTA – Mistenkelige transaksjoner (aggregater)")
    if df.empty:
        st.info("Ingen transaksjoner i minnet.")
    else:
        flag = (
                _bool_series(df, "mistenkelig") |
                _bool_series(df, "mistenkelig_ml") |
                _bool_series(df, "sanksjonert") |
                _bool_series(df, "fuzzy_sanksjonert") |
                (pd.to_numeric(df.get("score", pd.Series(0, index=df.index)), errors="coerce").fillna(0) > 0.8)
        )
        mta_df = df[flag].copy()
        col1, col2, col3 = st.columns(3)
        col1.metric("Antall flaggede", len(mta_df))
        col2.metric("Andel av total", f"{(len(mta_df) / len(df) * 100):.1f}%" if len(df) else "0%")
        col3.metric("Snittbeløp (flagget)",
                    f"{mta_df['beløp'].mean():,.0f}" if "beløp" in mta_df.columns and not mta_df.empty else "–")

        if not mta_df.empty:
            st.markdown("**Topp-land (flagget)**")
            st.bar_chart(mta_df["land"].value_counts().head(10))
            if "til_konto" in mta_df.columns:
                st.markdown("**Topp mottakerkontoer (flagget)**")
                st.bar_chart(mta_df["til_konto"].value_counts().head(10))
        st.dataframe(mta_df, use_container_width=True)


        def export_mta_pdf() -> bytes | None:
            if not REPORTLAB_OK:
                return None
            c, buf, width, height = _pdf_start("MTA – Mistenkelige transaksjoner (aggregat)")
            y = 35
            c.setFont("Helvetica", 10)
            c.drawString(20 * mm, height - (y * mm),
                         f"Antall flaggede: {len(mta_df)} av {len(df)} ({(len(mta_df) / len(df) * 100):.1f}%)")
            y += 7
            for _, row in mta_df.head(40).iterrows():
                line = f"{row.get('trans_id', '')} | {row.get('tidspunkt', '')} | {row.get('beløp', '')} | {row.get('fra_konto', '')} → {row.get('til_konto', '')} | {row.get('land', '')} | {row.get('score', '')}"
                c.drawString(20 * mm, height - (y * mm), line[:110])
                y += 5
                if y > 270:
                    c.showPage();
                    y = 20
            c.showPage();
            c.save()
            pdf = buf.getvalue();
            buf.close()
            return pdf


        if REPORTLAB_OK:
            pdf = export_mta_pdf()
            if pdf:
                st.download_button("📄 Last ned MTA (PDF)", data=pdf, file_name="mta_report.pdf", mime="application/pdf")
        else:
            st.info("For MTA-PDF: `pip install reportlab`.")

    st.divider()

    st.markdown("### ROS – Risiko- og sårbarhetsanalyse (oversikt)")
    try:
        conn = get_conn()
        customers_df = pd.read_sql_query("SELECT * FROM customers", conn)
        conn.close()
        if customers_df.empty:
            st.info("Ingen kunder registrert.")
        else:
            band_count = customers_df["risk_band"].fillna("MEDIUM").value_counts()
            pep_count = int(customers_df.get("pep_flag", 0).sum())
            edd_count = int(customers_df.get("edd_required", 0).sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("Kunder totalt", len(customers_df))
            c2.metric("PEP flagget", pep_count)
            c3.metric("EDD påkrevd", edd_count)

            st.markdown("**Fordeling risk band**")
            st.bar_chart(band_count)

            st.dataframe(customers_df[["customer_id", "name", "risk_band", "kyc_status", "next_review_at", "pep_flag",
                                       "edd_required"]], use_container_width=True)

            if REPORTLAB_OK:
                ros_pdf = export_gap_pdf()
                if ros_pdf:
                    st.download_button("📄 Last ned ROS (PDF)", data=ros_pdf, file_name="ros_overview.pdf",
                                       mime="application/pdf")
            else:
                st.info("For ROS-PDF: `pip install reportlab`.")
    except Exception as e:
        st.error(f"ROS-feil: {e}")

# ---------- Tab 10: Bank-klar (sjekkliste & gap) ----------
with tab_bankklar:
    st.subheader("🏁 Bank-klar – sjekkliste & gap-analyse")

    df_chk = checklist_load()

    colA, colB = st.columns([1.6, 1])
    with colA:
        st.markdown("### ✅ Sjekkliste")
        if df_chk.empty:
            st.info("Ingen sjekklisteelementer enda.")
        else:
            domain = st.selectbox("Velg domene", ["Alle"] + sorted(df_chk["domain"].unique().tolist()))
            view = df_chk if domain == "Alle" else df_chk[df_chk["domain"] == domain]
            st.dataframe(view, use_container_width=True)

            st.markdown("#### Oppdater element")
            item_id = st.text_input("Item-ID (f.eks. CHK001)")
            new_status = st.selectbox("Status", ["", "open", "in_progress", "done"])
            new_owner = st.text_input("Owner (valgfritt)")
            new_notes = st.text_area("Notater (valgfritt)", height=80)
            if st.button("💾 Lagre endring"):
                if item_id.strip():
                    checklist_update(
                        item_id=item_id.strip(),
                        status=(new_status if new_status else None),
                        owner=(new_owner if new_owner else None),
                        notes=(new_notes if new_notes else None),
                    )
                    st.success("Oppdatert.")
                    try:
                        st.rerun()
                    except Exception:
                        pass
                else:
                    st.warning("Skriv inn Item-ID.")

            st.markdown("#### PDF-eksport")
            if REPORTLAB_OK:
                pdf1 = export_checklist_pdf()
                if pdf1:
                    st.download_button("📄 Last ned sjekkliste (PDF)", data=pdf1, file_name="sjekkliste.pdf",
                                       mime="application/pdf")
            else:
                st.info("For PDF-eksport: `pip install reportlab`.")

    with colB:
        st.markdown("### 🧭 Gap-analyse")
        summary = gap_summary(df_chk)
        if summary.empty:
            st.info("Ingen data å oppsummere ennå.")
        else:
            st.dataframe(summary, use_container_width=True)
            for _, r in summary.iterrows():
                st.markdown(f"**{r['domain']}** – {r['progress']}% ferdig")
                st.progress(min(100, int(r["progress"])) / 100)

            if REPORTLAB_OK:
                pdf2 = export_gap_pdf()
                if pdf2:
                    st.download_button("📄 Last ned gap-analyse (PDF)", data=pdf2, file_name="gap_analyse.pdf",
                                       mime="application/pdf")
            else:
                st.info("For PDF-eksport: `pip install reportlab`.")


with tab_alerts:
    st.subheader(" Alerts")

    # Status / filtre
    c1, c2, c3 = st.columns(3)
    with c1:
        alert_status = st.selectbox("Status", ["open", "in_case", "closed"], index=0, key="alerts_status")
    with c2:
        min_score = st.slider("Min score", 0.0, 1.0, 0.50, 0.01, key="alerts_min_score")
    with c3:
        customer_q = st.text_input("Filter customer_id", value="", key="alerts_customer_q")

    alerts_df = fetch_alerts(status=alert_status)

    if alerts_df.empty:
        st.info("Ingen alerts funnet for valgt status.")
    else:
        filt = alerts_df[alerts_df["risk_score"] >= float(min_score)].copy()
        if customer_q.strip():
            filt = filt[filt["customer_id"].astype(str).str.contains(customer_q.strip(), case=False, na=False)]

        # Sorter: høy score først
        try:
            filt = filt.sort_values(["risk_score", "alert_id"], ascending=[False, False])
        except Exception:
            pass

        st.dataframe(
            filt.drop(columns=["reasons_json"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )

        ids = filt["alert_id"].tolist()
        chosen_id = st.selectbox("Velg alert_id for detaljer", ids, key="alerts_selected_id")
        chosen_row = filt[filt["alert_id"] == chosen_id].iloc[0].to_dict()

        st.markdown("#### Detaljer")
        st.write({
            "alert_id": chosen_row.get("alert_id"),
            "transaction_id": chosen_row.get("transaction_id"),
            "customer_id": chosen_row.get("customer_id"),
            "risk_score": chosen_row.get("risk_score"),
            "risk_level": chosen_row.get("risk_level"),
            "created_at": chosen_row.get("created_at"),
            "status": chosen_row.get("status"),
        })

        with st.expander("Forklaring (reasons_json)"):
            rj = chosen_row.get("reasons_json") or ""
            try:
                st.json(json.loads(rj))
            except Exception:
                st.code(rj)

        # Vis transaksjonsdetaljer (hvis finnes)
        try:
            with get_conn() as conn:
                tx = pd.read_sql_query(
                    """
                    SELECT transaction_id, customer_id, amount, currency, timestamp, counterparty, country, description
                    FROM transactions
                    WHERE transaction_id = ?
                    """,
                    conn,
                    params=(str(chosen_row.get("transaction_id")),),
                )
            if not tx.empty:
                st.markdown("#### Transaksjon")
                st.dataframe(tx, use_container_width=True, hide_index=True)
        except Exception:
            pass

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("Opprett sak fra alert", type="primary", key="btn_create_case_from_alert"):
                prio = "high" if "Høy" in str(chosen_row.get("risk_level", "")) else "medium"
                # Bruk eksisterende case-helper i dashboard (robust mot ulike schema)
                case_id = create_case_for_trans(str(chosen_row.get("transaction_id")), priority=prio)
                if case_id:
                    update_alert_status(int(chosen_row["alert_id"]), "in_case")
                    st.success(f"Case opprettet: {case_id}")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Klarte ikke å opprette case.")
        with colB:
            if st.button("Marker som lukket", key="btn_close_alert"):
                update_alert_status(int(chosen_row["alert_id"]), "closed")
                st.cache_data.clear()
                st.rerun()

with tab_saker:
    st.subheader("🧾 Saker (case management)")

    # Sikre at vi har transaksjoner i minnet for detaljvisning (case -> transaksjon)
    df_all = _rydde_df(st.session_state.get("df", pd.DataFrame()))

    me = (st.session_state.get("analyst_name") or "").strip()


    def _rerun_local():
        try:
            st.rerun()
        except Exception:
            try:
                st.rerun()
            except Exception:
                pass


    # -----------------------------
    # Team queue (uassignede saker per tier)
    # -----------------------------
    role = (st.session_state.get("user_role") or "T1").strip().upper()
    if role in {"T1", "T2"}:
        st.markdown("### 👥 Team queue (uassignede saker)")
        st.caption("Viser åpne saker uten eier i din tier. Klikk **Ta saken** for å legge den i **Min kø**.")
        try:
            with get_conn() as c:
                team_df = pd.read_sql_query(
                    """
                    SELECT case_id, trans_id, status, priority, owner, COALESCE(tier,'T1') AS tier,
                           tags, note, created_at, updated_at
                    FROM cases
                    WHERE status IN ('open','in_review','escalated')
                      AND (owner IS NULL OR TRIM(owner) = '')
                      AND COALESCE(tier,'T1') = ?
                    ORDER BY datetime(updated_at) DESC
                    LIMIT 200
                    """,
                    c,
                    params=(role,),
                )
        except Exception:
            team_df = pd.DataFrame()

        if team_df.empty:
            st.info("Ingen uassignede saker i din team-kø akkurat nå.")
        else:
            for _, r in team_df.iterrows():
                c1, c2, c3, c4 = st.columns([2.2, 1.2, 1, 1])
                with c1:
                    st.markdown(f"**{r['case_id']}**  •  {r.get('status', '')}  •  {r.get('priority', '')}")
                    if r.get("tags"):
                        st.caption(f"Tags: {r.get('tags')}")
                    if r.get("note"):
                        st.caption(str(r.get("note"))[:140])
                with c2:
                    st.caption("Tier")
                    st.write(r.get("tier", "T1"))
                with c3:
                    st.caption("Transaksjon")
                    st.write(r.get("trans_id", ""))
                with c4:
                    if st.button("🫴 Ta saken", key=f"claim_case_{r['case_id']}"):
                        try:
                            now = datetime.utcnow().isoformat()
                            with get_conn() as c:
                                c.execute(
                                    "UPDATE cases SET owner=?, updated_at=? WHERE case_id=?",
                                    (me, now, r["case_id"]),
                                )
                                try:
                                    log_case_event(r["case_id"], "assign", f"Sak tildelt {me}")
                                except Exception:
                                    pass
                                c.commit()
                            st.success(f"Tildelt {r['case_id']} til {me}.")
                            _rerun_local()
                        except Exception as e:
                            st.error(f"Klarte ikke tildele: {e}")
        st.divider()
st.markdown("### 🧍 Min kø (med SLA)")

if not me:
    st.info("Skriv inn navn i sidepanelet for å bruke **Min kø**.")
else:
    open_status = ["open", "in_review", "escalated"]
    myq = fetch_cases_filtered(
        search="",
        status_list=open_status,
        priority_list=[],
        owner_contains="",
        mine_only_owner=me,
        limit=200,
        offset=0,
    )
    myq = _case_sla_enrich(myq)

    if myq.empty:
        st.info("Ingen åpne saker tildelt deg.")
    else:
        # KPI-er
        breached = int((myq["sla_status"] == "🔴 Brutt").sum())
        urgent = int((myq["sla_status"] == "🟠 Hastrer").sum())
        soon = int((myq["sla_status"] == "🟡 Snart").sum())
        total_open = len(myq)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Åpne saker", total_open)
        k2.metric("🔴 SLA brutt", breached)
        k3.metric("🟠 Haster", urgent)
        k4.metric("🟡 Snart", soon)

        # Sorter: mest kritisk først
        myq_sorted = myq.sort_values(["_sla_rank", "sla_remaining_hours", "updated_at"], ascending=[True, True, False])

        st.dataframe(
            myq_sorted[
                ["case_id", "trans_id", "status", "priority", "sla_status", "age_days", "sla_due_at", "updated_at"]]
            .rename(columns={"age_days": "age_days (d)"}),
            use_container_width=True
        )

    st.divider()
    st.markdown("### 🔎 Saksfilter (SQL-first)")

    # --- Live filter widgets ---
    # Important Streamlit rule:
    # Do not assign to st.session_state for a key after a widget with the same key
    # has been instantiated in the current run. Capture widget return values instead.
    c1, c2, c3, c4, c5 = st.columns([1.4, 1, 1, 1.1, 0.9])
    with c1:
        case_search_live = st.text_input(
            "Søk (case_id/trans_id/tags/note)",
            key="case_search_live",
            placeholder="f.eks. T0007, escalated, UBO…",
        )
    with c2:
        case_status_live = st.multiselect(
            "Status",
            options=CASE_STATUSES,
            default=st.session_state.get("case_status_live", ["open", "in_review", "escalated", "sar_draft"]),
            key="case_status_live",
        )
    with c3:
        case_priority_live = st.multiselect(
            "Prioritet",
            options=CASE_PRIORITIES,
            default=st.session_state.get("case_priority_live", ["low", "medium", "high"]),
            key="case_priority_live",
        )
    with c4:
        case_owner_live = st.text_input(
            "Owner (inneholder)",
            key="case_owner_live",
            placeholder="navn/initialer…",
        )
    with c5:
        case_only_mine_live = st.checkbox(
            "Kun mine",
            value=st.session_state.get("case_only_mine_live", False),
            key="case_only_mine_live",
        )

    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("✅ Bruk filter", key="apply_case_filters"):
            st.session_state["case_search_applied"] = case_search_live
            st.session_state["case_status_applied"] = case_status_live
            st.session_state["case_priority_applied"] = case_priority_live
            st.session_state["case_owner_applied"] = case_owner_live
            st.session_state["case_only_mine_applied"] = bool(case_only_mine_live)
            st.session_state["case_page"] = 0
            try:
                st.rerun()
            except Exception:
                pass
    with b2:
        if st.button("🧼 Nullstill", key="reset_case_filters"):
            # Only reset the applied filter state here. Do not write to widget keys
            # such as case_status_live/case_search_live after the widgets exist.
            for k, v in {
                "case_search_applied": "",
                "case_status_applied": ["open", "in_review", "escalated"],
                "case_priority_applied": ["low", "medium", "high"],
                "case_owner_applied": "",
                "case_only_mine_applied": False,
                "case_page": 0,
            }.items():
                st.session_state[k] = v
            try:
                st.rerun()
            except Exception:
                pass

    # --- Applied values ---
    search_applied = st.session_state.get("case_search_applied", "")
    status_applied = st.session_state.get("case_status_applied", ["open", "in_review", "escalated"])
    priority_applied = st.session_state.get("case_priority_applied", ["low", "medium", "high"])
    owner_applied = st.session_state.get("case_owner_applied", "")
    only_mine_applied = bool(st.session_state.get("case_only_mine_applied", False))
    mine_owner = me if (only_mine_applied and me) else ""

    st.caption(
        f"Aktive filter: søk='{search_applied or '-'}', "
        f"status={','.join(status_applied) if status_applied else '-'}, "
        f"prio={','.join(priority_applied) if priority_applied else '-'}, "
        f"owner~'{owner_applied or '-'}', "
        f"kun_mine={'ja' if mine_owner else 'nei'}"
    )

    # --- Paging ---
    page_size = st.selectbox("Rader per side", options=[50, 100, 200, 500], index=2, key="case_page_size")
    page = int(st.session_state.get("case_page", 0))
    offset = page * int(page_size)

    total = count_cases_filtered(
        search=search_applied,
        status_list=status_applied,
        priority_list=priority_applied,
        owner_contains=owner_applied,
        mine_only_owner=mine_owner,
    )

    total_pages = max(1, (total + int(page_size) - 1) // int(page_size))

    nav1, nav2, nav3 = st.columns([1, 1, 2])
    with nav1:
        if st.button("⬅️ Forrige", disabled=(page <= 0), key="case_prev"):
            st.session_state["case_page"] = max(0, page - 1)
            try:
                st.rerun()
            except Exception:
                pass
    with nav2:
        if st.button("➡️ Neste", disabled=(page >= total_pages - 1), key="case_next"):
            st.session_state["case_page"] = min(total_pages - 1, page + 1)
            try:
                st.rerun()
            except Exception:
                pass
    with nav3:
        st.write(f"Side {page + 1} / {total_pages} • Treff: {total}")

    cases_page = fetch_cases_filtered(
        search=search_applied,
        status_list=status_applied,
        priority_list=priority_applied,
        owner_contains=owner_applied,
        mine_only_owner=mine_owner,
        limit=int(page_size),
        offset=offset,
    )

    if cases_page.empty:
        st.info("Ingen saker matcher filteret.")
        selected_case_id = None
    else:
        st.dataframe(
            cases_page[["case_id", "trans_id", "status", "priority", "owner", "updated_at"]],
            use_container_width=True
        )
        options = [
            f"{r.case_id} | {r.trans_id} | {r.status}/{r.priority} | owner:{r.owner or '-'}"
            for _, r in cases_page.iterrows()
        ]
        picked = st.selectbox("Velg sak", options, key="case_pick")
        selected_case_id = picked.split("|")[0].strip() if picked else None

    st.divider()

    st.markdown("### 🛠️ Case view")

    if not selected_case_id:
        st.info("Velg en sak fra listen over.")
    else:
        with get_conn() as c:
            _case_df = pd.read_sql_query("SELECT * FROM cases WHERE case_id=?", c, params=(selected_case_id,))
        if _case_df.empty:
            st.warning("Fant ikke saken i databasen.")
        else:
            case_row = _case_df.iloc[0].to_dict()
            trans_id = str(case_row.get("trans_id", ""))

            # Hent matchende transaksjon fra minne (for explainability)
            tx_row = None
            if not df_all.empty and "trans_id" in df_all.columns and trans_id:
                hit = df_all[df_all["trans_id"].astype(str) == trans_id]
                if not hit.empty:
                    tx_row = hit.iloc[0].to_dict()

            disable_case_edit = (case_row.get("status") == "pending_approval")
            if disable_case_edit:
                st.warning("Denne saken er sendt til godkjenning og er låst for redigering (pending_approval).")

            topA, topB = st.columns([1.2, 1])
            with topA:
                st.write({
                    "case_id": case_row.get("case_id"),
                    "trans_id": trans_id,
                    "status": case_row.get("status"),
                    "priority": case_row.get("priority"),
                    "owner": case_row.get("owner"),
                    "tags": case_row.get("tags"),
                    "updated_at": case_row.get("updated_at"),
                })

                b1, b2 = st.columns(2)
                with b1:
                    if me and (case_row.get("owner") or "") != me:
                        if st.button("🙋 Ta eierskap", key=f"take_ownership_{selected_case_id}",
                                     disabled=disable_case_edit):
                            update_case(
                                case_id=case_row["case_id"],
                                status=case_row.get("status") or "open",
                                priority=case_row.get("priority") or "medium",
                                owner=me,
                                tags=case_row.get("tags") or "",
                                note=case_row.get("note") or "",
                                event_msg=f"Owner satt til {me}",
                            )
                            st.success("Eierskap oppdatert.")
                            st.rerun()

                with b2:
                    can_submit = (case_row.get("status") or "open") in {"open", "in_review", "escalated", "sar_draft"}
                    if st.button("🚦 Send til godkjenning", key=f"submit_for_approval_{selected_case_id}",
                                 disabled=(disable_case_edit or (not can_submit))):
                        submit_case_for_approval(
                            case_id=case_row["case_id"],
                            submitted_by=me or "analyst",
                            comment="",
                        )
                        st.success("Sak sendt til godkjenning.")
                        st.rerun()

            with topB:
                st.markdown("#### Oppdater sak")
                with st.form(key=f"case_update_form_{selected_case_id}"):
                    allowed_opts = allowed_next_status(case_row.get("status") or "open")
                    new_status = st.selectbox(
                        "Status (workflow)",
                        options=allowed_opts,
                        index=allowed_opts.index(case_row.get("status") or "open"),
                        disabled=disable_case_edit,
                    )
                    new_prio = st.selectbox(
                        "Prioritet",
                        options=["low", "medium", "high"],
                        index=["low", "medium", "high"].index(case_row.get("priority") or "medium"),
                        disabled=disable_case_edit,
                    )
                    new_owner = st.text_input("Owner", value=case_row.get("owner") or "", disabled=disable_case_edit)
                    new_tags = st.text_input("Tags", value=case_row.get("tags") or "", disabled=disable_case_edit)
                    new_note = st.text_area("Notat (felt på saken)", value=case_row.get("note") or "", height=90,
                                            disabled=disable_case_edit)

                    checker_name = ""
                    outcome = ""
                    closed_reason = ""
                    if new_status in ("reported", "closed"):
                        st.caption("🔒 Maker-checker: Krever ekstra godkjenning for 'reported'/'closed'.")
                        checker_name = st.text_input("Checker (annen person)", value="", disabled=disable_case_edit)
                        if new_status == "closed":
                            outcome = st.selectbox(
                                "Outcome",
                                options=["", "false_positive", "sar_filed", "monitoring", "other"],
                                index=0,
                                disabled=disable_case_edit,
                            )
                            closed_reason = st.text_area("Lukkingsgrunn (kort)", value="", height=70,
                                                         disabled=disable_case_edit)

                    ok = st.form_submit_button("💾 Lagre", disabled=disable_case_edit)
                if ok:
                    # Oppdater felt + ønsket status (med maker-checker hvis relevant)
                    update_case(
                        case_id=case_row["case_id"],
                        status=new_status,
                        priority=new_prio,
                        owner=new_owner,
                        tags=new_tags,
                        note=new_note,
                        checker=(checker_name or None),
                        outcome=(outcome or None),
                        closed_reason=(closed_reason or None),
                        event_msg=f"Oppdatert sak: status={case_row.get('status')}→{new_status}, prio={case_row.get('priority')}→{new_prio}",
                    )
                    st.success("Sak oppdatert.")
                    st.rerun()

            st.divider()
            st.markdown("### 📝 Case-notater")
            note_txt = st.text_area("Legg til notat", key=f"case_note_text_{selected_case_id}", height=90,
                                    disabled=disable_case_edit)
            if st.button("📝 Lagre notat", key=f"case_note_save_{selected_case_id}", disabled=disable_case_edit):
                if note_txt.strip():
                    add_case_note(selected_case_id, note_txt.strip(), actor=(me or "analyst"))
                    st.success("Notat lagret.")
                    st.rerun()
                else:
                    st.warning("Skriv et notat først.")

            st.divider()
            st.markdown("### 📜 Tidslinje")
            try:
                events_df = load_case_events(selected_case_id)
                render_case_timeline(events_df, key_prefix=f"timeline_{selected_case_id}")
            except Exception as e:
                st.info(f"Kunne ikke laste tidslinje: {e}")

with tab_godkjenning:
    st.subheader("🛂 Godkjenning (maker–checker)")

    checker = (st.session_state.get("analyst_name") or "").strip() or "checker"

    with get_conn() as c:
        pending = pd.read_sql_query(
            "SELECT case_id, trans_id, status, priority, owner, submitted_by, submitted_at, tags, note, updated_at "
            "FROM cases WHERE status='pending_approval' ORDER BY datetime(submitted_at) ASC",
            c
        )

    if pending.empty:
        st.info("Ingen saker venter på godkjenning.")
    else:
        st.markdown("### ⏳ Pending approvals")
        st.dataframe(pending, use_container_width=True)

        options = [
            f"{r.case_id} | trans:{r.trans_id} | pri:{r.priority} | owner:{r.owner or '-'} | by:{r.submitted_by or '-'} | at:{r.submitted_at or '-'}"
            for _, r in pending.iterrows()
        ]
        picked = st.selectbox("Velg sak for godkjenning", options, key="approval_pick")
        picked_id = picked.split("|")[0].strip() if picked else None

        if picked_id:
            with get_conn() as c:
                case_df = pd.read_sql_query("SELECT * FROM cases WHERE case_id=?", c, params=(picked_id,))
            if case_df.empty:
                st.warning("Fant ikke saken.")
            else:
                case_row = case_df.iloc[0].to_dict()

                st.markdown("### 🧾 Saksdetaljer")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Case", case_row.get("case_id"))
                c2.metric("Status", case_row.get("status"))
                c3.metric("Priority", case_row.get("priority"))
                c4.metric("Owner", case_row.get("owner") or "-")

                st.write({
                    "submitted_by": case_row.get("submitted_by"),
                    "submitted_at": case_row.get("submitted_at"),
                    "tags": case_row.get("tags"),
                })

                comment = st.text_area("Kommentar (påkrevd)", key="approval_comment")

                colA, colB = st.columns(2)
                with colA:
                    if st.button("✅ Godkjenn", key="approve_btn"):
                        if not comment.strip():
                            st.warning("Kommentar er påkrevd.")
                        else:
                            approve_case(picked_id, checker, comment.strip())
                            st.success("Sak godkjent.")
                            st.rerun()
                with colB:
                    if st.button("❌ Avvis", key="reject_btn"):
                        if not comment.strip():
                            st.warning("Kommentar er påkrevd.")
                        else:
                            reject_case(picked_id, checker, comment.strip())
                            st.error("Sak avvist.")
                            st.rerun()

                            st.divider()
            st.markdown("### 📜 Tidslinje")
            try:
                with get_conn() as c:
                    ev = pd.read_sql_query(
                        "SELECT event_id, case_id, event_type, message, actor, meta_json, created_at "
                        "FROM case_events WHERE case_id=? ORDER BY datetime(created_at) DESC",
                        c, params=(picked_id,)
                    )
                render_case_timeline(ev, key_prefix=f"approval_timeline_{picked_id}")
            except Exception:
                # fallback: raw
                with get_conn() as c:
                    ev = pd.read_sql_query(
                        "SELECT * FROM case_events WHERE case_id=? ORDER BY datetime(created_at) DESC",
                        c, params=(picked_id,)
                    )
                st.dataframe(ev, use_container_width=True)