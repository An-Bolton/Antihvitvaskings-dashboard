import streamlit as st
import pandas as pd
import sqlite3
import os
import json
import difflib
import urllib.request, urllib.error
from io import BytesIO
from datetime import datetime, date, timedelta
import numpy as np
import plotly.graph_objects as go

# External modules from your project
from db_handler import hent_transaksjoner, lagre_til_db
from risk_engine import analyser_transaksjoner
from ml_module import legg_til_anomalikluster
from utils import trygg_les_csv
from pep_checker import hent_sanksjonsdata, sjekk_mot_sanksjonsliste
from slack_notifier import send_slack_varsel  # transaction alerts

# Optional PDF deps
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# =============================
#   INIT / DB UTILITIES
# =============================

DB_PATH = "hvitvask.db"

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def ensure_tables():
    """Create tables used by the app if they don't exist, and seed checklist."""
    conn = get_conn()
    cur = conn.cursor()

    # Vurderinger (manuell behandling av transaksjoner)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vurderinger (
            trans_id TEXT PRIMARY KEY,
            kommentar TEXT,
            avklart INTEGER
        )
    """)

    # Enkle "kundetiltak"
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kundetiltak (
            kunde_id TEXT,
            risikonivå TEXT,
            tiltakstype TEXT,
            kommentar TEXT,
            dato TEXT
        )
    """)

    # Revisjonslogg
    cur.execute("""
        CREATE TABLE IF NOT EXISTS revisjonslogg (
            tidspunkt TEXT,
            handling TEXT,
            antall INTEGER
        )
    """)

    # KYC/EDD: kunder
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

    # KYC/EDD: reviews
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

    # KYC/EDD: tasks (checklists)
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

    # KYC/EDD: dokumenter
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

    # KYC/EDD: UBO-register
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kyc_ubos (
            ubo_id TEXT PRIMARY KEY,
            customer_id TEXT,
            name TEXT,
            dob TEXT,
            country TEXT,
            role TEXT,           -- e.g. 'UBO', 'Director'
            created_at TEXT
        )
    """)

    # Bank-klar sjekkliste
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

    conn.commit()

    # Seed sjekkliste hvis tom
    row = cur.execute("SELECT COUNT(*) FROM checklist").fetchone()
    if row and int(row[0]) == 0:
        seed_items = [
            # Functional
            ("Functional", "Regelbasert overvåking – prod-regler definert"),
            ("Functional", "ML-modell – treningsdata, validering, driftstrategi"),
            ("Functional", "Sanksjons/PEP-screening – full kjede m/ logging"),
            ("Functional", "KYC/KYB/EDD-checklist – maker-checker"),
            # Quality
            ("Quality", "Testdekning >70% kritisk logikk"),
            ("Quality", "CI: lint, typecheck, enhetstester"),
            ("Quality", "Datakvalitetsvarsler (skjevheter, mangler)"),
            # Security
            ("Security", "RBAC/least-privilege implementert"),
            ("Security", "Secrets i vault (ikke i kode/env)"),
            ("Security", "Kryptering i ro/overført – verifisert"),
            ("Security", "Sikker SDLC + pentest rapport"),
            # Operations
            ("Operations", "Observability: metrics, logs, alerter"),
            ("Operations", "Backup/restore og DR-test"),
            ("Operations", "Skalerings- og failover-strategi"),
            # Compliance
            ("Compliance", "ROS/DPIA med tiltaksliste"),
            ("Compliance", "Policy/prosedyrer + opplæring"),
            ("Compliance", "Internrevisjon/ekstern test planlagt"),
        ]
        now = datetime.utcnow().isoformat()
        for idx, (dom, txt) in enumerate(seed_items, start=1):
            cur.execute(
                "INSERT OR IGNORE INTO checklist (item_id, domain, item, owner, status, notes, updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (f"CHK{idx:03d}", dom, txt, "", "open", "", now)
            )
        conn.commit()

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
st.title("Anti-hvitvasking dashboard")
ensure_tables()

# =============================
#   DATA / PIPELINE HELPERS
# =============================

def simuler_transaksjoner(n=100, seed: int | None = 42):
    """Demo-generator. Sett seed=None for ekte random."""
    if seed is not None:
        np.random.seed(seed)
    transaksjoner = pd.DataFrame({
        "trans_id": [f"T{i+1:04d}" for i in range(n)],
        "fra_konto": np.random.choice(["NO9386011117947", "NO9386011117948", "NO9386011117949"], size=n),
        "til_konto": np.random.choice(["DE89370400440532013000", "FR7630006000011234567890189", "GB29NWBK60161331926819"], size=n),
        "beløp": np.round(np.random.uniform(1000, 100000, size=n), 2),
        "land": np.random.choice(["Norge", "Tyskland", "Frankrike", "UK", "Kina", "USA", "IR", "KP", "SY", "RU"], size=n),
        "tidspunkt": pd.date_range(end=pd.Timestamp.today(), periods=n).to_list(),
    })
    scores = np.concatenate([
        np.random.uniform(0.0, 0.5, n // 3),
        np.random.uniform(0.5, 0.8, n // 3),
        np.random.uniform(0.8, 1.0, n - 2*(n // 3))
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
            np.random.uniform(0.8, 1.0, n - 2*(n // 3)),
        ])
        np.random.shuffle(s)
        df["score"] = s

    if "risikonivå" not in df.columns:
        df["risikonivå"] = df["score"].apply(
            lambda x: "🔺 Høy" if x > 0.8 else ("⚡ Medium" if x > 0.5 else "✅ Lav")
        )
    return df

def beregn_risikoscore(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    score = np.zeros(len(df), dtype=float)

    if "beløp" in df.columns:
        bel = pd.to_numeric(df["beløp"], errors="coerce").fillna(0.0)
        max_beløp = bel.max()
        if max_beløp > 0:
            score += bel / max_beløp

    høyrisikoland = {"IR", "KP", "SY", "RU"}
    if "land" in df.columns:
        score += df["land"].isin(høyrisikoland).astype(float) * 0.5

    if "tidspunkt" in df.columns:
        tids = pd.to_datetime(df["tidspunkt"], errors="coerce")
        natt = tids.dt.hour.between(0, 5)
        score += natt.fillna(False).astype(float) * 0.3

    if "sanksjonert" in df.columns:
        score += df["sanksjonert"].fillna(False).astype(float) * 0.5
    if "fuzzy_sanksjonert" in df.columns:
        score += df["fuzzy_sanksjonert"].fillna(False).astype(float) * 0.4

    df["score"] = np.clip(score, 0, 1)
    df["risikonivå"] = df["score"].apply(
        lambda x: "🔺 Høy" if x > 0.8 else ("⚡ Medium" if x > 0.5 else "✅ Lav")
    )
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
    return (start + timedelta(days=months*30)).date().isoformat()  # enkel mnd-approx

def checklist_for(customer_type: str, pep: bool, edd: bool, risk_band: str):
    key = (customer_type or "PERSON", "ENHANCED" if (pep or edd or (risk_band or "MEDIUM").upper() in {"HIGH","CRITICAL"}) else "STANDARD")
    return CHECKLIST.get(key, CHECKLIST[("PERSON","STANDARD")])

def create_review(conn, customer_id: str, review_type: str, due_at_iso: str,
                  customer_type: str, pep: bool, edd: bool, risk_band: str):
    rid = f"R{int(datetime.utcnow().timestamp()*1000)}{np.random.randint(1000):04d}"
    conn.execute("INSERT INTO kyc_reviews(review_id, customer_id, review_type, due_at) VALUES (?,?,?,?)",
                 (rid, customer_id, review_type, due_at_iso))
    for task_type, title in checklist_for(customer_type, pep, edd, risk_band):
        tid = f"TASK{int(datetime.utcnow().timestamp()*1000)}{np.random.randint(1000):04d}"
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
            create_review(conn, cid, "event", today, c.get("type") or "PERSON", True, True, c.get("risk_band") or "HIGH")
            slack_send(f"🚨 Re-screen UBO: Treff på {uname} (UBO for {c.get('name')} / {cid}) – EDD/event review opprettet.")
    conn.commit()
    return {"matches": hits, "checked": len(df_u)}

def notify_upcoming_reviews(conn, days:int=7):
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
        for tbl in ["transaksjoner", "customers", "kyc_reviews", "kyc_tasks", "kyc_documents", "kyc_ubos", "vurderinger", "kundetiltak"]:
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
                ubo_id = f"UBO{int(datetime.utcnow().timestamp()*1000)}{np.random.randint(1000):04d}"
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
    c.drawString(20*mm, height-20*mm, title)
    c.setFont("Helvetica", 9)
    c.drawString(20*mm, height-26*mm, f"Generert: {datetime.utcnow().isoformat()}Z")
    c.line(20*mm, height-28*mm, width-20*mm, height-28*mm)
    return c, buf, width, height

def _pdf_multiline(c, x_mm: float, y_start_mm: float, text: str, line_height: float=5):
    width, height = A4
    y = y_start_mm
    for line in text.splitlines():
        c.drawString(x_mm*mm, (height - y*mm), line)
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
        f"Type: {row.get('type','')}, Risiko: {row.get('risk_band','')}, Status: {row.get('kyc_status','')}",
        f"Review-type: {row.get('review_type','')}",
        f"Forfallsdato: {row.get('due_at','')}",
        f"Startet: {row.get('started_at','') or '-'}  Fullført: {row.get('completed_at','') or '-'}  UtfalI: {row.get('outcome','') or '-'}",
        f"PEP: {'Ja' if int(row.get('pep_flag',0)) else 'Nei'}   EDD: {'Ja' if int(row.get('edd_required',0)) else 'Nei'}",
    ]
    y = _pdf_multiline(c, 20, y+6, "\n".join(summary))

    y += 6
    c.setFont("Helvetica-Bold", 11)
    y = _pdf_multiline(c, 20, y, "Sjekkliste")
    c.setFont("Helvetica", 10)
    if tasks.empty:
        y = _pdf_multiline(c, 20, y+6, "- (ingen oppgaver)")
    else:
        for _, t in tasks.iterrows():
            line = f"[{'x' if t['status']=='done' else ' '}] {t['title']}  ({t['task_type']})"
            y = _pdf_multiline(c, 25, y+5, line)

    y += 8
    c.setFont("Helvetica-Bold", 11)
    y = _pdf_multiline(c, 20, y, "Funn (JSON)")
    c.setFont("Helvetica", 9)
    findings = row.get("findings_json") or "{}"
    y = _pdf_multiline(c, 20, y+6, findings)

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
    y = _pdf_multiline(c, 20, y+6, f"Kunde: {row['name']} ({cust_id})  | Risk: {row.get('risk_band','')}")
    y = _pdf_multiline(c, 20, y+5, f"Review: {row['review_id']} ({row.get('review_type','')})  Due: {row.get('due_at','')}")

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
    y = _pdf_multiline(c, 20, y+6, kladd)

    y += 8
    c.setFont("Helvetica-Bold", 11)
    y = _pdf_multiline(c, 20, y, f"Transaksjoner (siste 30 dager) – {len(tx)} stk")
    c.setFont("Helvetica", 9)
    if tx.empty:
        y = _pdf_multiline(c, 20, y+6, "(Ingen funnet for filteret.)")
    else:
        n = 0
        for _, t in tx.head(35).iterrows():
            n += 1
            line = f"{n:02d}. {t.get('tidspunkt','')} | {t.get('beløp','')} | {t.get('fra_konto','')} → {t.get('til_konto','')} | {t.get('land','')}"
            y = _pdf_multiline(c, 20, y+5, line)
            if y > 260:
                c.showPage(); y = 20

    y += 8
    c.setFont("Helvetica-Bold", 11)
    y = _pdf_multiline(c, 20, y, "Sjekkliste-status")
    c.setFont("Helvetica", 10)
    if tasks.empty:
        y = _pdf_multiline(c, 20, y+6, "- (ingen oppgaver)")
    else:
        done = int((tasks["status"]=="done").sum()); total = len(tasks)
        y = _pdf_multiline(c, 20, y+6, f"Ferdigstilt: {done}/{total}")

    c.showPage(); c.save()
    pdf = buf.getvalue(); buf.close()
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
    sets.append("updated_at=?"); params.append(datetime.utcnow().isoformat())
    if not sets: return
    with get_conn() as c:
        c.execute(f"UPDATE checklist SET {', '.join(sets)} WHERE item_id=?", (*params, item_id))

def gap_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["domain","open","in_progress","done","total","progress"])
    g = df.groupby(["domain","status"]).size().unstack(fill_value=0)
    g["total"] = g.sum(axis=1)
    g["done"] = g.get("done", 0)
    g["open"] = g.get("open", 0)
    g["in_progress"] = g.get("in_progress", 0)
    g["progress"] = (g["done"] / g["total"] * 100).round(1)
    return g.reset_index()[["domain","open","in_progress","done","total","progress"]]

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
        y = _pdf_multiline(c, 25, y+5, line[:110])
        if y > 270: c.showPage(); y = 20
    c.showPage(); c.save()
    pdf = buf.getvalue(); buf.close()
    return pdf

def export_gap_pdf() -> bytes | None:
    if not REPORTLAB_OK:
        return None
    df = checklist_load()
    s = gap_summary(df)
    c, buf, width, height = _pdf_start("Gap-analyse – oversikt")
    y = 35; c.setFont("Helvetica", 10)
    for _, r in s.iterrows():
        line = f"{r['domain']:<12} | total: {int(r['total'])}  done: {int(r['done'])}  in_prog: {int(r['in_progress'])}  open: {int(r['open'])}  progress: {r['progress']}%"
        y = _pdf_multiline(c, 20, y, line)
        if y > 270: c.showPage(); y = 20
    c.showPage(); c.save()
    pdf = buf.getvalue(); buf.close()
    return pdf

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
    return df

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
#   TABS
# =============================

faner = st.tabs([
    " Alle transaksjoner",
    " Mistenkelige",
    " Statistikk & grafer",
    "️ Sanksjonsliste",
    " Avklaring",
    " Historikk",
    " Last opp CSV",
    " Kundetiltak",
    " KYC/EDD",
    " Rapporter (MTA/ROS)",
    " Bank-klar (sjekkliste & gap)",
    " Om plattformen"
])

# ---------- Tab 0: Alle transaksjoner ----------
with faner[0]:
    st.subheader("📋 Alle transaksjoner")

    def _rerun():
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
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
    datoformat_input = st.text_input("Datoformat (f.eks. %d.%m.%Y %H:%M)", value="")
    har_header = st.checkbox("Filen har kolonnenavn", value=True)

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
                        (nye_trans.get("mistenkelig", False)) |
                        (nye_trans.get("mistenkelig_ml", False)) |
                        (nye_trans.get("sanksjonert", False)) |
                        (nye_trans.get("fuzzy_sanksjonert", False))
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

        konto_filter = st.text_input("Filtrer på fra_konto (søk)")
        land_filter = st.multiselect("Filtrer på land", options=sorted(df["land"].dropna().unique()) if "land" in df.columns else [])
        risikonivå_filter = st.multiselect("Filtrer på risikonivå", options=sorted(df["risikonivå"].dropna().unique()))

        dato_start, dato_slutt = None, None
        if "tidspunkt" in df.columns and df["tidspunkt"].notna().any():
            min_dato = pd.to_datetime(df["tidspunkt"]).min().date()
            max_dato = pd.to_datetime(df["tidspunkt"]).max().date()
            dato_interval = st.date_input("Velg datointervall", [min_dato, max_dato], min_value=min_dato, max_value=max_dato)
            if isinstance(dato_interval, list) and len(dato_interval) == 2:
                dato_start, dato_slutt = dato_interval

        filtered_df = df.copy()
        if konto_filter and "fra_konto" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["fra_konto"].astype(str).str.contains(konto_filter, na=False, case=False)]
        if land_filter and "land" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["land"].isin(land_filter)]
        if risikonivå_filter:
            filtered_df = filtered_df[filtered_df["risikonivå"].isin(risikonivå_filter)]
        if dato_start and dato_slutt and "tidspunkt" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["tidspunkt"].notna()]
            filtered_df = filtered_df[
                (pd.to_datetime(filtered_df["tidspunkt"]).dt.date >= dato_start) &
                (pd.to_datetime(filtered_df["tidspunkt"]).dt.date <= dato_slutt)
            ]

        st.markdown("### 🧾 Transaksjonsoversikt")
        vis_kolonner = ["trans_id", "fra_konto", "til_konto", "beløp", "land", "risikonivå", "tidspunkt"]
        eksisterende = [k for k in vis_kolonner if k in filtered_df.columns]
        st.dataframe(filtered_df[eksisterende], use_container_width=True)

# ---------- Tab 1: Mistenkelige ----------
with faner[1]:
    st.subheader("⚠️ Mistenkelige transaksjoner")

    df = _rydde_df(st.session_state.get("df", pd.DataFrame()))
    if df.empty:
        st.info("Ingen transaksjoner i minnet.")
    else:
        mistenkelig_col = df["mistenkelig"] if "mistenkelig" in df.columns else pd.Series([False] * len(df))
        ml_col = df["mistenkelig_ml"] if "mistenkelig_ml" in df.columns else pd.Series([False] * len(df))
        sanksjonert_col = df["sanksjonert"] if "sanksjonert" in df.columns else pd.Series([False] * len(df))
        fuzzy_col = df["fuzzy_sanksjonert"] if "fuzzy_sanksjonert" in df.columns else pd.Series([False] * len(df))

        mistenkelige_df = df[
            mistenkelig_col.fillna(False) |
            ml_col.fillna(False) |
            sanksjonert_col.fillna(False) |
            fuzzy_col.fillna(False)
        ].copy()

        if mistenkelige_df.empty:
            st.info("Ingen mistenkelige transaksjoner funnet.")
        else:
            st.markdown("### 📋 Oversikt over flaggede transaksjoner")
            vis_kolonner = ["trans_id", "fra_konto", "til_konto", "beløp", "land", "risikonivå"]
            if "fuzzy_sanksjonert" in mistenkelige_df.columns:
                vis_kolonner.append("fuzzy_sanksjonert")
            st.dataframe(mistenkelige_df[[k for k in vis_kolonner if k in mistenkelige_df.columns]], use_container_width=True)

            csv = mistenkelige_df.to_csv(index=False).encode("utf-8")
            st.download_button("📤 Last ned som CSV", data=csv, file_name="mistenkelige_transaksjoner.csv", mime="text/csv")

# ---------- Tab 2: Statistikk & grafer ----------
with faner[2]:
    st.subheader("📈 Statistikk og visuelle innsikter")

    df = _rydde_df(st.session_state.get("df", pd.DataFrame()))
    if df.empty:
        st.info("Ingen data tilgjengelig for statistikk.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("🔢 Totalt antall transaksjoner", len(df))
        col2.metric("⚠️ Flagget (regler)", int(df["mistenkelig"].sum()) if "mistenkelig" in df.columns else 0)
        col3.metric("🤖 Flagget (ML)", int(df["mistenkelig_ml"].sum()) if "mistenkelig_ml" in df.columns else 0)

        st.divider()
        if "land" in df.columns:
            st.markdown("### 🌍 Antall transaksjoner per land")
            st.bar_chart(df["land"].value_counts())

        if "tidspunkt" in df.columns and "beløp" in df.columns:
            st.markdown("### 📈 Beløpsutvikling over tid")
            df_sorted = df.sort_values("tidspunkt")
            st.line_chart(df_sorted.set_index("tidspunkt")["beløp"].tail(100))

        if "score" in df.columns and "land" in df.columns:
            st.markdown("### 🧠 Risiko per land (gj.snittlig score)")
            risiko_per_land = df.groupby("land")["score"].mean().sort_values(ascending=False)
            st.bar_chart(risiko_per_land)

        if "risikonivå" in df.columns:
            st.markdown("### 🚨 Fordeling av risikonivå")
            st.bar_chart(df["risikonivå"].value_counts())

        sankey_transaksjoner(df)

# ---------- Tab 3: PEP/Sanksjonsliste ----------
with faner[3]:
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
with faner[4]:
    st.subheader("✅ Manuell vurdering og avklaring")

    df = _rydde_df(st.session_state.get("df", pd.DataFrame()))
    if df.empty:
        st.warning("⚠️ Ingen transaksjoner å vurdere.")
    else:
        mistenkelig_filter = df["mistenkelig"].fillna(False).astype(bool) if "mistenkelig" in df.columns else pd.Series(False, index=df.index)
        ml_filter = df["mistenkelig_ml"].fillna(False).astype(bool) if "mistenkelig_ml" in df.columns else pd.Series(False, index=df.index)
        kombinasjon = mistenkelig_filter | ml_filter
        vurder_df = df.loc[kombinasjon].copy()

        vurderte = []
        if not vurder_df.empty:
            for _, row in vurder_df.iterrows():
                trans_id = row.get("trans_id", "ukjent")
                beløp = row.get("beløp", "ukjent")
                fra = row.get("fra_konto", "ukjent")
                til = row.get("til_konto", "ukjent")
                st.markdown(f"**Transaksjon {trans_id} – {beløp} kr fra {fra} til {til}**")

                kommentar = st.text_area("Kommentar", key=f"kommentar_{trans_id}")
                avklart = st.checkbox("Avklart", key=f"avklart_{trans_id}")

                vurderte.append({"trans_id": trans_id, "kommentar": kommentar, "avklart": bool(avklart)})

        if st.button("💾 Lagre vurderinger"):
            try:
                conn = get_conn()
                cur = conn.cursor()
                for v in vurderte:
                    cur.execute(
                        """
                        INSERT INTO vurderinger (trans_id, kommentar, avklart)
                        VALUES (?, ?, ?)
                        ON CONFLICT(trans_id) DO UPDATE SET
                          kommentar=excluded.kommentar,
                          avklart=excluded.avklart
                        """,
                        (v["trans_id"], v["kommentar"], int(v["avklart"])),
                    )
                conn.commit()
                conn.close()
                st.success("✅ Vurderinger lagret")
                logg_hendelse("Lagre vurderinger", antall=len(vurderte))
            except Exception as e:
                st.error(f"❌ Feil under lagring: {e}")

# ---------- Tab 5: Vurderingshistorikk ----------
with faner[5]:
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
            st.download_button(label="📥 Last ned vurderinger som CSV", data=csv_data, file_name="vurderingshistorikk.csv", mime="text/csv")
    except Exception as e:
        st.error(f"❌ Klarte ikke hente vurderingshistorikk: {e}")

# ---------- Tab 6: Last opp CSV ----------
with faner[6]:
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
                    (nye_trans.get("mistenkelig", False)) |
                    (nye_trans.get("mistenkelig_ml", False)) |
                    (nye_trans.get("sanksjonert", False)) |
                    (nye_trans.get("fuzzy_sanksjonert", False))
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
                pairs = set(zip(existing.get("customer_id", []), existing.get("name", []))) if not existing.empty else set()
                new_ubos = 0
                for cid, uboname in ny_df[["fra_konto", "ubo_name"]].dropna().astype(str).drop_duplicates().itertuples(index=False):
                    if (cid, uboname) in pairs:
                        continue
                    ubo_id = f"UBO{int(datetime.utcnow().timestamp()*1000)}{np.random.randint(1000):04d}"
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
with faner[7]:
    st.subheader("🧾 Kundetiltak og oppfølging")

    st.markdown("### ➕ Registrer tiltak")
    with st.form("kundetiltak_form"):
        kunde_id = st.text_input("Kunde-ID")
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
with faner[8]:
    st.subheader("🔐 KYC/EDD – periodiske og løpende kundetiltak")

    def _st_rerun():
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass

    with st.sidebar:
        st.header("KYC/EDD handlinger")

        if st.button("Planlegg KYC nå"):
            with get_conn() as c:
                made = plan_reviews(c)
            st.success(f"Opprettet {made} nye reviews.")
            _st_rerun()

        if st.button("Re-screen kunder + UBO mot sanksjons-/PEP"):
            with get_conn() as c:
                res_c = rescreen_customers(c, sanksjonsliste)
                res_u = rescreen_ubos(c, sanksjonsliste)
            st.success(f"Re-screen ferdig: Kunder {res_c['matches']}/{res_c['checked']} treff • UBO {res_u['matches']}/{res_u['checked']} treff.")
            _st_rerun()

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
                        _st_rerun()
                    else:
                        st.warning("Task-ID mangler.")

            if st.button("▶️ Start review (sett started_at)"):
                with get_conn() as c:
                    c.execute(
                        "UPDATE kyc_reviews SET started_at=? WHERE review_id=? AND (started_at IS NULL OR started_at='')",
                        (datetime.utcnow().isoformat(), selected_rid)
                    )
                st.success("Review satt til 'startet'.")
                _st_rerun()

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
                        _st_rerun()
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
        st.markdown("### ➕ Legg til UBO")
        with st.form("add_ubo_form"):
            ubo_cid = st.text_input("Kunde-ID (tilhører)")
            ubo_name = st.text_input("UBO-navn")
            ubo_role = st.text_input("Rolle", value="UBO")
            ubo_country = st.text_input("Land (valgfritt)", value="")
            add_ubo = st.form_submit_button("➕ Legg til UBO")
            if add_ubo:
                if ubo_cid and ubo_name:
                    ubo_id = f"UBO{int(datetime.utcnow().timestamp()*1000)}{np.random.randint(1000):04d}"
                    with get_conn() as c:
                        c.execute(
                            "INSERT INTO kyc_ubos (ubo_id, customer_id, name, country, role, created_at) VALUES (?,?,?,?,?,?)",
                            (ubo_id, ubo_cid, ubo_name, ubo_country, ubo_role, datetime.utcnow().isoformat())
                        )
                    st.success(f"UBO '{ubo_name}' lagt til for {ubo_cid}.")
                    _st_rerun()
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
            st.dataframe(df_soon[df_soon["days"].between(0,30)], use_container_width=True)

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
with faner[9]:
    st.subheader("📑 Rapporter – MTA og ROS")

    df = _rydde_df(st.session_state.get("df", pd.DataFrame()))

    st.markdown("### MTA – Mistenkelige transaksjoner (aggregater)")
    if df.empty:
        st.info("Ingen transaksjoner i minnet.")
    else:
        flag = (
            (df.get("mistenkelig", pd.Series(False, index=df.index)).fillna(False)) |
            (df.get("mistenkelig_ml", pd.Series(False, index=df.index)).fillna(False)) |
            (df.get("sanksjonert", pd.Series(False, index=df.index)).fillna(False)) |
            (df.get("fuzzy_sanksjonert", pd.Series(False, index=df.index)).fillna(False)) |
            (df.get("score", pd.Series(0, index=df.index)) > 0.8)
        )
        mta_df = df[flag].copy()
        col1, col2, col3 = st.columns(3)
        col1.metric("Antall flaggede", len(mta_df))
        col2.metric("Andel av total", f"{(len(mta_df)/len(df)*100):.1f}%" if len(df) else "0%")
        col3.metric("Snittbeløp (flagget)", f"{mta_df['beløp'].mean():,.0f}" if "beløp" in mta_df.columns and not mta_df.empty else "–")

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
            c.drawString(20*mm, height-(y*mm), f"Antall flaggede: {len(mta_df)} av {len(df)} ({(len(mta_df)/len(df)*100):.1f}%)")
            y += 7
            for _, row in mta_df.head(40).iterrows():
                line = f"{row.get('trans_id','')} | {row.get('tidspunkt','')} | {row.get('beløp','')} | {row.get('fra_konto','')} → {row.get('til_konto','')} | {row.get('land','')} | {row.get('score','')}"
                c.drawString(20*mm, height-(y*mm), line[:110])
                y += 5
                if y > 270:
                    c.showPage(); y = 20
            c.showPage(); c.save()
            pdf = buf.getvalue(); buf.close()
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

            st.dataframe(customers_df[["customer_id","name","risk_band","kyc_status","next_review_at","pep_flag","edd_required"]], use_container_width=True)

            if REPORTLAB_OK:
                ros_pdf = export_gap_pdf()
                if ros_pdf:
                    st.download_button("📄 Last ned ROS (PDF)", data=ros_pdf, file_name="ros_overview.pdf", mime="application/pdf")
            else:
                st.info("For ROS-PDF: `pip install reportlab`.")
    except Exception as e:
        st.error(f"ROS-feil: {e}")

# ---------- Tab 10: Bank-klar (sjekkliste & gap) ----------
with faner[10]:
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
                    st.download_button("📄 Last ned sjekkliste (PDF)", data=pdf1, file_name="sjekkliste.pdf", mime="application/pdf")
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
                    st.download_button("📄 Last ned gap-analyse (PDF)", data=pdf2, file_name="gap_analyse.pdf", mime="application/pdf")
            else:
                st.info("For PDF-eksport: `pip install reportlab`.")

        with faner[11]:
            st.subheader("Om plattformen")

            st.markdown("""
            Velkommen til **Anti-hvitvasking Dashboardet** 👋  

            Dette systemet er utviklet som et *proof-of-concept* for sanntidsovervåking av:
            -  Transaksjoner  
            - ️Sanksjons- og PEP-sjekk  
            -  KYC/EDD-prosesser  
            -  Risikorapportering og revisjon 

            ---
            **Formål:**  
            Det er for å vise hvordan teknologi kan støtte banker, fintechs og finansmyndigheter i jobben mot hvitvasking og annen ulovlig finansering.

            **Plattformen er bygget med:**  
            - Python  
            - Streamlit 
            - Pandas / NumPy  
            - Plotly  
            - SQLite  

            ---
           Bygget av Andreas Bolton Seielstad
            
            ---
            ** Kontakt:**  
            For spørsmål eller forslag, ta kontakt via GitHub:  
            [👉 GitHub-repoet mitt:](https://github.com/An-Bolton/Antihvitvaskings-dashboard)
            """)
