# db_handler.py
import os
import sqlite3
from pathlib import Path

import pandas as pd

# IMPORTANT: Point both FastAPI + Streamlit to the same SQLite file.
# Use AML_DB_PATH to override. If not set, default to <project_root>/transaksjoner.db.
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.getenv("AML_DB_PATH", str(BASE_DIR / "transaksjoner.db"))


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    """Opprett tabell + legg til kolonner hvis de mangler (migrering light)."""
    with get_conn() as conn:
        cur = conn.cursor()

        # 1) Base-tabell (minimum)
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS transaksjoner (
            trans_id TEXT PRIMARY KEY,
            fra_konto TEXT,
            til_konto TEXT,
            beløp REAL,
            tidspunkt TEXT,
            land TEXT,

            score REAL,
            risikonivå TEXT,
            reasons_json TEXT,

            mistenkelig INTEGER DEFAULT 0,
            mistenkelig_ml INTEGER DEFAULT 0,
            sanksjonert INTEGER DEFAULT 0,
            fuzzy_sanksjonert INTEGER DEFAULT 0,
            anomaly_score REAL
        )
        """
        )

        # 2) "Migrations light": prøv å legge til kolonner om de mangler
        alter_stmts = [
            "ALTER TABLE transaksjoner ADD COLUMN risikonivå TEXT",
            "ALTER TABLE transaksjoner ADD COLUMN reasons_json TEXT",
            "ALTER TABLE transaksjoner ADD COLUMN mistenkelig_ml INTEGER DEFAULT 0",
            "ALTER TABLE transaksjoner ADD COLUMN sanksjonert INTEGER DEFAULT 0",
            "ALTER TABLE transaksjoner ADD COLUMN fuzzy_sanksjonert INTEGER DEFAULT 0",
            "ALTER TABLE transaksjoner ADD COLUMN anomaly_score REAL",
            "ALTER TABLE transaksjoner ADD COLUMN score REAL",
            "ALTER TABLE transaksjoner ADD COLUMN land TEXT",
        ]
        for stmt in alter_stmts:
            try:
                cur.execute(stmt)
            except sqlite3.OperationalError:
                pass

        # 3) Indekser for ytelse
        idx = [
            "CREATE INDEX IF NOT EXISTS idx_tx_tidspunkt ON transaksjoner(tidspunkt)",
            "CREATE INDEX IF NOT EXISTS idx_tx_fra ON transaksjoner(fra_konto)",
            "CREATE INDEX IF NOT EXISTS idx_tx_til ON transaksjoner(til_konto)",
            "CREATE INDEX IF NOT EXISTS idx_tx_land ON transaksjoner(land)",
            "CREATE INDEX IF NOT EXISTS idx_tx_score ON transaksjoner(score)",
        ]
        for stmt in idx:
            try:
                cur.execute(stmt)
            except sqlite3.OperationalError:
                pass

        conn.commit()


def lagre_til_db(df: pd.DataFrame):
    """Upsert transaksjoner (ingen wipe). Krever trans_id."""
    if df is None or df.empty:
        return

    init_db()
    df = df.copy()

    # Sørg for at trans_id finnes og er tekst
    if "trans_id" not in df.columns:
        raise ValueError("Mangler kolonnen 'trans_id' i DataFrame.")
    df["trans_id"] = df["trans_id"].astype(str)

    # Normaliser bool -> int for SQLite
    for bcol in ["mistenkelig", "mistenkelig_ml", "sanksjonert", "fuzzy_sanksjonert"]:
        if bcol in df.columns:
            df[bcol] = df[bcol].fillna(False).astype(bool).astype(int)

    # Sørg for at tidspunkt blir lagret som tekst
    if "tidspunkt" in df.columns:
        df["tidspunkt"] = pd.to_datetime(df["tidspunkt"], errors="coerce")
        df["tidspunkt"] = df["tidspunkt"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Velg bare kolonner vi støtter (andre ignoreres)
    allowed = [
        "trans_id",
        "fra_konto",
        "til_konto",
        "beløp",
        "tidspunkt",
        "land",
        "score",
        "risikonivå",
        "reasons_json",
        "mistenkelig",
        "mistenkelig_ml",
        "sanksjonert",
        "fuzzy_sanksjonert",
        "anomaly_score",
    ]
    for c in allowed:
        if c not in df.columns:
            df[c] = None
    df = df[allowed]

    # Upsert (REPLACE)
    placeholders = ",".join(["?"] * len(allowed))
    cols = ",".join(allowed)
    sql = f"REPLACE INTO transaksjoner ({cols}) VALUES ({placeholders})"

    rows = df.where(pd.notnull(df), None).values.tolist()

    with get_conn() as conn:
        cur = conn.cursor()
        cur.executemany(sql, rows)
        conn.commit()


def hent_transaksjoner(limit: int | None = None) -> pd.DataFrame:
    init_db()
    q = "SELECT * FROM transaksjoner ORDER BY datetime(tidspunkt) DESC"
    if limit:
        q += " LIMIT ?"
    with get_conn() as conn:
        if limit:
            df = pd.read_sql_query(q, conn, params=(int(limit),))
        else:
            df = pd.read_sql_query(q, conn)

    if "tidspunkt" in df.columns:
        df["tidspunkt"] = pd.to_datetime(df["tidspunkt"], errors="coerce")
    return df
