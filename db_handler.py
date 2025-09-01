import sqlite3
import pandas as pd

def init_db():
    conn = sqlite3.connect("transaksjoner.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS transaksjoner (
            trans_id INTEGER PRIMARY KEY,
            fra_konto TEXT,
            til_konto TEXT,
            beløp REAL,
            tidspunkt TEXT,
            land TEXT,
            score INTEGER,
            mistenkelig INTEGER,
            anomaly_score INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def lagre_til_db(df):
    conn = sqlite3.connect("transaksjoner.db")
    df.to_sql("transaksjoner", conn, if_exists='replace', index=False)
    conn.close()

def hent_transaksjoner():
    conn = sqlite3.connect("transaksjoner.db")
    df = pd.read_sql_query("SELECT * FROM transaksjoner", conn, parse_dates=['tidspunkt'])
    conn.close()
    return df
