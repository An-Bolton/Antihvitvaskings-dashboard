import time
import pandas as pd
import os

from risk_engine import analyser_transaksjoner
from ml_module import legg_til_anomalikluster
from db_handler import init_db, lagre_til_db
from utils import trygg_les_csv
from pep_checker import hent_sanksjonsdata, sjekk_mot_sanksjonsliste

# Sørger for at databasen er klar
init_db()

# Hent liste fra OpenSanctions
sanksjonsliste = hent_sanksjonsdata()

def overvåk_csv(filnavn, intervall=30):
    print("🔍 Starter overvåking av transaksjoner...")
    sist_lest = None

    while True:
        try:
            df = trygg_les_csv(filnavn, datoformat="%d.%m.%Y %H:%M", har_header=True)

            if sist_lest is None or len(df) > len(sist_lest):
                print("🔄 Nye transaksjoner oppdaget – analyserer...")

                # Analyse + maskinlæring
                df = analyser_transaksjoner(df)
                df = legg_til_anomalikluster(df)
                df = sjekk_mot_sanksjonsliste(df, sanksjonsliste)

                # Lagre til database
                lagre_til_db(df)

                # Filtrer mistenkelige
                mistenkelige = df[
                    df.get('mistenkelig', False) |
                    df.get('mistenkelig_ml', False) |
                    df.get('sanksjonert', False)
                ]

                if not mistenkelige.empty:
                    print(f"⚠️  Fant {len(mistenkelige)} mistenkelige transaksjoner:")
                    kolonner = ['trans_id', 'fra_konto', 'til_konto', 'beløp', 'land', 'score']
                    eksisterende = [k for k in kolonner if k in mistenkelige.columns]
                    print("Eksisterende kolonner:", eksisterende)
                    print(mistenkelige.loc[:, eksisterende])
                else:
                    print("✅ Ingen mistenkelige transaksjoner i nye data.")

                sist_lest = df.copy()
            else:
                print(f"🟡 Ingen nye rader. Sjekker igjen om {intervall} sek...")

        except Exception as e:
            print("❌ Feil under overvåking:", e)

        time.sleep(intervall)

# Hvis du også vil overvåke input-mappen (valgfritt)
def overvåk_inputmappe(mappe="input/"):
    sett = set()

    while True:
        alle_filer = set(f for f in os.listdir(mappe) if f.endswith('.csv'))
        nye_filer = alle_filer - sett

        for fil in nye_filer:
            full_path = os.path.join(mappe, fil)
            print(f"📥 Ny CSV funnet: {fil}")
            try:
                df = trygg_les_csv(full_path)
                df = analyser_transaksjoner(df)
                df = legg_til_anomalikluster(df)
                df = sjekk_mot_sanksjonsliste(df, sanksjonsliste)
                lagre_til_db(df)
                print(f"✅ {fil} analysert og lagret")
            except Exception as e:
                print(f"❌ Feil med {fil}: {e}")

        sett = alle_filer
        time.sleep(30)

if __name__ == "__main__":
    overvåk_csv("transaksjoner.csv")
    # Evt: overvåk_inputmappe()  # for å aktivere automatisk import
