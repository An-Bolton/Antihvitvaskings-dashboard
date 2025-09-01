import pandas as pd
import requests
import os
from datetime import datetime, timedelta

# URL til OpenSanctions (PEP/sanksjoner)
SANCTIONS_URL = "https://data.opensanctions.org/datasets/latest/default/entities.ftm.json"
CACHE_FIL = "sanksjoner_cache.csv"
LAST_UPDATE_FILE = "sanksjoner_lastet.txt"

import os
import requests
import pandas as pd
from datetime import datetime, timedelta

SANCTIONS_URL = "https://data.opensanctions.org/entities/latest.json"
CACHE_FIL = "cache/sanksjonsliste.csv"
LAST_UPDATE_FILE = "cache/sist_oppdatert.txt"

def hent_sanksjonsdata(force_update=False):
    """
    Henter og cacher sanksjonslister fra ulike kilder (lokale og OpenSanctions).
    """
    nå = datetime.now()

    # Sjekk cache fra OpenSanctions
    if os.path.exists(LAST_UPDATE_FILE) and not force_update:
        with open(LAST_UPDATE_FILE, "r") as f:
            sist = datetime.fromisoformat(f.read().strip())
            if nå - sist < timedelta(hours=24):
                if os.path.exists(CACHE_FIL):
                    cached = pd.read_csv(CACHE_FIL)
                    cached['kilde'] = "OpenSanctions"
                    return cached

    print("⬆️ Laster ned sanksjonsliste fra OpenSanctions...")
    try:
        response = requests.get(SANCTIONS_URL)
        response.raise_for_status()
        data = response.json()

        navn = []
        for entity in data:
            if entity.get("schema") in ["Person", "LegalEntity"]:
                navn_felt = entity.get("names", [])
                for n in navn_felt:
                    navn.append({"navn": n, "kilde": "OpenSanctions"})

        opensanctions_df = pd.DataFrame(navn)
        opensanctions_df.drop_duplicates(inplace=True)
        opensanctions_df.to_csv(CACHE_FIL, index=False)
        with open(LAST_UPDATE_FILE, "w") as f:
            f.write(nå.isoformat())
    except Exception as e:
        print(f"❌ Klarte ikke hente data fra OpenSanctions: {e}")
        if os.path.exists(CACHE_FIL):
            opensanctions_df = pd.read_csv(CACHE_FIL)
            opensanctions_df['kilde'] = "OpenSanctions"
        else:
            opensanctions_df = pd.DataFrame(columns=["navn", "kilde"])

    # Last inn lokale lister
    lokale_kilder = {
        "FN": "data/sanksjonsliste_fn.csv",
        "OFAC": "data/sanksjonsliste_ofac.csv",
        "EU": "data/sanksjonsliste_eu.csv",
        "UK": "data/sanksjonsliste_uk.csv"
    }

    lokale_lister = []
    for kilde, sti in lokale_kilder.items():
        try:
            df = pd.read_csv(sti)
            df['kilde'] = kilde
            lokale_lister.append(df[['navn', 'kilde']])
        except Exception as e:
            print(f"⚠️ Kunne ikke lese lokal liste {kilde}: {e}")

    if lokale_lister:
        lokal_df = pd.concat(lokale_lister, ignore_index=True)
    else:
        lokal_df = pd.DataFrame(columns=["navn", "kilde"])

    samlet = pd.concat([opensanctions_df, lokal_df], ignore_index=True)
    samlet.drop_duplicates(inplace=True)
    return samlet


def sjekk_mot_sanksjonsliste(df, sanksjonsliste):
    df = df.copy()

    # Initier tomme kolonner
    df['sanksjonert'] = False
    df['sanksjonskilde'] = ""

    for index, rad in df.iterrows():
        fra = rad['fra_konto']
        til = rad['til_konto']

        match = sanksjonsliste[
            (sanksjonsliste['navn'] == fra) | (sanksjonsliste['navn'] == til)
            ]

        if not match.empty:
            df.at[index, 'sanksjonert'] = True
            kilder = match['kilde'].unique()
            df.at[index, 'sanksjonskilde'] = ", ".join(kilder)

    return df

