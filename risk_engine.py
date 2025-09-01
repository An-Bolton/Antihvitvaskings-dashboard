import pandas as pd

def analyser_transaksjoner(df):
    df = df.copy()
    df['mistenkelig'] = False

    # Høy sum
    df.loc[df['beløp'] > 1000000, 'mistenkelig'] = True

    # Risikoland
    risikoland = ['IR', 'KP', 'SY', 'AF']
    df.loc[df['land'].isin(risikoland), 'mistenkelig'] = True

    # Returtransaksjoner
    df['retur'] = df.duplicated(subset=['fra_konto', 'til_konto'], keep=False)
    df.loc[df['retur'], 'mistenkelig'] = True

    # Kombinert score
    df['score'] = 0
    df['score'] += df['beløp'] / 1_000_000
    df['score'] += df['land'].isin(risikoland).astype(int) * 2
    df['score'] += df['retur'].astype(int) * 1.5

    return df

def sjekk_mot_sanksjonsliste(df, sanksjonsliste):
    df = df.copy()
    df['sanksjonert'] = df['til_konto'].isin(sanksjonsliste['navn']) | df['fra_konto'].isin(sanksjonsliste['navn'])
    return df

