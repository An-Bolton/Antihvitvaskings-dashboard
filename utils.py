import pandas as pd

def trygg_les_csv(filnavn, datoformat=None, har_header=True, vis_advarsel=True):
    kolonnenavn = ['trans_id', 'fra_konto', 'til_konto', 'beløp', 'tidspunkt', 'land']

    def prøv_lesing(med_encoding):
        try:
            if har_header:
                df = pd.read_csv(
                    filnavn,
                    encoding=med_encoding,
                    parse_dates=['tidspunkt'],
                    date_parser=(
                        lambda x: pd.to_datetime(x, format=datoformat, errors='coerce')
                        if datoformat else None
                    )
                )
            else:
                df = pd.read_csv(
                    filnavn,
                    encoding=med_encoding,
                    names=kolonnenavn,
                    header=None,
                    parse_dates=['tidspunkt'],
                    date_parser=(
                        lambda x: pd.to_datetime(x, format=datoformat, errors='coerce')
                        if datoformat else None
                    )
                )

            # Tving datetime på nytt (sikkerhet)
            df['tidspunkt'] = pd.to_datetime(df['tidspunkt'], format=datoformat, errors='coerce')

            if vis_advarsel:
                ugyldige = df['tidspunkt'].isna().sum()
                if ugyldige > 0:
                    print(f"[⚠️] {ugyldige} rader med ugyldig dato – satt som NaT")

            return df

        except UnicodeDecodeError as e:
            raise e  # Fanges høyere opp

    # Prøv først med utf-8, så latin1
    try:
        return prøv_lesing('utf-8')
    except UnicodeDecodeError as e_utf:
        print(f"[!] UTF-8 feilet: {e_utf}")
        try:
            return prøv_lesing('latin1')
        except UnicodeDecodeError as e_latin:
            print(f"[!] Latin1 feilet: {e_latin}")
            raise
