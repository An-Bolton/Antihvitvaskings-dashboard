import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def legg_til_anomalikluster(df, n_clusters=3):
    """
    Kjører KMeans-klustering på utvalgte variabler og legger til cluster-ID.
    Returnerer DataFrame med ny kolonne: 'kluster'
    """

    # Sjekk at nødvendige kolonner finnes
    påkrevde = ['beløp']
    for kol in påkrevde:
        if kol not in df.columns:
            raise ValueError(f"Mangler kolonne '{kol}' i DataFrame")

    # Forbered data
    X = df[['beløp']].copy()
    X.fillna(0, inplace=True)

    # Standardiser
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Kjør KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    klustere = kmeans.fit_predict(X_scaled)

    # Legg til i DataFrame
    df['kluster'] = klustere

    return df
