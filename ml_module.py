from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def legg_til_anomalikluster(df, antall_klynger=3):
    df = df.copy()

    # Forbered data for clustering
    features = ['beløp']
    X = df[features].fillna(0).values.reshape(-1, 1)

    # Unngå crash ved for lite data
    if len(X) < antall_klynger:
        df['mistenkelig_ml'] = False
        df['cluster'] = -1
        return df

    # KMeans clustering
    kmeans = KMeans(n_clusters=antall_klynger, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(X)

    # Identifiser anomaliklynge (den med høyest gj.snitt)
    cluster_means = df.groupby('cluster')['beløp'].mean()
    anomaliklynge = cluster_means.idxmax()
    df['mistenkelig_ml'] = df['cluster'] == anomaliklynge

    return df