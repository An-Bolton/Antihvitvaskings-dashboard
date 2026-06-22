import pandas as pd
import requests
import sys

API_URL = "http://127.0.0.1:8000/ingest/transactions"

def main(csv_path: str):
    df = pd.read_csv(csv_path)

    # Tilpass kolonner til din Transaction-model
    required = ["transaction_id", "customer_id", "amount", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Mangler kolonner i CSV: {missing}")

    payload = df.to_dict(orient="records")

    r = requests.post(API_URL, json=payload, timeout=30)
    r.raise_for_status()
    print(r.json())

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "transaksjoner.csv"
    main(path)
