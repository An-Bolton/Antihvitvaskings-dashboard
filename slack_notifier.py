import requests
import json

# Sett inn Slack webhook-URL her
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/XXXX/YYYY/ZZZZ"

def send_slack_varsel(transaksjoner):
    if transaksjoner.empty:
        return

    for _, row in transaksjoner.iterrows():
        melding = f"*🚨 Mistenkelig transaksjon oppdaget:*"
        melding += f"Transaksjons-ID: `{row.get('trans_id', 'ukjent')}`"
        melding += f"Fra konto: `{row.get('fra_konto', 'ukjent')}`"
        melding += f"Til konto: `{row.get('til_konto', 'ukjent')}`"
        melding += f"Beløp: `{row.get('beløp', 'ukjent')}`"
        melding += f"Land: `{row.get('land', 'ukjent')}`"
        melding += f"Score: `{round(row.get('score', 0), 2)}`"

        if row.get("sanksjonert", False):
            melding += "\n:rotating_light: *TREFF I SANKSJONSLISTE* :rotating_light:"

        payload = {"text": melding}

        try:
            response = requests.post(SLACK_WEBHOOK_URL, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
            if response.status_code != 200:
                print(f"❌ Slack-feil: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"❌ Klarte ikke sende Slack-varsel: {e}")
