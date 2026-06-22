# Anti-Money Laundering Dashboard

A Python-based Anti-Money Laundering (AML) platform built with Streamlit and FastAPI for transaction monitoring, risk assessment, sanctions screening, and case management.

## Features

* Transaction monitoring
* Risk scoring engine
* PEP (Politically Exposed Person) screening
* Sanctions list matching
* Case management workflow
* KYC (Know Your Customer) / EDD (Enhanced Due Diligence)
* PDF report generation
* Slack notifications
* Machine learning analytics

## Technology Stack

* Python
* Streamlit
* FastAPI
* SQLite
* Docker
* Scikit-learn
* Plotly

## Run Locally

```bash
streamlit run dashboard.py
```

## Run with Docker

Build the Docker image:

```bash
docker build -t bolton-aml-dashboard:v3 .
```

Run the container:

```bash
docker run --rm -p 127.0.0.1:8525:8501 bolton-aml-dashboard:v3
```

Open the application in your browser:

```text
http://127.0.0.1:8525
```

## Project Structure

```text
dashboard.py       # Main Streamlit dashboard
risk_engine.py     # Risk scoring engine
auth.py            # Authentication logic
auth_db.py         # User authentication database
db_handler.py      # Database operations
models.py          # Data models
api/               # FastAPI backend endpoints
```

## Core AML Capabilities

* Transaction monitoring
* Risk scoring and alerting
* PEP and sanctions screening
* KYC / EDD workflows
* Case management
* PDF reporting
* Machine learning-assisted analysis

## Disclaimer

This project is a prototype AML platform developed for educational, research, and software development purposes.
