from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes_cases import router as cases_router
from .routes_ingest import router as ingest_router

# DB schema init (må finnes i api/db.py)
from .db import ensure_core_schema

# Auth tables ligger i auth_db.py (ikke api/db.py)
from auth_db import ensure_auth_tables

app = FastAPI(title="AML Case API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # stram inn i prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    # Core DB
    ensure_core_schema()

    # Auth DB
    ensure_auth_tables()

    # Signals (optional): ikke krasj hvis den ikke finnes
    try:
        from .db import ensure_signal_tables
        ensure_signal_tables()
    except Exception:
        pass

    # Unified view (optional): ikke krasj hvis ikke implementert ennå
    try:
        from .db import ensure_v_all_transactions_view
        ensure_v_all_transactions_view()
    except Exception:
        pass


# Routers må inkluderes ETTER app er laget
app.include_router(cases_router, prefix="/cases", tags=["cases"])
app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])

@app.get("/health")
def health():
    return {"ok": True}
