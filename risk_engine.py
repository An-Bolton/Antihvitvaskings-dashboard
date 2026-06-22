# risk_engine.py
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class RiskConfig:
    high_amount_threshold: float = 1_000_000.0
    medium_amount_threshold: float = 250_000.0

    # ISO-like country codes or your internal codes
    high_risk_countries: tuple[str, ...] = ("IR", "KP", "SY", "AF", "RU")

    # Night hours (0-5)
    night_start: int = 0
    night_end: int = 5

    # Duplicate/return window logic: same pair within dataset
    # (simple baseline; can be upgraded to time-window later)
    return_pair_weight: float = 0.20

    # Weights for score composition (sum doesn't have to be 1.0)
    w_amount: float = 0.35
    w_high_risk_country: float = 0.35
    w_night: float = 0.10
    w_return: float = 0.10
    w_sanctions: float = 0.50  # strong signal if true

    # Risk band cutoffs
    cutoff_medium: float = 0.50
    cutoff_high: float = 0.80


CFG = RiskConfig()

# ----------------------------
# DB-backed config (Rule engine v2)
# ----------------------------

DEFAULT_DB_PATH = os.environ.get("AML_DB_PATH", "hvitvask.db")


def _get_conn(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path, check_same_thread=False)


def _fetch_rules(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Read rules from SQLite. Returns empty list if table missing."""
    try:
        with _get_conn(db_path) as c:
            cur = c.cursor()
            rows = cur.execute(
                """SELECT rule_id, enabled, weight, rule_type, params_json
                   FROM rules"""
            ).fetchall()
        out = []
        for rule_id, enabled, weight, rule_type, params_json in rows:
            try:
                params = json.loads(params_json) if params_json else {}
            except Exception:
                params = {}
            out.append(
                {
                    "rule_id": rule_id,
                    "enabled": int(enabled or 0),
                    "weight": float(weight or 0.0),
                    "rule_type": str(rule_type or ""),
                    "params": params,
                }
            )
        return out
    except Exception:
        return []


def load_config_from_db(db_path: str = DEFAULT_DB_PATH, fallback: RiskConfig = CFG) -> RiskConfig:
    """Build a RiskConfig from rules in DB (Rule engine v2)."""
    rules = _fetch_rules(db_path)
    if not rules:
        return fallback

    # start with fallback defaults
    cfg = fallback

    # helper to find rule
    def r(rule_id: str) -> dict | None:
        for it in rules:
            if it.get("rule_id") == rule_id:
                return it
        return None

    amount = r("amount_band")
    country = r("high_risk_country")
    night = r("night_activity")
    ret = r("return_payment")
    sanc = r("sanctions_hit")
    bands = r("risk_bands")

    kwargs = cfg.__dict__.copy()

    if amount and amount["enabled"]:
        p = amount.get("params") or {}
        kwargs["high_amount_threshold"] = float(p.get("high_amount_threshold", kwargs["high_amount_threshold"]))
        kwargs["medium_amount_threshold"] = float(p.get("medium_amount_threshold", kwargs["medium_amount_threshold"]))
        kwargs["w_amount"] = float(amount.get("weight", kwargs["w_amount"]))

    if country and country["enabled"]:
        p = country.get("params") or {}
        countries = p.get("high_risk_countries", kwargs["high_risk_countries"])
        if isinstance(countries, (list, tuple)):
            kwargs["high_risk_countries"] = tuple(str(x) for x in countries)
        kwargs["w_high_risk_country"] = float(country.get("weight", kwargs["w_high_risk_country"]))

    if night and night["enabled"]:
        p = night.get("params") or {}
        kwargs["night_start"] = int(p.get("night_start", kwargs["night_start"]))
        kwargs["night_end"] = int(p.get("night_end", kwargs["night_end"]))
        kwargs["w_night"] = float(night.get("weight", kwargs["w_night"]))

    if ret and ret["enabled"]:
        kwargs["w_return"] = float(ret.get("weight", kwargs["w_return"]))
    else:
        kwargs["w_return"] = 0.0

    if sanc and sanc["enabled"]:
        kwargs["w_sanctions"] = float(sanc.get("weight", kwargs["w_sanctions"]))
    else:
        kwargs["w_sanctions"] = 0.0

    if bands and bands["enabled"]:
        p = bands.get("params") or {}
        kwargs["cutoff_medium"] = float(p.get("cutoff_medium", kwargs["cutoff_medium"]))
        kwargs["cutoff_high"] = float(p.get("cutoff_high", kwargs["cutoff_high"]))

    return RiskConfig(**kwargs)



# ----------------------------
# Helpers
# ----------------------------

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan


def _to_amount(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _risk_label(score: float, cutoff_medium: float, cutoff_high: float) -> str:
    if score >= cutoff_high:
        return "🔺 Høy"
    if score >= cutoff_medium:
        return "⚡ Medium"
    return "✅ Lav"


def _safe_bool_series(x, n: int) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.fillna(False).astype(bool)
    return pd.Series([False] * n)


# ----------------------------
# Public API
# ----------------------------

def analyser_transaksjoner(df: pd.DataFrame, config: RiskConfig | None = None, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Superduper risk engine:
    - Robust against missing cols / types
    - Produces: mistenkelig (rules), score (0..1), risikonivå, reasons_json
    - Adds: retur (duplicate pair indicator), natt (night activity flag), land_risk flag, amount_band
    """
    if config is None:
        config = load_config_from_db(db_path=db_path, fallback=CFG)

    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    df = df.copy()

    _ensure_cols(df, ["trans_id", "fra_konto", "til_konto", "beløp", "land", "tidspunkt"])
    n = len(df)

    # Normalize types
    df["beløp"] = _to_amount(df["beløp"])
    df["tidspunkt"] = _to_dt(df["tidspunkt"])
    df["land"] = df["land"].astype(str).str.upper().replace({"NAN": ""})

    # ---------- Rule features ----------
    # Amount bands
    df["amount_band"] = np.where(
        df["beløp"] >= config.high_amount_threshold, "HIGH",
        np.where(df["beløp"] >= config.medium_amount_threshold, "MEDIUM", "LOW")
    )
    high_amount_flag = df["beløp"] >= config.high_amount_threshold
    med_amount_flag = (df["beløp"] >= config.medium_amount_threshold) & (~high_amount_flag)

    # Country risk
    land_risk_flag = df["land"].isin(set(config.high_risk_countries))

    # Night activity
    hours = df["tidspunkt"].dt.hour
    night_flag = hours.between(config.night_start, config.night_end, inclusive="both").fillna(False)

    # Return/duplicate pairs (simple baseline)
    pair = df[["fra_konto", "til_konto"]].astype(str).fillna("")
    df["retur"] = pair.duplicated(keep=False)

    # ---------- Rule-based suspicious flag ----------
    df["mistenkelig"] = False
    df.loc[high_amount_flag, "mistenkelig"] = True
    df.loc[land_risk_flag, "mistenkelig"] = True
    df.loc[df["retur"].fillna(False), "mistenkelig"] = True

    # Ensure optional columns exist (dashboard expects them sometimes)
    if "mistenkelig_ml" not in df.columns:
        df["mistenkelig_ml"] = False

    # ---------- Score components (0..1-ish) ----------
    # Amount contribution normalized within current dataset
    max_amount = float(df["beløp"].max()) if df["beløp"].max() > 0 else 1.0
    amount_norm = (df["beløp"] / max_amount).clip(0, 1)

    c_amount = amount_norm * config.w_amount
    c_country = land_risk_flag.astype(float) * config.w_high_risk_country
    c_night = night_flag.astype(float) * config.w_night
    c_return = df["retur"].fillna(False).astype(float) * config.w_return

    # sanctions handled here only if df already has sanksjonert column
    sanc_flag = _safe_bool_series(df.get("sanksjonert", False), n)
    c_sanctions = sanc_flag.astype(float) * config.w_sanctions

    raw_score = c_amount + c_country + c_night + c_return + c_sanctions
    df["score"] = raw_score.clip(0, 1)

    df["risikonivå"] = df["score"].apply(lambda s: _risk_label(float(s), config.cutoff_medium, config.cutoff_high))

    # ---------- Explainability ----------
    # Build reasons per row as a dict (stored as JSON string for SQLite friendliness)
    reasons: List[str] = []
    for i in range(n):
        r: Dict[str, Any] = {
            "components": {
                "amount": round(float(c_amount.iat[i]), 4),
                "high_risk_country": round(float(c_country.iat[i]), 4),
                "night": round(float(c_night.iat[i]), 4),
                "return_pair": round(float(c_return.iat[i]), 4),
                "sanctions": round(float(c_sanctions.iat[i]), 4),
            },
            "rules_triggered": [],
            "signals": {},
        }

        if bool(high_amount_flag.iat[i]):
            r["rules_triggered"].append(f"beløp >= {int(config.high_amount_threshold):,}".replace(",", " "))
        elif bool(med_amount_flag.iat[i]):
            r["rules_triggered"].append(f"beløp >= {int(config.medium_amount_threshold):,}".replace(",", " "))

        if bool(land_risk_flag.iat[i]):
            r["rules_triggered"].append(f"høyrisikoland={df['land'].iat[i]}")

        if bool(night_flag.iat[i]):
            r["rules_triggered"].append("natt (00–05)")

        if bool(df["retur"].iat[i]):
            r["rules_triggered"].append("retur/duplikat konto-par")

        if bool(sanc_flag.iat[i]):
            r["rules_triggered"].append("sanksjonert")

        r["signals"]["amount_band"] = df["amount_band"].iat[i]
        r["signals"]["land"] = df["land"].iat[i]
        r["signals"]["hour"] = int(df["tidspunkt"].iat[i].hour) if pd.notnull(df["tidspunkt"].iat[i]) else None

        reasons.append(json.dumps(r, ensure_ascii=False))

    df["reasons_json"] = reasons

    return df


def sjekk_mot_sanksjonsliste(df: pd.DataFrame, sanksjonsliste: pd.DataFrame) -> pd.DataFrame:
    """
    Robust sanctions check:
    - supports varying column names in sanctions list
    - sets df['sanksjonert'] boolean
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    df = df.copy()
    if sanksjonsliste is None or getattr(sanksjonsliste, "empty", True):
        df["sanksjonert"] = False
        return df

    # figure out which column contains names/ids in sanctions list
    cand_cols = [c for c in ["navn", "name", "entity", "person", "alias"] if c in sanksjonsliste.columns]
    if not cand_cols:
        df["sanksjonert"] = False
        return df

    sanc_values = set()
    for c in cand_cols:
        sanc_values.update(sanksjonsliste[c].dropna().astype(str).tolist())

    # compare against both accounts
    _ensure_cols(df, ["fra_konto", "til_konto"])
    fra = df["fra_konto"].astype(str)
    til = df["til_konto"].astype(str)

    df["sanksjonert"] = fra.isin(sanc_values) | til.isin(sanc_values)
    return df
