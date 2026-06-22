"""
auth_db.py

Sidebar-based authentication for Streamlit with persistent login across refresh.

Key properties:
- Login UI is rendered in the sidebar via `sidebar_auth_box(prefix=...)`.
- The rest of the app is gated by `login_gate(title)`.
- Persistence across hard refresh uses a JS cookie probe that redirects once with ?__aml_token=...
- Sessions are stored server-side in SQLite; cookie stores only the raw token.
- No idle timeout (by design). Logout ONLY when user clicks "Logg ut".
- Widget keys are namespaced by `prefix` to avoid DuplicateElementId issues.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

import core_schema

COOKIE_NAME = "aml_session"
TOKEN_QP = "__aml_token"

# "Praktisk talt aldri" auto-logout
FOREVER_DAYS = int(os.environ.get("AML_SESSION_TTL_DAYS", "3650"))  # ~10 years
FOREVER_SECONDS = FOREVER_DAYS * 24 * 3600


# --------------------------
# Time helpers
# --------------------------
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


# --------------------------
# DB helpers
# --------------------------
def _db_path() -> str:
    return os.environ.get("AML_DB_PATH", "transaksjoner.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    core_schema.ensure_core_schema(conn)
    conn.commit()
    return conn


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pbkdf2_hash(password: str, salt: Optional[str] = None, iterations: int = 200_000) -> str:
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${dk.hex()}"


def _pbkdf2_verify(password: str, stored: str) -> bool:
    try:
        algo, it_s, salt, _hex = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(it_s)
        check = _pbkdf2_hash(password, salt=salt, iterations=iterations)
        return secrets.compare_digest(check, stored)
    except Exception:
        return False


def _users_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])


def create_user(conn: sqlite3.Connection, username: str, password: str, role: str) -> None:
    ph = _pbkdf2_hash(password)
    conn.execute(
        "INSERT OR REPLACE INTO users(username,password_hash,role,is_active,created_at) VALUES (?,?,?,?,?)",
        (username.strip(), ph, role, 1, _iso(_utcnow())),
    )
    conn.commit()


def authenticate_user(conn: sqlite3.Connection, username: str, password: str) -> tuple[bool, str]:
    row = conn.execute(
        "SELECT password_hash, role, is_active FROM users WHERE username=?",
        (username.strip(),),
    ).fetchone()
    if not row:
        return False, ""
    stored, role, is_active = row
    if int(is_active) != 1:
        return False, ""
    if stored and _pbkdf2_verify(password, stored):
        return True, str(role or "analyst")
    return False, ""


def create_session(conn: sqlite3.Connection, username: str, role: str, ttl_days: int = FOREVER_DAYS) -> str:
    token = secrets.token_urlsafe(32)
    token_hash = _sha256(token)
    now = _utcnow()
    exp = now + timedelta(days=ttl_days)
    conn.execute(
        """
        INSERT OR REPLACE INTO sessions(token_hash, username, role, created_at, expires_at, revoked_at)
        VALUES (?,?,?,?,?,NULL)
        """,
        (token_hash, username, role, _iso(now), _iso(exp)),
    )
    conn.commit()
    return token


def revoke_session(conn: sqlite3.Connection, token: str) -> None:
    token_hash = _sha256(token)
    conn.execute("UPDATE sessions SET revoked_at=? WHERE token_hash=?", (_iso(_utcnow()), token_hash))
    conn.commit()


def validate_session(conn: sqlite3.Connection, token: str) -> Optional[dict]:
    token_hash = _sha256(token)
    row = conn.execute(
        "SELECT username, role, expires_at, revoked_at FROM sessions WHERE token_hash=?",
        (token_hash,),
    ).fetchone()
    if not row:
        return None
    username, role, expires_at, revoked_at = row
    if revoked_at:
        return None
    try:
        exp = _parse_iso(str(expires_at))
    except Exception:
        return None
    if exp < _utcnow():
        return None
    return {"username": str(username), "role": str(role or "analyst"), "expires_at": str(expires_at)}


# --------------------------
# Cookie helpers (JS)
# --------------------------
def _cookie_set_js(token: str, max_age_seconds: int = FOREVER_SECONDS) -> None:
    # Cookie-hardening for real-world Streamlit quirks:
    # - Write a Lax cookie (works for normal top-level navigation).
    # - Additionally, when possible, write SameSite=None;Secure (helps if the app is embedded/iframes).
    #   Browsers require Secure when SameSite=None; localhost is typically allowed.
    js = f"""
    <script>
    (function() {{
        var maxAge = {int(max_age_seconds)};
        var v = encodeURIComponent("{token}");
        // Default: Lax
        document.cookie = "{COOKIE_NAME}=" + v + "; Path=/; Max-Age=" + maxAge + "; SameSite=Lax";

        // Extra: try None+Secure for embedded contexts
        try {{
            var isHttps = (window.location && window.location.protocol === 'https:');
            var isLocalhost = (window.location && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'));
            if (isHttps || isLocalhost) {{
                document.cookie = "{COOKIE_NAME}=" + v + "; Path=/; Max-Age=" + maxAge + "; SameSite=None; Secure";
            }}
        }} catch (e) {{}}
    }})();
    </script>
    """
    components.html(js, height=0, width=0)


def _cookie_clear_js() -> None:
    js = f"""
    <script>
    (function() {{
        // Clear both variants we might have set.
        document.cookie = "{COOKIE_NAME}=; Path=/; Max-Age=0; SameSite=Lax";
        document.cookie = "{COOKIE_NAME}=; Path=/; Max-Age=0; SameSite=None; Secure";
    }})();
    </script>
    """
    components.html(js, height=0, width=0)


def _cookie_redirect_probe_js() -> None:
    js = f"""
    <script>
    (function() {{
        function getCookie(name) {{
            var m = document.cookie.match('(^|;)\\\\s*' + name + '\\\\s*=\\\\s*([^;]+)');
            return m ? decodeURIComponent(m.pop()) : "";
        }}
        var token = getCookie("{COOKIE_NAME}");
        if (!token) return;

        var url = new URL(window.location.href);
        if (url.searchParams.get("{TOKEN_QP}")) return;

        url.searchParams.set("{TOKEN_QP}", token);
        window.location.replace(url.toString());
    }})();
    </script>
    """
    components.html(js, height=0, width=0)


# --------------------------
# Query param helpers
# --------------------------
def _get_query_param(name: str) -> Optional[str]:
    try:
        v = st.query_params.get(name)
        if isinstance(v, list):
            return v[0] if v else None
        return v
    except Exception:
        qp = st.experimental_get_query_params()
        v = qp.get(name)
        return v[0] if v else None


def _clear_query_params() -> None:
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()


# --------------------------
# Public helpers
# --------------------------
def has_role(*roles: str) -> bool:
    auth = st.session_state.get("auth") or {}
    role = (auth.get("role") or "").lower()
    return role in {r.lower() for r in roles}


def sidebar_auth_box(prefix: str = "auth") -> None:
    """Render login/logout UI in sidebar."""
    with st.sidebar:
        conn = _connect()

        # Debug (helps when DB path mismatch causes "lost users")
        st.caption(f"DB: `{os.path.abspath(_db_path())}`")

        auth = st.session_state.get("auth") or {}
        if auth.get("username"):
            st.caption(f"✅ Logget inn som **{auth['username']}** ({auth.get('role','')})")
            # IMPORTANT: Do NOT rerun on every render (would cause an infinite rerun loop).
            # Only rerun when the user explicitly logs out.
            if st.button("🚪 Logg ut", use_container_width=True, key=f"{prefix}_logout"):
                token = st.session_state.get("auth_token")
                if token:
                    try:
                        revoke_session(conn, token)
                    except Exception:
                        pass
                st.session_state.pop("auth", None)
                st.session_state.pop("auth_token", None)
                _cookie_clear_js()
                _clear_query_params()
                st.rerun()
            return

        # Bootstrap first user
        if _users_count(conn) == 0:
            st.warning("Ingen brukere finnes ennå. Opprett første admin-bruker.")
            with st.form(f"{prefix}_bootstrap_form", clear_on_submit=False):
                u = st.text_input("Brukernavn", value="admin", key=f"{prefix}_bootstrap_user")
                p = st.text_input("Passord", type="password", key=f"{prefix}_bootstrap_pw")
                role = st.selectbox("Rolle", ["admin", "compliance", "senior", "analyst"], index=0, key=f"{prefix}_bootstrap_role")
                ok = st.form_submit_button("Opprett bruker", use_container_width=True)
            if ok:
                if not u.strip() or not p:
                    st.error("Fyll inn brukernavn og passord.")
                else:
                    create_user(conn, u.strip(), p, role)
                    st.success("Bruker opprettet. Logg inn under.")
            st.divider()

        with st.form(f"{prefix}_login_form", clear_on_submit=False):
            username = st.text_input("Brukernavn", key=f"{prefix}_u")
            password = st.text_input("Passord", type="password", key=f"{prefix}_p")
            submitted = st.form_submit_button("Logg inn", use_container_width=True)

        if submitted:
            ok, role = authenticate_user(conn, username, password)
            if not ok:
                st.error("Feil brukernavn eller passord.")
                return

            token = create_session(conn, username.strip(), role, ttl_days=FOREVER_DAYS)
            _cookie_set_js(token, max_age_seconds=FOREVER_SECONDS)

            st.session_state["auth"] = {"username": username.strip(), "role": role}
            st.session_state["auth_token"] = token
            # Make login resilient: carry token in query params for the next rerun (cookie write can lag a frame).
            # NOTE: Do NOT immediately clear query params after setting, otherwise the token is lost.
            # We keep __aml_token for one hop; login_gate will clear it after successful validation.
            try:
                st.query_params[TOKEN_QP] = token
            except Exception:
                try:
                    st.experimental_set_query_params(**{TOKEN_QP: token})
                except Exception:
                    pass
            st.rerun()


def login_gate(title: str = "Login") -> None:
    """Gate the app until authenticated."""
    auth = st.session_state.get("auth") or {}
    if auth.get("username"):
        return

    # Important: run cookie probe BEFORE stopping so refresh can re-auth.
    _cookie_redirect_probe_js()

    conn = _connect()
    token = _get_query_param(TOKEN_QP)

    if token:
        info = validate_session(conn, token)
        if info:
            st.session_state["auth"] = {"username": info["username"], "role": info.get("role", "analyst")}
            st.session_state["auth_token"] = token
            _clear_query_params()
            # Keep cookie fresh "forever"
            _cookie_set_js(token, max_age_seconds=FOREVER_SECONDS)
            st.rerun()
        else:
            # Do NOT auto-clear cookie here; allow user to re-login if needed.
            _clear_query_params()

    st.title(title)
    st.info("Logg inn i sidebaren for å fortsette.")
    st.stop()
