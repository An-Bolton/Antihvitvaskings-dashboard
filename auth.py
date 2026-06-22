# auth.py
import streamlit as st

# =========================
# Session bootstrap
# =========================
def init_auth():
    if "auth" not in st.session_state:
        st.session_state.auth = {
            "logged_in": False,
            "username": None,
            "role": None,
        }


# =========================
# Users (MVP – bytt til DB senere)
# =========================
USERS = {
    "analyst":    {"password": "analyst",    "role": "analyst"},
    "senior":     {"password": "senior",     "role": "senior"},
    "compliance": {"password": "compliance", "role": "compliance"},
    "admin":      {"password": "admin",      "role": "admin"},
}

ROLE_ORDER = ["analyst", "senior", "compliance", "admin"]


# =========================
# Auth logic
# =========================
def verify_user(username: str, password: str):
    u = USERS.get(username)
    if u and u["password"] == password:
        return {"username": username, "role": u["role"]}
    return None


def login_ui():
    st.markdown("## 🔐 AML Intelligence Platform")
    st.caption("Sikker innlogging")

    with st.form("login_form"):
        username = st.text_input("Brukernavn")
        password = st.text_input("Passord", type="password")
        submit = st.form_submit_button("Logg inn")

    if submit:
        user = verify_user(username, password)
        if user:
            st.session_state.auth = {
                "logged_in": True,
                "username": user["username"],
                "role": user["role"],
            }
            st.success("Innlogging OK")
            st.rerun()
        else:
            st.error("Feil brukernavn eller passord")


def logout_button():
    if st.button("🚪 Logg ut"):
        st.session_state.auth = {
            "logged_in": False,
            "username": None,
            "role": None,
        }
        st.rerun()


# =========================
# Guards / helpers
# =========================
def require_login():
    if not st.session_state.auth["logged_in"]:
        login_ui()
        st.stop()


def has_role(required: str) -> bool:
    user_role = st.session_state.auth["role"]
    return ROLE_ORDER.index(user_role) >= ROLE_ORDER.index(required)
