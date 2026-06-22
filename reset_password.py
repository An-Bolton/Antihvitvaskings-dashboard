import os
import sqlite3
import hashlib
import secrets

DB_PATH = os.environ.get("AML_DB_PATH", "transaksjoner.db")
USERNAME = "admin"                  # <- endre ved behov
NEW_PASSWORD = "CHANGE_ME" # <- endre ved behov

def pbkdf2_hash(password: str, salt: str | None = None, iterations: int = 200_000) -> str:
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${dk.hex()}"

conn = sqlite3.connect(DB_PATH)
row = conn.execute("SELECT 1 FROM users WHERE username=?", (USERNAME.strip(),)).fetchone()
if not row:
    raise SystemExit(f"Fant ikke bruker '{USERNAME}' i {DB_PATH}")

new_hash = pbkdf2_hash(NEW_PASSWORD)

conn.execute(
    "UPDATE users SET password_hash=? WHERE username=?",
    (new_hash, USERNAME.strip()),
)
conn.commit()
conn.close()

print(f"✅ Passord resatt for '{USERNAME}' i {DB_PATH}")
