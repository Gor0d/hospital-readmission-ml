"""
Módulo de autenticação JWT para a API clínica.

Roles disponíveis:
  - admin     → acesso total (predições + auditoria + administração)
  - clinician → acesso às predições e explicabilidade
  - viewer    → apenas leitura (health, model-info)

Usuários armazenados em SQLite (mesma DB de auditoria, tabela 'users').
"""

import hashlib
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from api.config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Crypto helpers
# ──────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

ROLES = {"admin", "clinician", "viewer"}


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ──────────────────────────────────────────────
# Database helpers
# ──────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    db_path = settings.audit_db_path
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_users_table(conn: Optional[sqlite3.Connection] = None) -> None:
    """Cria tabela de usuários se não existir."""
    close = conn is None
    if conn is None:
        conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT UNIQUE NOT NULL,
            hashed_pw   TEXT NOT NULL,
            role        TEXT NOT NULL CHECK(role IN ('admin', 'clinician', 'viewer')),
            full_name   TEXT,
            active      INTEGER NOT NULL DEFAULT 1,
            created_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    if close:
        conn.close()


def create_user(username: str, password: str, role: str, full_name: str = "",
                conn: Optional[sqlite3.Connection] = None) -> None:
    """Cria novo usuário. Levanta ValueError se username já existe."""
    if role not in ROLES:
        raise ValueError(f"Role inválido: {role}. Use: {ROLES}")
    close = conn is None
    if conn is None:
        conn = _get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, hashed_pw, role, full_name, created_at) VALUES (?,?,?,?,?)",
            (username, hash_password(password), role, full_name,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        logger.info("Usuário criado: %s [%s]", username, role)
    except sqlite3.IntegrityError:
        raise ValueError(f"Usuário '{username}' já existe.")
    finally:
        if close:
            conn.close()


def get_user(username: str, conn: Optional[sqlite3.Connection] = None) -> Optional[dict]:
    close = conn is None
    if conn is None:
        conn = _get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE username=? AND active=1", (username,)
    ).fetchone()
    if close:
        conn.close()
    return dict(row) if row else None


def ensure_default_admin() -> None:
    """
    Garante que exista ao menos um usuário admin na base.
    Usado apenas em desenvolvimento. Em produção, crie usuários via CLI.
    """
    if settings.is_production:
        return
    conn = _get_db()
    init_users_table(conn)
    if not conn.execute("SELECT 1 FROM users WHERE role='admin'").fetchone():
        create_user("admin", "admin123", "admin", "Administrador Padrão", conn)
        logger.warning(
            "Usuário admin padrão criado (senha: admin123). "
            "ALTERE IMEDIATAMENTE em produção via POST /admin/users."
        )
    conn.close()


# ──────────────────────────────────────────────
# Token
# ──────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    expires_in: int


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None


def create_access_token(data: dict) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload.update({"exp": expire})
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_pw"]):
        return None
    return user


# ──────────────────────────────────────────────
# FastAPI dependencies
# ──────────────────────────────────────────────

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido ou expirado.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    user = get_user(username)
    if user is None:
        raise credentials_exc
    return user


def require_role(allowed_roles: list[str]):
    """
    Dependency factory para controle de acesso por role.
    Uso: Depends(require_role(["admin", "clinician"]))
    """
    async def role_checker(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permissão insuficiente. Role necessário: {allowed_roles}",
            )
        return user
    return role_checker


# ──────────────────────────────────────────────
# CLI helper — cria usuário via linha de comando
# ──────────────────────────────────────────────

def cli_create_user(username: str, password: str, role: str, full_name: str = "") -> None:
    """
    Uso: python -c "from api.auth import cli_create_user; cli_create_user('joao', 'senha', 'clinician')"
    """
    conn = _get_db()
    init_users_table(conn)
    create_user(username, password, role, full_name, conn)
    conn.close()
    print(f"Usuário '{username}' criado com role '{role}'.")
