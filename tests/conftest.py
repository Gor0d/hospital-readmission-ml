"""
Fixtures compartilhadas entre todos os testes.
"""

import os
import sqlite3
import sys
import tempfile

import pytest

# Garante que o root do projeto está no path
ROOT = os.path.join(os.path.dirname(__file__), "..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Configura variáveis de ambiente antes de qualquer import da aplicação
os.environ["SECRET_KEY"] = "test-secret-key-32-chars-minimum-ok"
os.environ["ENVIRONMENT"] = "development"


@pytest.fixture(scope="session")
def sample_patient() -> dict:
    """Paciente de exemplo válido para todos os testes."""
    return {
        "age_numeric": 72,
        "gender": "Female",
        "diag_primary": "Circulatory",
        "time_in_hospital": 5,
        "num_medications": 18,
        "num_procedures": 2,
        "num_diagnoses": 8,
        "num_lab_procedures": 45,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 2,
        "hba1c_result": ">8",
        "glucose_serum_test": ">200",
        "insulin": "Up",
        "change_medications": "Ch",
        "diabetes_medication": "Yes",
    }


@pytest.fixture(scope="session")
def sample_patient_low_risk() -> dict:
    """Paciente com perfil de baixo risco."""
    return {
        "age_numeric": 35,
        "gender": "Male",
        "diag_primary": "Injury",
        "time_in_hospital": 1,
        "num_medications": 3,
        "num_procedures": 0,
        "num_diagnoses": 1,
        "num_lab_procedures": 10,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "hba1c_result": "None",
        "glucose_serum_test": "None",
        "insulin": "No",
        "change_medications": "No",
        "diabetes_medication": "No",
    }


@pytest.fixture
def in_memory_audit_db() -> sqlite3.Connection:
    """Banco SQLite em memória para testes de auditoria isolados."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    from api.audit import init_audit_table
    init_audit_table(conn)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    """TestClient da aplicação FastAPI com banco de auditoria em arquivo temporário."""
    # Cria arquivo temporário para o banco de auditoria (evita problemas com :memory: e threads)
    tmp_dir = tmp_path_factory.mktemp("audit")
    audit_db_path = str(tmp_dir / "test_audit.db")

    # Patch do settings antes de importar a app
    import api.config as config_mod
    config_mod.settings.audit_db_path = audit_db_path

    from fastapi.testclient import TestClient
    from api.main import app
    from api.audit import init_audit_table
    from api.auth import init_users_table, create_user

    # Pré-inicializa o banco e cria usuários de teste
    init_audit_table()
    init_users_table()
    create_user("test_clinician", "senha_segura_123", "clinician", "Clínico Teste")
    create_user("test_admin", "senha_admin_123", "admin", "Admin Teste")
    create_user("test_viewer", "senha_viewer_123", "viewer", "Viewer Teste")

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def auth_headers(client):
    """Factory que retorna headers de autenticação para cada role."""
    _cache = {}

    def _get_headers(role: str) -> dict:
        if role in _cache:
            return _cache[role]
        user_map = {
            "clinician": ("test_clinician", "senha_segura_123"),
            "admin": ("test_admin", "senha_admin_123"),
            "viewer": ("test_viewer", "senha_viewer_123"),
        }
        username, password = user_map[role]
        resp = client.post("/token", data={"username": username, "password": password})
        assert resp.status_code == 200, f"Falha ao autenticar {role}: {resp.text}"
        token = resp.json()["access_token"]
        _cache[role] = {"Authorization": f"Bearer {token}"}
        return _cache[role]

    return _get_headers
