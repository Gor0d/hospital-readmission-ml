"""
Testes de integração para api/main.py
"""

import pytest


class TestAuth:
    def test_login_valid_clinician(self, client):
        resp = client.post("/token", data={"username": "test_clinician", "password": "senha_segura_123"})
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert body["role"] == "clinician"

    def test_login_invalid_password(self, client):
        resp = client.post("/token", data={"username": "test_clinician", "password": "errada"})
        assert resp.status_code == 401

    def test_login_unknown_user(self, client):
        resp = client.post("/token", data={"username": "naoexiste", "password": "qualquer"})
        assert resp.status_code == 401


class TestPublicEndpoints:
    def test_health_no_auth(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True

    def test_root_no_auth(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "model" in resp.json()


class TestPredict:
    def test_predict_without_auth_returns_401(self, client, sample_patient):
        resp = client.post("/predict", json=sample_patient)
        assert resp.status_code == 401

    def test_predict_viewer_returns_403(self, client, sample_patient, auth_headers):
        resp = client.post("/predict", json=sample_patient, headers=auth_headers("viewer"))
        assert resp.status_code == 403

    def test_predict_clinician_returns_200(self, client, sample_patient, auth_headers):
        resp = client.post("/predict", json=sample_patient, headers=auth_headers("clinician"))
        assert resp.status_code == 200

    def test_predict_response_schema(self, client, sample_patient, auth_headers):
        resp = client.post("/predict", json=sample_patient, headers=auth_headers("clinician"))
        body = resp.json()
        assert "readmission_probability" in body
        assert "risk_level" in body
        assert "prediction" in body
        assert "recommendation" in body
        assert "model_auc" in body
        assert "threshold_used" in body
        assert "calibrated" in body

    def test_predict_probability_in_range(self, client, sample_patient, auth_headers):
        resp = client.post("/predict", json=sample_patient, headers=auth_headers("clinician"))
        prob = resp.json()["readmission_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_prediction_is_binary(self, client, sample_patient, auth_headers):
        resp = client.post("/predict", json=sample_patient, headers=auth_headers("clinician"))
        assert resp.json()["prediction"] in (0, 1)

    def test_predict_risk_level_valid(self, client, sample_patient, auth_headers):
        resp = client.post("/predict", json=sample_patient, headers=auth_headers("clinician"))
        assert resp.json()["risk_level"] in ("Baixo", "Moderado", "Alto")

    def test_predict_invalid_field_returns_422(self, client, auth_headers):
        bad_patient = {"age_numeric": 999}  # campo fora do range e incompleto
        resp = client.post("/predict", json=bad_patient, headers=auth_headers("clinician"))
        assert resp.status_code == 422


class TestBatch:
    def test_batch_valid(self, client, sample_patient, sample_patient_low_risk, auth_headers):
        resp = client.post("/predict-batch",
                           json=[sample_patient, sample_patient_low_risk],
                           headers=auth_headers("clinician"))
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_batch_empty_returns_422(self, client, auth_headers):
        resp = client.post("/predict-batch", json=[], headers=auth_headers("clinician"))
        assert resp.status_code == 422

    def test_batch_without_auth_returns_401(self, client, sample_patient):
        resp = client.post("/predict-batch", json=[sample_patient])
        assert resp.status_code == 401


class TestExplain:
    def test_explain_returns_feature_contributions(self, client, sample_patient, auth_headers):
        resp = client.post("/explain", json=sample_patient, headers=auth_headers("clinician"))
        assert resp.status_code == 200
        body = resp.json()
        assert "feature_contributions" in body
        assert len(body["feature_contributions"]) == 19

    def test_explain_top_risk_factors_positive_shap(self, client, sample_patient, auth_headers):
        resp = client.post("/explain", json=sample_patient, headers=auth_headers("clinician"))
        body = resp.json()
        contributions = body["feature_contributions"]
        for feat in body["top_risk_factors"]:
            assert contributions[feat] > 0, f"Fator de risco '{feat}' deveria ter SHAP positivo"

    def test_explain_top_protective_factors_negative_shap(self, client, sample_patient, auth_headers):
        resp = client.post("/explain", json=sample_patient, headers=auth_headers("clinician"))
        body = resp.json()
        contributions = body["feature_contributions"]
        for feat in body["top_protective_factors"]:
            assert contributions[feat] < 0, f"Fator protetor '{feat}' deveria ter SHAP negativo"

    def test_explain_without_auth_returns_401(self, client, sample_patient):
        resp = client.post("/explain", json=sample_patient)
        assert resp.status_code == 401


class TestModelInfo:
    def test_model_info_requires_auth(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 401

    def test_model_info_viewer_allowed(self, client, auth_headers):
        resp = client.get("/model-info", headers=auth_headers("viewer"))
        assert resp.status_code == 200

    def test_model_info_contains_features(self, client, auth_headers):
        resp = client.get("/model-info", headers=auth_headers("viewer"))
        body = resp.json()
        assert "features" in body
        assert len(body["features"]) == 19
        assert "metrics" in body
        assert "model_version" in body


class TestAudit:
    def test_audit_summary_requires_admin(self, client, auth_headers):
        resp = client.get("/audit/summary", headers=auth_headers("clinician"))
        assert resp.status_code == 403

    def test_audit_summary_admin_returns_200(self, client, auth_headers):
        resp = client.get("/audit/summary", headers=auth_headers("admin"))
        assert resp.status_code == 200
        body = resp.json()
        assert "total_predicoes" in body
        assert "distribuicao_risco" in body

    def test_audit_export_admin(self, client, auth_headers):
        resp = client.get("/audit/export", headers=auth_headers("admin"))
        assert resp.status_code == 200
        assert "exported_rows" in resp.json()


class TestAdminUsers:
    def test_create_user_admin(self, client, auth_headers):
        resp = client.post("/admin/users", json={
            "username": "novo_medico",
            "password": "senha_forte_123",
            "role": "clinician",
            "full_name": "Dr. Teste",
        }, headers=auth_headers("admin"))
        assert resp.status_code == 200

    def test_create_user_clinician_forbidden(self, client, auth_headers):
        resp = client.post("/admin/users", json={
            "username": "outro",
            "password": "senha_forte_456",
            "role": "viewer",
        }, headers=auth_headers("clinician"))
        assert resp.status_code == 403
