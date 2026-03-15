"""
Testes de integridade dos artefatos do modelo.
"""

import json
import os
import sys

import joblib
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(ROOT, "model")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="module")
def dnn_model():
    import tensorflow as tf
    return tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))


@pytest.fixture(scope="module")
def xgb_model():
    return joblib.load(os.path.join(MODEL_DIR, "best_model_xgb.pkl"))


@pytest.fixture(scope="module")
def scaler():
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


@pytest.fixture(scope="module")
def feature_cols():
    return joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))


@pytest.fixture(scope="module")
def encoders():
    return joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))


@pytest.fixture(scope="module")
def ensemble_config():
    with open(os.path.join(MODEL_DIR, "metrics_ensemble.json")) as f:
        return json.load(f)


class TestArtifactLoading:
    def test_dnn_model_loads(self, dnn_model):
        assert dnn_model is not None

    def test_xgb_model_loads(self, xgb_model):
        assert xgb_model is not None
        assert hasattr(xgb_model, "predict_proba")

    def test_scaler_loads(self, scaler):
        assert scaler is not None
        assert hasattr(scaler, "transform")

    def test_feature_cols_length(self, feature_cols):
        assert len(feature_cols) == 19, f"Esperado 19 features, obtido {len(feature_cols)}"

    def test_scaler_feature_count(self, scaler):
        assert scaler.n_features_in_ == 19

    def test_encoders_keys(self, encoders):
        expected_cols = {
            "diag_primary", "hba1c_result", "glucose_serum_test",
            "insulin", "change_medications", "diabetes_medication", "gender"
        }
        assert expected_cols.issubset(set(encoders.keys()))


class TestEnsembleConfig:
    def test_required_keys(self, ensemble_config):
        for key in ["dnn_weight", "xgb_weight", "best_threshold", "roc_auc"]:
            assert key in ensemble_config, f"Chave '{key}' ausente em metrics_ensemble.json"

    def test_weights_sum_to_one(self, ensemble_config):
        total = ensemble_config["dnn_weight"] + ensemble_config["xgb_weight"]
        assert abs(total - 1.0) < 1e-6, f"Pesos devem somar 1.0, soma atual: {total}"

    def test_threshold_in_range(self, ensemble_config):
        t = ensemble_config["best_threshold"]
        assert 0.0 < t < 1.0, f"Threshold deve estar entre 0 e 1: {t}"

    def test_roc_auc_acceptable(self, ensemble_config):
        """ROC-AUC deve ser razoável para uso clínico."""
        assert ensemble_config["roc_auc"] >= 0.60, (
            f"ROC-AUC {ensemble_config['roc_auc']} abaixo do mínimo aceitável (0.60)"
        )


class TestModelInference:
    def test_dnn_output_in_range(self, dnn_model, scaler, sample_patient, encoders):
        from utils.preprocessing import build_feature_vector
        X = build_feature_vector(sample_patient, encoders)
        X_scaled = scaler.transform(X)
        pred = dnn_model.predict(X_scaled, verbose=0)
        assert 0.0 <= float(pred[0][0]) <= 1.0

    def test_xgb_output_in_range(self, xgb_model, sample_patient, encoders):
        from utils.preprocessing import build_feature_vector
        X = build_feature_vector(sample_patient, encoders)
        prob = xgb_model.predict_proba(X)[:, 1][0]
        assert 0.0 <= float(prob) <= 1.0

    def test_ensemble_output_consistent(self, dnn_model, xgb_model, scaler,
                                        ensemble_config, sample_patient, encoders):
        """Saída do ensemble deve estar dentro do range esperado."""
        from utils.preprocessing import build_feature_vector
        X = build_feature_vector(sample_patient, encoders)
        X_scaled = scaler.transform(X)
        prob_dnn = float(dnn_model.predict(X_scaled, verbose=0)[0][0])
        prob_xgb = float(xgb_model.predict_proba(X)[:, 1][0])
        prob_ensemble = (
            ensemble_config["dnn_weight"] * prob_dnn +
            ensemble_config["xgb_weight"] * prob_xgb
        )
        assert 0.0 <= prob_ensemble <= 1.0
