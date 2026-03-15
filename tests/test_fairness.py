"""
Testes de equidade (fairness) do modelo ensemble.

Verifica que o modelo não discrimina significativamente entre
subgrupos demográficos e clínicos relevantes.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(ROOT, "model")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MIN_AUC = 0.60        # AUC mínimo por subgrupo
MAX_TPR_DELTA = 0.20  # diferença máxima de TPR entre subgrupos (equalized odds)
MIN_SAMPLES = 50      # subgrupos com menos amostras são ignorados


@pytest.fixture(scope="module")
def test_data():
    """Carrega o CSV de dados e separa um conjunto de teste representativo."""
    csv_path = os.path.join(ROOT, "data", "hospital_readmission.csv")
    if not os.path.exists(csv_path):
        pytest.skip("Dataset não encontrado. Execute data/generate_data.py primeiro.")
    df = pd.read_csv(csv_path).dropna(subset=["readmitted_30days"])
    # Usa os últimos 20% como proxy de conjunto de teste
    return df.tail(int(len(df) * 0.2)).reset_index(drop=True)


@pytest.fixture(scope="module")
def predict_fn():
    """Função de predição do ensemble."""
    import tensorflow as tf
    dnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))
    xgb = joblib.load(os.path.join(MODEL_DIR, "best_model_xgb.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics_ensemble.json")) as f:
        cfg = json.load(f)
    dnn_w, xgb_w = cfg["dnn_weight"], cfg["xgb_weight"]

    from utils.preprocessing import build_feature_vector

    def _predict(row: dict) -> float:
        X = build_feature_vector(row, encoders)
        X_scaled = scaler.transform(X)
        p_dnn = float(dnn.predict(X_scaled, verbose=0)[0][0])
        p_xgb = float(xgb.predict_proba(X)[:, 1][0])
        return dnn_w * p_dnn + xgb_w * p_xgb

    return _predict


@pytest.fixture(scope="module")
def predictions(test_data, predict_fn):
    """Calcula probabilidades para todo o conjunto de teste."""
    probs = []
    cat_cols = ["diag_primary", "hba1c_result", "glucose_serum_test",
                "insulin", "change_medications", "diabetes_medication", "gender"]
    for _, row in test_data.iterrows():
        d = row.to_dict()
        # Normaliza NaN → 'None' string (compatível com preprocessing)
        for col in cat_cols:
            if pd.isna(d.get(col)):
                d[col] = "None"
        try:
            probs.append(predict_fn(d))
        except Exception:
            probs.append(0.5)  # fallback para linhas problemáticas
    return np.array(probs)


def _auc_for_subgroup(y_true, y_prob, mask):
    y_sub = y_true[mask]
    p_sub = y_prob[mask]
    if len(y_sub) < MIN_SAMPLES or y_sub.nunique() < 2:
        return None  # subgrupo muito pequeno ou sem variância
    return roc_auc_score(y_sub, p_sub)


class TestFairnessByGender:
    def test_auc_male_above_minimum(self, test_data, predictions):
        mask = test_data["gender"] == "Male"
        auc = _auc_for_subgroup(test_data["readmitted_30days"], predictions, mask)
        if auc is None:
            pytest.skip("Subgrupo insuficiente")
        assert auc >= MIN_AUC, f"AUC Masculino={auc:.3f} abaixo do mínimo {MIN_AUC}"

    def test_auc_female_above_minimum(self, test_data, predictions):
        mask = test_data["gender"] == "Female"
        auc = _auc_for_subgroup(test_data["readmitted_30days"], predictions, mask)
        if auc is None:
            pytest.skip("Subgrupo insuficiente")
        assert auc >= MIN_AUC, f"AUC Feminino={auc:.3f} abaixo do mínimo {MIN_AUC}"


class TestFairnessByAgeGroup:
    @pytest.mark.parametrize("age_min,age_max,label", [
        (0, 30, "<30"),
        (30, 50, "30-50"),
        (50, 70, "50-70"),
        (70, 101, ">70"),
    ])
    def test_auc_age_group_above_minimum(self, test_data, predictions, age_min, age_max, label):
        mask = (test_data["age_numeric"] >= age_min) & (test_data["age_numeric"] < age_max)
        auc = _auc_for_subgroup(test_data["readmitted_30days"], predictions, mask)
        if auc is None:
            pytest.skip(f"Subgrupo {label} insuficiente")
        assert auc >= MIN_AUC, f"AUC faixa {label}={auc:.3f} abaixo do mínimo {MIN_AUC}"


class TestFairnessByDiagnosis:
    @pytest.mark.parametrize("diag", [
        "Circulatory", "Respiratory", "Digestive", "Diabetes",
        "Injury", "Musculoskeletal", "Genitourinary", "Other"
    ])
    def test_auc_diagnosis_above_minimum(self, test_data, predictions, diag):
        mask = test_data["diag_primary"] == diag
        auc = _auc_for_subgroup(test_data["readmitted_30days"], predictions, mask)
        if auc is None:
            pytest.skip(f"Subgrupo {diag} insuficiente")
        assert auc >= MIN_AUC, f"AUC {diag}={auc:.3f} abaixo do mínimo {MIN_AUC}"


class TestEqualizedOdds:
    def test_tpr_gender_delta(self, test_data, predictions):
        """Diferença de TPR entre gêneros não deve exceder MAX_TPR_DELTA."""
        with open(os.path.join(MODEL_DIR, "metrics_ensemble.json")) as f:
            threshold = json.load(f)["best_threshold"]

        y = test_data["readmitted_30days"].values
        preds_bin = (predictions >= threshold).astype(int)

        tprs = {}
        for gender in ["Male", "Female"]:
            mask = (test_data["gender"] == gender).values
            y_sub = y[mask]
            p_sub = preds_bin[mask]
            if y_sub.sum() > 0:
                tprs[gender] = y_sub[p_sub == 1].sum() / y_sub.sum()

        if len(tprs) == 2:
            delta = abs(tprs["Male"] - tprs["Female"])
            assert delta <= MAX_TPR_DELTA, (
                f"Delta TPR por gênero={delta:.3f} excede máximo permitido {MAX_TPR_DELTA}. "
                f"TPR Masculino={tprs.get('Male', 'N/A'):.3f}, "
                f"TPR Feminino={tprs.get('Female', 'N/A'):.3f}"
            )
