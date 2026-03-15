"""
Calibração do modelo ensemble para uso clínico.

Problema: a saída bruta do ensemble (média ponderada de DNN sigmoid + XGBoost)
não é necessariamente uma probabilidade calibrada. Uma saída de 0.72 deve
significar que ~72% dos pacientes com esse score serão readmitidos.

Solução: Regressão isotônica (CalibratedClassifierCV, method='isotonic').
Não-paramétrica, sem suposição de forma, superior a Platt (sigmoid) para n > 1000.

Uso:
    python model/calibrate.py

Saída:
    model/calibrated_isotonic.pkl  — artefato para uso na API
    model/calibration_metrics.json — Brier score antes/depois
    model/calibration_curve.png    — curva de confiabilidade
"""

import json
import logging
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.preprocessing import build_feature_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Classes no nível do módulo (necessário para serialização com joblib)
# ──────────────────────────────────────────────

class CalibratedEnsemble:
    """Ensemble calibrado via regressão isotônica. Serializável com joblib."""
    def __init__(self, base_estimator, calibrator):
        self.base_estimator = base_estimator
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.base_estimator.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])


class EnsembleEstimator(BaseEstimator, ClassifierMixin):
    """
    Wrapper sklearn-compatível para o ensemble DNN + XGBoost.
    Necessário para usar CalibratedClassifierCV.
    """
    def __init__(self, dnn, xgb_model, scaler, dnn_weight=0.55, xgb_weight=0.45):
        self.dnn = dnn
        self.xgb_model = xgb_model
        self.scaler = scaler
        self.dnn_weight = dnn_weight
        self.xgb_weight = xgb_weight
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        """Não re-treina — modelos já foram treinados."""
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        p_dnn = self.dnn.predict(X_scaled, verbose=0).flatten()
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        p_ensemble = self.dnn_weight * p_dnn + self.xgb_weight * p_xgb
        return np.column_stack([1 - p_ensemble, p_ensemble])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ──────────────────────────────────────────────
# Calibração
# ──────────────────────────────────────────────

def load_data():
    """Carrega o dataset e prepara features."""
    csv_path = os.path.join(DATA_DIR, "hospital_readmission.csv")
    df = pd.read_csv(csv_path).dropna(subset=["readmitted_30days"])
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))

    cat_cols = ["diag_primary", "hba1c_result", "glucose_serum_test",
                "insulin", "change_medications", "diabetes_medication", "gender"]

    X_rows = []
    y_list = []
    for _, row in df.iterrows():
        d = row.to_dict()
        for col in cat_cols:
            if pd.isna(d.get(col)):
                d[col] = "None"
        try:
            X_rows.append(build_feature_vector(d, encoders)[0])
            y_list.append(int(row["readmitted_30days"]))
        except (ValueError, KeyError):
            continue

    return np.array(X_rows), np.array(y_list)


def calibrate_model(method: str = "isotonic") -> dict:
    """
    Treina calibrador sobre um split de calibração dedicado.
    Retorna métricas de qualidade da calibração.
    """
    logger.info("Carregando artefatos do modelo...")
    dnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))
    xgb = joblib.load(os.path.join(MODEL_DIR, "best_model_xgb.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics_ensemble.json")) as f:
        cfg = json.load(f)

    logger.info("Carregando dados...")
    X, y = load_data()

    # Split: 65% treino (já usado), 15% calibração, 20% teste
    # Usamos stratify para preservar proporção de classes
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_calib, _, y_calib, _ = train_test_split(
        X_trainval, y_trainval, test_size=0.1875, random_state=42, stratify=y_trainval
    )  # 0.1875 * 0.80 ≈ 15% do total

    logger.info("Tamanhos — calibração: %d | teste: %d", len(y_calib), len(y_test))

    estimator = EnsembleEstimator(dnn, xgb, scaler, cfg["dnn_weight"], cfg["xgb_weight"])

    # Probabilidades brutas antes da calibração
    prob_raw_calib = estimator.predict_proba(X_calib)[:, 1]
    prob_raw_test  = estimator.predict_proba(X_test)[:, 1]

    brier_before = brier_score_loss(y_test, prob_raw_test)
    logger.info("Brier Score antes: %.4f", brier_before)

    # Calibração com regressão isotônica diretamente sobre as probabilidades brutas
    # (cv="prefit" foi removido no sklearn 1.6+; usamos IsotonicRegression diretamente)
    logger.info("Calibrando com método: %s", method)
    prob_raw_calib_full = estimator.predict_proba(X_calib)[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(prob_raw_calib_full, y_calib)

    calibrated = CalibratedEnsemble(estimator, isotonic)
    prob_cal_test = calibrated.predict_proba(X_test)[:, 1]
    brier_after = brier_score_loss(y_test, prob_cal_test)
    logger.info("Brier Score depois: %.4f", brier_after)

    # Salva artefato
    output_path = os.path.join(MODEL_DIR, "calibrated_isotonic.pkl")
    joblib.dump(calibrated, output_path)
    logger.info("Calibrador salvo: %s", output_path)

    # Métricas
    metrics = {
        "method": method,
        "brier_score_before": round(float(brier_before), 6),
        "brier_score_after": round(float(brier_after), 6),
        "brier_improvement": round(float(brier_before - brier_after), 6),
        "calibration_samples": len(y_calib),
        "test_samples": len(y_test),
    }
    with open(os.path.join(MODEL_DIR, "calibration_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Curva de calibração
    _plot_calibration_curves(y_test, prob_raw_test, prob_cal_test)

    return metrics


def _plot_calibration_curves(y_true, prob_raw, prob_cal):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, probs, label, color in [
        (axes[0], prob_raw, "Antes da calibração", "steelblue"),
        (axes[1], prob_cal, "Após calibração isotônica", "darkorange"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", color=color, label=label)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Calibração perfeita")
        ax.set_xlabel("Probabilidade prevista")
        ax.set_ylabel("Fração de positivos reais")
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Curvas de Calibração — Ensemble Hospital Readmission")
    plt.tight_layout()
    output = os.path.join(MODEL_DIR, "calibration_curve.png")
    plt.savefig(output, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("Curva de calibração salva: %s", output)


if __name__ == "__main__":
    metrics = calibrate_model()
    print("\n=== Resultados da Calibração ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nPara ativar o modelo calibrado na API: USE_CALIBRATED_MODEL=true no .env")
