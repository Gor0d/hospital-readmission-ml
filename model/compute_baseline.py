"""
Computa estatísticas baseline do modelo para monitoramento de deriva (PSI).

Deve ser executado APÓS o treinamento e ANTES de colocar o modelo em produção.
Gera:
  - model/baseline_stats.json  — distribuições de treino para PSI
  - model/checksums.sha256     — hashes SHA-256 dos artefatos

Uso:
    python model/compute_baseline.py
"""

import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.preprocessing import build_feature_vector
from api.monitoring import generate_checksums

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def compute_baseline():
    logger.info("Carregando artefatos...")
    dnn    = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))
    xgb    = joblib.load(os.path.join(MODEL_DIR, "best_model_xgb.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics_ensemble.json")) as f:
        cfg = json.load(f)
    dnn_w, xgb_w = cfg["dnn_weight"], cfg["xgb_weight"]

    logger.info("Carregando dataset de treino...")
    csv_path = os.path.join(DATA_DIR, "hospital_readmission.csv")
    df = pd.read_csv(csv_path).dropna(subset=["readmitted_30days"])
    # Usa os primeiros 80% como dados de treino (espelho do split original)
    df_train = df.head(int(len(df) * 0.8)).reset_index(drop=True)

    cat_cols = ["diag_primary", "hba1c_result", "glucose_serum_test",
                "insulin", "change_medications", "diabetes_medication", "gender"]

    X_rows = []
    for _, row in df_train.iterrows():
        d = row.to_dict()
        for col in cat_cols:
            if pd.isna(d.get(col)):
                d[col] = "None"
        try:
            X_rows.append(build_feature_vector(d, encoders)[0])
        except (ValueError, KeyError):
            continue

    X = np.array(X_rows)
    logger.info("Amostras de treino processadas: %d", len(X))

    # Probabilidades preditas no treino (baseline da distribuição)
    X_scaled = scaler.transform(X)
    p_dnn = dnn.predict(X_scaled, verbose=0).flatten()
    p_xgb = xgb.predict_proba(X)[:, 1]
    probs = dnn_w * p_dnn + xgb_w * p_xgb

    # Estatísticas por feature (para PSI por feature no futuro)
    feature_stats = {}
    for i, feat in enumerate(feature_cols):
        col_data = X[:, i]
        feature_stats[feat] = {
            "mean": float(np.mean(col_data)),
            "std":  float(np.std(col_data)),
            "min":  float(np.min(col_data)),
            "max":  float(np.max(col_data)),
            "p25":  float(np.percentile(col_data, 25)),
            "p50":  float(np.percentile(col_data, 50)),
            "p75":  float(np.percentile(col_data, 75)),
        }

    baseline_stats = {
        "n_samples": len(X),
        "prediction_distribution": probs.tolist(),
        "prediction_mean": float(probs.mean()),
        "prediction_std": float(probs.std()),
        "positive_rate": float((probs >= cfg["best_threshold"]).mean()),
        "feature_stats": feature_stats,
        "model_version": cfg.get("model_version", "1.0.0"),
    }

    output_path = os.path.join(MODEL_DIR, "baseline_stats.json")
    with open(output_path, "w") as f:
        json.dump(baseline_stats, f, indent=2)
    logger.info("Baseline salvo: %s (%d amostras)", output_path, len(X))

    # Gera checksums
    logger.info("Gerando checksums dos artefatos...")
    checksums = generate_checksums(MODEL_DIR)
    logger.info("Checksums gerados para %d artefatos.", len(checksums))

    print("\n=== Baseline Computado ===")
    print(f"  Amostras: {len(X)}")
    print(f"  Probabilidade média: {baseline_stats['prediction_mean']:.4f}")
    print(f"  Taxa de positivos (threshold={cfg['best_threshold']}): {baseline_stats['positive_rate']:.4f}")
    print(f"  Artefatos com checksums: {list(checksums.keys())}")


if __name__ == "__main__":
    compute_baseline()
