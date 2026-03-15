"""
Análise de Equidade (Fairness) do modelo ensemble.

Verifica que o modelo não discrimina significativamente entre
subgrupos demográficos e clínicos. Requisito para submissão ao CEP
e documentação ANVISA RDC 657/2022.

Métricas calculadas por subgrupo:
  - ROC-AUC
  - Precisão, Recall, F1 (no threshold otimizado)
  - TPR — True Positive Rate (sensibilidade) — equalized odds
  - FPR — False Positive Rate — equalized odds
  - Positive Rate — paridade demográfica

Alertas:
  - AUC < MIN_AUC (0.60) → alerta de desempenho insuficiente
  - |TPR_subgrupo - TPR_global| > MAX_TPR_DELTA (0.20) → alerta de disparidade

Uso:
    python model/fairness.py

Saída:
    model/fairness_report.json
    model/fairness_report.png
"""

import json
import logging
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.preprocessing import build_feature_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MIN_AUC       = 0.60
MAX_TPR_DELTA = 0.20
MIN_SAMPLES   = 50


def load_test_split() -> tuple:
    """Carrega os últimos 20% do dataset como conjunto de teste."""
    csv_path = os.path.join(DATA_DIR, "hospital_readmission.csv")
    df = pd.read_csv(csv_path).dropna(subset=["readmitted_30days"])
    df_test = df.tail(int(len(df) * 0.2)).reset_index(drop=True)

    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    cat_cols = ["diag_primary", "hba1c_result", "glucose_serum_test",
                "insulin", "change_medications", "diabetes_medication", "gender"]

    X_rows, y_list, idx_list = [], [], []
    for i, row in df_test.iterrows():
        d = row.to_dict()
        for col in cat_cols:
            if pd.isna(d.get(col)):
                d[col] = "None"
        try:
            X_rows.append(build_feature_vector(d, encoders)[0])
            y_list.append(int(row["readmitted_30days"]))
            idx_list.append(i)
        except (ValueError, KeyError):
            continue

    X = np.array(X_rows)
    y = np.array(y_list)
    df_subset = df_test.loc[idx_list].reset_index(drop=True)
    return X, y, df_subset


def load_ensemble():
    """Carrega o ensemble e retorna função de predição."""
    dnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))
    xgb = joblib.load(os.path.join(MODEL_DIR, "best_model_xgb.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics_ensemble.json")) as f:
        cfg = json.load(f)
    dnn_w, xgb_w = cfg["dnn_weight"], cfg["xgb_weight"]

    def predict_proba(X: np.ndarray) -> np.ndarray:
        X_scaled = scaler.transform(X)
        p_dnn = dnn.predict(X_scaled, verbose=0).flatten()
        p_xgb = xgb.predict_proba(X)[:, 1]
        return dnn_w * p_dnn + xgb_w * p_xgb

    return predict_proba, cfg["best_threshold"]


def _metrics_for_subgroup(y_true: np.ndarray, probs: np.ndarray,
                           threshold: float, label: str) -> dict:
    n = len(y_true)
    if n < MIN_SAMPLES:
        return {"subgrupo": label, "n": n, "aviso": f"Amostras insuficientes (< {MIN_SAMPLES})"}

    if y_true.sum() == 0 or y_true.sum() == n:
        return {"subgrupo": label, "n": n, "aviso": "Sem variância na classe alvo"}

    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    auc = roc_auc_score(y_true, probs)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    pos_rate = preds.mean()

    return {
        "subgrupo": label,
        "n": int(n),
        "n_positivos": int(y_true.sum()),
        "roc_auc": round(float(auc), 4),
        "precisao": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "tpr": round(float(tpr), 4),
        "fpr": round(float(fpr), 4),
        "positive_rate": round(float(pos_rate), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "alerta_auc": auc < MIN_AUC,
    }


def compute_fairness_report() -> dict:
    logger.info("Carregando dados e modelos...")
    X, y, df = load_test_split()
    predict_fn, threshold = load_ensemble()

    logger.info("Calculando probabilidades para %d amostras...", len(y))
    probs = predict_fn(X)

    # Métricas globais
    global_metrics = _metrics_for_subgroup(y, probs, threshold, "Global")
    global_tpr = global_metrics.get("tpr", 0.0)

    subgroups = []
    alerts = []

    # Faixas etárias
    age_bins = [("<30", 0, 30), ("30-50", 30, 50), ("50-70", 50, 70), (">70", 70, 101)]
    for label, lo, hi in age_bins:
        mask = (df["age_numeric"] >= lo) & (df["age_numeric"] < hi)
        m = _metrics_for_subgroup(y[mask], probs[mask], threshold, f"Idade {label}")
        m["categoria"] = "idade"
        if m.get("alerta_auc"):
            alerts.append(f"⚠ AUC baixo ({m['roc_auc']:.3f}) para: {m['subgrupo']}")
        if "tpr" in m and abs(m["tpr"] - global_tpr) > MAX_TPR_DELTA:
            alerts.append(f"⚠ Disparidade de TPR ({m['tpr']:.3f} vs global {global_tpr:.3f}) para: {m['subgrupo']}")
        subgroups.append(m)

    # Gênero
    for gender in ["Male", "Female"]:
        mask = df["gender"] == gender
        m = _metrics_for_subgroup(y[mask], probs[mask], threshold, f"Gênero: {gender}")
        m["categoria"] = "genero"
        if m.get("alerta_auc"):
            alerts.append(f"⚠ AUC baixo ({m['roc_auc']:.3f}) para: {m['subgrupo']}")
        subgroups.append(m)

    # Diagnóstico primário
    diags = ["Circulatory", "Respiratory", "Digestive", "Diabetes",
             "Injury", "Musculoskeletal", "Genitourinary", "Other"]
    for diag in diags:
        mask = df["diag_primary"] == diag
        m = _metrics_for_subgroup(y[mask], probs[mask], threshold, f"Diagnóstico: {diag}")
        m["categoria"] = "diagnostico"
        if m.get("alerta_auc"):
            alerts.append(f"⚠ AUC baixo ({m['roc_auc']:.3f}) para: {m['subgrupo']}")
        subgroups.append(m)

    report = {
        "global": global_metrics,
        "subgrupos": subgroups,
        "alertas": alerts,
        "parametros": {
            "min_auc": MIN_AUC,
            "max_tpr_delta": MAX_TPR_DELTA,
            "min_amostras": MIN_SAMPLES,
            "threshold": threshold,
            "total_amostras": len(y),
        }
    }

    output_json = os.path.join(MODEL_DIR, "fairness_report.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Relatório de fairness salvo: %s", output_json)

    # Plot
    _plot_fairness(subgroups)

    if alerts:
        logger.warning("=== ALERTAS DE FAIRNESS (%d) ===", len(alerts))
        for a in alerts:
            logger.warning(a)
    else:
        logger.info("Nenhum alerta de fairness detectado.")

    return report


def _plot_fairness(subgroups: list):
    valid = [s for s in subgroups if "roc_auc" in s]
    if not valid:
        return

    df_plot = pd.DataFrame(valid)
    df_plot["cor"] = df_plot["roc_auc"].apply(
        lambda x: "salmon" if x < MIN_AUC else "steelblue"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(valid) * 0.4)))

    # AUC por subgrupo
    ax = axes[0]
    bars = ax.barh(df_plot["subgrupo"], df_plot["roc_auc"], color=df_plot["cor"])
    ax.axvline(MIN_AUC, color="red", linestyle="--", alpha=0.7, label=f"Mínimo AUC ({MIN_AUC})")
    ax.set_xlabel("ROC-AUC")
    ax.set_title("ROC-AUC por Subgrupo")
    ax.legend()
    ax.set_xlim(0.4, 1.0)
    ax.grid(axis="x", alpha=0.3)

    # TPR por subgrupo
    ax2 = axes[1]
    df_valid_tpr = df_plot.dropna(subset=["tpr"])
    ax2.barh(df_valid_tpr["subgrupo"], df_valid_tpr["tpr"], color="steelblue", alpha=0.8)
    ax2.set_xlabel("TPR (Sensibilidade)")
    ax2.set_title("Sensibilidade por Subgrupo")
    ax2.set_xlim(0.0, 1.0)
    ax2.grid(axis="x", alpha=0.3)

    plt.suptitle("Análise de Equidade (Fairness) — Hospital Readmission ML", fontsize=13)
    plt.tight_layout()
    output = os.path.join(MODEL_DIR, "fairness_report.png")
    plt.savefig(output, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("Gráfico de fairness salvo: %s", output)


if __name__ == "__main__":
    report = compute_fairness_report()
    global_m = report["global"]
    print(f"\n=== Relatório de Equidade ===")
    print(f"Global AUC: {global_m.get('roc_auc', 'N/A')}")
    print(f"Global Recall: {global_m.get('recall', 'N/A')}")
    print(f"\nSubgrupos analisados: {len(report['subgrupos'])}")
    if report["alertas"]:
        print(f"\n⚠ ALERTAS ({len(report['alertas'])}):")
        for a in report["alertas"]:
            print(f"  {a}")
    else:
        print("\n✓ Nenhum alerta de equidade detectado.")
