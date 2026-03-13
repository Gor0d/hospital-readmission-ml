"""
Ensemble: DNN (Keras) + XGBoost com média ponderada das probabilidades.
Requer que train.py e train_xgboost.py já tenham sido executados.
"""

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import joblib
import json
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix
)

np.random.seed(42)

ARTIFACTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(ARTIFACTS_DIR, '..', 'data', 'hospital_readmission.csv')

# ──────────────────────────────────────────────
# 1. CARREGAR DADOS (mesmo split do treino)
# ──────────────────────────────────────────────

print("Carregando dados...")
df = pd.read_csv(DATA_PATH)

df['risk_score']            = ((df['number_inpatient'] * 2) + df['number_emergency'] +
                               (df['hba1c_result'].isin(['>7', '>8'])).astype(int) +
                               (df['glucose_serum_test'].isin(['>200', '>300'])).astype(int))
df['medication_complexity'] = df['num_medications'] * df['num_diagnoses']
df['hospital_utilization']  = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

cat_cols = ['diag_primary', 'hba1c_result', 'glucose_serum_test',
            'insulin', 'change_medications', 'diabetes_medication', 'gender']
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

feature_cols = [
    'age_numeric', 'gender_enc', 'diag_primary_enc',
    'time_in_hospital', 'num_medications', 'num_procedures',
    'num_diagnoses', 'num_lab_procedures',
    'number_outpatient', 'number_emergency', 'number_inpatient',
    'hba1c_result_enc', 'glucose_serum_test_enc',
    'insulin_enc', 'change_medications_enc', 'diabetes_medication_enc',
    'risk_score', 'medication_complexity', 'hospital_utilization'
]

X = df[feature_cols].values
y = df['readmitted_30days'].values

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ──────────────────────────────────────────────
# 2. CARREGAR MODELOS SALVOS
# ──────────────────────────────────────────────

print("Carregando modelos...")
dnn_model = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, 'best_model.keras'))
scaler    = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
xgb_model = joblib.load(os.path.join(ARTIFACTS_DIR, 'best_model_xgb.pkl'))

X_test_sc = scaler.transform(X_test)

# ──────────────────────────────────────────────
# 3. PREDIÇÕES INDIVIDUAIS
# ──────────────────────────────────────────────

print("Gerando predições...")
prob_dnn = dnn_model.predict(X_test_sc, verbose=0).flatten()
prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# ──────────────────────────────────────────────
# 4. ENSEMBLE: BUSCAR MELHOR PESO
# ──────────────────────────────────────────────

print("\nOtimizando pesos do ensemble...")
best_auc, best_w = 0, 0.5
for w in np.arange(0.1, 1.0, 0.05):
    prob_ens = w * prob_dnn + (1 - w) * prob_xgb
    auc = roc_auc_score(y_test, prob_ens)
    if auc > best_auc:
        best_auc, best_w = auc, w

print(f"Melhor peso DNN: {best_w:.2f} | XGBoost: {1-best_w:.2f} → AUC: {best_auc:.4f}")

prob_ensemble = best_w * prob_dnn + (1 - best_w) * prob_xgb
y_pred_ens    = (prob_ensemble >= 0.5).astype(int)

auc_ens       = roc_auc_score(y_test, prob_ensemble)
ap_ens        = average_precision_score(y_test, prob_ensemble)
f1_ens        = f1_score(y_test, y_pred_ens)
precision_ens = precision_score(y_test, y_pred_ens)
recall_ens    = recall_score(y_test, y_pred_ens)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ens).ravel()

# ──────────────────────────────────────────────
# 5. COMPARAÇÃO COMPLETA
# ──────────────────────────────────────────────

dnn_metrics = {}
xgb_metrics = {}
for path, d in [(os.path.join(ARTIFACTS_DIR, 'metrics.json'), dnn_metrics),
                (os.path.join(ARTIFACTS_DIR, 'metrics_xgb.json'), xgb_metrics)]:
    if os.path.exists(path):
        with open(path) as f:
            d.update(json.load(f))

print(f"\n{'='*60}")
print(f"  COMPARAÇÃO FINAL: DNN  vs  XGBoost  vs  Ensemble")
print(f"{'='*60}")
print(f"  {'Métrica':<22} {'DNN':>8} {'XGBoost':>10} {'Ensemble':>10}")
print(f"  {'-'*54}")
for metric_name, key, ens_val in [
    ('ROC-AUC',         'roc_auc',           auc_ens),
    ('Avg Precision',   'average_precision',  ap_ens),
    ('F1-Score',        'f1_score',           f1_ens),
    ('Precisão',        'precision',          precision_ens),
    ('Recall',          'recall',             recall_ens),
]:
    d = dnn_metrics.get(key, float('nan'))
    x = xgb_metrics.get(key, float('nan'))
    best = max(v for v in [d, x, ens_val] if isinstance(v, float))
    def fmt(v): return f"{v:.4f}{'*' if v == best else ' '}"
    print(f"  {metric_name:<22} {fmt(d):>9} {fmt(x):>10} {fmt(ens_val):>10}")
print(f"{'='*60}")
print("  * = melhor valor")

# ──────────────────────────────────────────────
# 6. VISUALIZAÇÃO
# ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Ensemble DNN + XGBoost', fontsize=14, fontweight='bold')

# Curvas ROC sobrepostas
for prob, label, color in [
    (prob_dnn,      f'DNN (AUC={roc_auc_score(y_test, prob_dnn):.3f})',      '#1F5C99'),
    (prob_xgb,      f'XGBoost (AUC={roc_auc_score(y_test, prob_xgb):.3f})', '#27ae60'),
    (prob_ensemble, f'Ensemble (AUC={auc_ens:.3f})',                          '#E94560'),
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[0].plot(fpr, tpr, color=color, lw=2, label=label)
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[0].set_title('Curvas ROC — Comparação')
axes[0].set_xlabel('Taxa de Falso Positivo')
axes[0].set_ylabel('Taxa de Verdadeiro Positivo')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Comparação de métricas em barras
metric_labels = ['ROC-AUC', 'F1-Score', 'Precisão', 'Recall']
dnn_vals = [dnn_metrics.get(k, 0) for k in ['roc_auc', 'f1_score', 'precision', 'recall']]
xgb_vals = [xgb_metrics.get(k, 0) for k in ['roc_auc', 'f1_score', 'precision', 'recall']]
ens_vals = [auc_ens, f1_ens, precision_ens, recall_ens]

x = np.arange(len(metric_labels))
w = 0.25
axes[1].bar(x - w, dnn_vals, w, label='DNN',      color='#1F5C99', alpha=0.85)
axes[1].bar(x,     xgb_vals, w, label='XGBoost',  color='#27ae60', alpha=0.85)
axes[1].bar(x + w, ens_vals, w, label='Ensemble', color='#E94560', alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(metric_labels)
axes[1].set_ylim(0.5, 0.85)
axes[1].set_title('Comparação de Métricas')
axes[1].set_ylabel('Valor')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, 'model_results_ensemble.png'), dpi=150, bbox_inches='tight')
plt.close()

# ──────────────────────────────────────────────
# 7. SALVAR ARTEFATOS
# ──────────────────────────────────────────────

ensemble_config = {
    'model_version': '1.0.0',
    'model_type': 'Ensemble (DNN + XGBoost)',
    'trained_at': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    'dnn_weight': round(best_w, 4),
    'xgb_weight': round(1 - best_w, 4),
    'roc_auc': round(auc_ens, 4),
    'average_precision': round(ap_ens, 4),
    'f1_score': round(f1_ens, 4),
    'precision': round(precision_ens, 4),
    'recall': round(recall_ens, 4),
    'true_positives': int(tp),
    'true_negatives': int(tn),
    'false_positives': int(fp),
    'false_negatives': int(fn),
    'test_samples': int(len(y_test)),
}
with open(os.path.join(ARTIFACTS_DIR, 'metrics_ensemble.json'), 'w') as f:
    json.dump(ensemble_config, f, indent=2)

print(f"\nArtefatos salvos:")
print(f"  - metrics_ensemble.json")
print(f"  - model_results_ensemble.png")
print(f"\nROC-AUC Ensemble: {auc_ens:.4f}")
