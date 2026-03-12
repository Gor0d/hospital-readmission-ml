"""
Treinamento e comparação de modelos para predição de readmissão hospitalar.
Treina XGBoost e compara com a DNN (Keras) salva anteriormente.
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier

np.random.seed(42)

ARTIFACTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(ARTIFACTS_DIR, '..', 'data', 'hospital_readmission.csv')

# ──────────────────────────────────────────────
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ──────────────────────────────────────────────

print("Carregando dados...")
df = pd.read_csv(DATA_PATH)

# Feature engineering (idêntico ao train.py)
df['risk_score'] = (
    (df['number_inpatient'] * 2) +
    df['number_emergency'] +
    (df['hba1c_result'].isin(['>7', '>8'])).astype(int) +
    (df['glucose_serum_test'].isin(['>200', '>300'])).astype(int)
)
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ──────────────────────────────────────────────
# 2. TREINAMENTO XGBOOST
# ──────────────────────────────────────────────

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos  # equivalente ao class_weight da DNN

print("\nTreinando XGBoost...")
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print(f"Melhor iteração: {xgb.best_iteration}")

# ──────────────────────────────────────────────
# 3. AVALIAÇÃO
# ──────────────────────────────────────────────

print("\nAvaliando XGBoost no conjunto de teste...")
y_pred_prob_xgb = xgb.predict_proba(X_test)[:, 1]
y_pred_xgb      = (y_pred_prob_xgb >= 0.5).astype(int)

auc_xgb       = roc_auc_score(y_test, y_pred_prob_xgb)
ap_xgb        = average_precision_score(y_test, y_pred_prob_xgb)
f1_xgb        = f1_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb    = recall_score(y_test, y_pred_xgb)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_xgb).ravel()

print(f"\n{'='*40}")
print("  RESULTADOS — XGBoost")
print(f"{'='*40}")
print(f"  ROC-AUC:       {auc_xgb:.4f}")
print(f"  Avg Precision: {ap_xgb:.4f}")
print(f"  F1-Score:      {f1_xgb:.4f}")
print(f"  Precision:     {precision_xgb:.4f}")
print(f"  Recall:        {recall_xgb:.4f}")
print(f"{'='*40}")
print(classification_report(y_test, y_pred_xgb,
      target_names=['Não Readmitido', 'Readmitido']))

# ──────────────────────────────────────────────
# 4. COMPARAÇÃO COM A DNN
# ──────────────────────────────────────────────

dnn_metrics_path = os.path.join(ARTIFACTS_DIR, 'metrics.json')
dnn_metrics = {}
if os.path.exists(dnn_metrics_path):
    with open(dnn_metrics_path) as f:
        dnn_metrics = json.load(f)

print(f"\n{'='*50}")
print(f"  COMPARAÇÃO: DNN (Keras)  vs  XGBoost")
print(f"{'='*50}")
print(f"  {'Métrica':<22} {'DNN':>8} {'XGBoost':>10}")
print(f"  {'-'*42}")
for metric, xgb_val in [
    ('ROC-AUC',         auc_xgb),
    ('Average Precision', ap_xgb),
    ('F1-Score',        f1_xgb),
    ('Precisão',        precision_xgb),
    ('Recall',          recall_xgb),
]:
    dnn_val = dnn_metrics.get({
        'ROC-AUC': 'roc_auc',
        'Average Precision': 'average_precision',
        'F1-Score': 'f1_score',
        'Precisão': 'precision',
        'Recall': 'recall',
    }[metric], float('nan'))
    delta = xgb_val - dnn_val if isinstance(dnn_val, float) else 0
    flag  = '↑' if delta > 0 else ('↓' if delta < 0 else '=')
    print(f"  {metric:<22} {dnn_val:>8.4f} {xgb_val:>10.4f}  {flag} {abs(delta):.4f}")
print(f"{'='*50}")

# ──────────────────────────────────────────────
# 5. VISUALIZAÇÕES
# ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('XGBoost — Predição de Readmissão Hospitalar', fontsize=14, fontweight='bold')

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob_xgb)
axes[0].plot(fpr, tpr, color='#1F5C99', lw=2, label=f'XGBoost (AUC={auc_xgb:.3f})')
if dnn_metrics.get('roc_auc'):
    axes[0].axhline(dnn_metrics['roc_auc'], color='#E94560', linestyle='--',
                    alpha=0.6, label=f"DNN AUC={dnn_metrics['roc_auc']:.3f}")
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[0].set_title('Curva ROC')
axes[0].set_xlabel('Taxa de Falso Positivo')
axes[0].set_ylabel('Taxa de Verdadeiro Positivo')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[1], cmap='Blues',
            xticklabels=['Não Readmitido', 'Readmitido'],
            yticklabels=['Não Readmitido', 'Readmitido'])
axes[1].set_title('Matriz de Confusão')
axes[1].set_ylabel('Real')
axes[1].set_xlabel('Predito')

# Feature Importance (top 15)
importance = pd.Series(xgb.feature_importances_, index=feature_cols).nlargest(15)
importance.sort_values().plot(kind='barh', ax=axes[2], color='#1F5C99')
axes[2].set_title('Top 15 Features — Importância (XGBoost)')
axes[2].set_xlabel('Importância')

plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, 'model_results_xgb.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nVisualizações salvas em model_results_xgb.png")

# ──────────────────────────────────────────────
# 6. SALVAR ARTEFATOS
# ──────────────────────────────────────────────

joblib.dump(xgb, os.path.join(ARTIFACTS_DIR, 'best_model_xgb.pkl'))

metrics_xgb = {
    'model_version': '1.0.0',
    'model_type': 'XGBoost',
    'trained_at': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    'best_iteration': int(xgb.best_iteration),
    'roc_auc': round(auc_xgb, 4),
    'average_precision': round(ap_xgb, 4),
    'f1_score': round(f1_xgb, 4),
    'precision': round(precision_xgb, 4),
    'recall': round(recall_xgb, 4),
    'true_positives': int(tp),
    'true_negatives': int(tn),
    'false_positives': int(fp),
    'false_negatives': int(fn),
    'test_samples': int(len(y_test)),
    'feature_count': len(feature_cols),
}
with open(os.path.join(ARTIFACTS_DIR, 'metrics_xgb.json'), 'w') as f:
    json.dump(metrics_xgb, f, indent=2)

print("\nArtefatos salvos:")
print("  - best_model_xgb.pkl")
print("  - metrics_xgb.json")
print("  - model_results_xgb.png")
print(f"\nROC-AUC XGBoost: {auc_xgb:.4f}")
if dnn_metrics.get('roc_auc'):
    delta = auc_xgb - dnn_metrics['roc_auc']
    print(f"ROC-AUC DNN:     {dnn_metrics['roc_auc']:.4f}")
    print(f"Ganho:           {delta:+.4f} ({delta/dnn_metrics['roc_auc']*100:+.1f}%)")
