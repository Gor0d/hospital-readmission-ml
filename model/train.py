"""
Treinamento do modelo de predição de readmissão hospitalar com Keras.
"""

import sys
import os
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              average_precision_score)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

# Reproducibilidade
np.random.seed(42)
tf.random.set_seed(42)

ARTIFACTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ARTIFACTS_DIR, '..', 'data', 'hospital_readmission.csv')

# ──────────────────────────────────────────────
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ──────────────────────────────────────────────

print("Carregando dados...")
df = pd.read_csv(DATA_PATH)

# Feature engineering
df['risk_score'] = (
    (df['number_inpatient'] * 2) +
    df['number_emergency'] +
    (df['hba1c_result'].isin(['>7', '>8'])).astype(int) +
    (df['glucose_serum_test'].isin(['>200', '>300'])).astype(int)
)

df['medication_complexity'] = df['num_medications'] * df['num_diagnoses']
df['hospital_utilization'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

# Encoding categórico
cat_cols = ['diag_primary', 'hba1c_result', 'glucose_serum_test',
            'insulin', 'change_medications', 'diabetes_medication', 'gender']

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
    encoders[col] = le

# Features para o modelo
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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

# Normalização
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train_sc.shape} | Val: {X_val_sc.shape} | Test: {X_test_sc.shape}")

# ──────────────────────────────────────────────
# 2. ARQUITETURA DO MODELO KERAS
# ──────────────────────────────────────────────

def build_model(input_dim, dropout_rate=0.3, l2_lambda=0.001):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),

        layers.Dense(64, kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),

        layers.Dense(32, kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate / 2),

        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_model(input_dim=X_train_sc.shape[1])
model.summary()

# ──────────────────────────────────────────────
# 3. COMPILAÇÃO E TREINAMENTO
# ──────────────────────────────────────────────

# Class weights para desbalanceamento
neg, pos = np.bincount(y_train)
class_weight = {0: 1.0, 1: neg / pos}

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

cb_list = [
    callbacks.EarlyStopping(monitor='val_auc', patience=10,
                            restore_best_weights=True, mode='max'),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=5, min_lr=1e-6),
    callbacks.ModelCheckpoint(
        os.path.join(ARTIFACTS_DIR, 'best_model.keras'),
        monitor='val_auc', save_best_only=True, mode='max'
    )
]

print("\nTreinando modelo...")
history = model.fit(
    X_train_sc, y_train,
    validation_data=(X_val_sc, y_val),
    epochs=100,
    batch_size=256,
    class_weight=class_weight,
    callbacks=cb_list,
    verbose=1
)

# ──────────────────────────────────────────────
# 4. AVALIAÇÃO
# ──────────────────────────────────────────────

print("\nAvaliando no conjunto de teste...")
y_pred_prob = model.predict(X_test_sc).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_prob)
ap  = average_precision_score(y_test, y_pred_prob)

print(f"\nROC-AUC:  {auc:.4f}")
print(f"Avg Precision: {ap:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Não Readmitido', 'Readmitido']))

# ──────────────────────────────────────────────
# 5. VISUALIZAÇÕES
# ──────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Predição de Readmissão Hospitalar — Resultados do Modelo',
             fontsize=14, fontweight='bold')

# Loss
axes[0,0].plot(history.history['loss'], label='Treino', color='#1F5C99')
axes[0,0].plot(history.history['val_loss'], label='Validação', color='#E94560')
axes[0,0].set_title('Loss por Época')
axes[0,0].set_xlabel('Época')
axes[0,0].set_ylabel('Binary Crossentropy')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

# AUC
axes[0,1].plot(history.history['auc'], label='Treino', color='#1F5C99')
axes[0,1].plot(history.history['val_auc'], label='Validação', color='#E94560')
axes[0,1].set_title('AUC por Época')
axes[0,1].set_xlabel('Época')
axes[0,1].set_ylabel('AUC')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1,0].plot(fpr, tpr, color='#1F5C99', lw=2, label=f'ROC (AUC = {auc:.3f})')
axes[1,0].plot([0,1], [0,1], 'k--', alpha=0.5)
axes[1,0].set_title('Curva ROC')
axes[1,0].set_xlabel('Taxa de Falso Positivo')
axes[1,0].set_ylabel('Taxa de Verdadeiro Positivo')
axes[1,0].legend()
axes[1,0].grid(alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[1,1], cmap='Blues',
            xticklabels=['Não Readmitido', 'Readmitido'],
            yticklabels=['Não Readmitido', 'Readmitido'])
axes[1,1].set_title('Matriz de Confusão')
axes[1,1].set_ylabel('Real')
axes[1,1].set_xlabel('Predito')

plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, 'model_results.png'), dpi=150, bbox_inches='tight')
print("\nVisualizações salvas em model_results.png")

# ──────────────────────────────────────────────
# 6. SALVAR ARTEFATOS
# ──────────────────────────────────────────────

joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
joblib.dump(encoders, os.path.join(ARTIFACTS_DIR, 'encoders.pkl'))
joblib.dump(feature_cols, os.path.join(ARTIFACTS_DIR, 'feature_cols.pkl'))

metrics = {
    'roc_auc': round(auc, 4),
    'average_precision': round(ap, 4),
    'test_samples': int(len(y_test)),
    'feature_count': len(feature_cols)
}
with open(os.path.join(ARTIFACTS_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nArtefatos salvos:")
print("  - best_model.keras")
print("  - scaler.pkl")
print("  - encoders.pkl")
print("  - feature_cols.pkl")
print("  - metrics.json")
print("  - model_results.png")
print(f"\nROC-AUC final: {auc:.4f}")
