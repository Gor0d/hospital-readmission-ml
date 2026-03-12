"""
Dashboard interativo para predição de readmissão hospitalar.
Executa localmente usando Streamlit + modelo Keras salvo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import joblib
import json
import os
import tensorflow as tf

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'hospital_readmission.csv')

st.set_page_config(
    page_title="Readmissão Hospitalar | ML Dashboard",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model        = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best_model.keras'))
    scaler       = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    encoders     = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))
    with open(os.path.join(MODEL_DIR, 'metrics.json')) as f:
        metrics = json.load(f)
    return model, scaler, encoders, feature_cols, metrics

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model, scaler, encoders, feature_cols, metrics = load_artifacts()
df = load_data()

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def preprocess_input(inputs: dict) -> np.ndarray:
    cat_cols = ['diag_primary', 'hba1c_result', 'glucose_serum_test',
                'insulin', 'change_medications', 'diabetes_medication', 'gender']
    for col in cat_cols:
        le = encoders[col]
        inputs[col + '_enc'] = int(le.transform([inputs[col]])[0])

    risk_score = (
        inputs['number_inpatient'] * 2 +
        inputs['number_emergency'] +
        (1 if inputs['hba1c_result'] in ['>7', '>8'] else 0) +
        (1 if inputs['glucose_serum_test'] in ['>200', '>300'] else 0)
    )
    row = [
        inputs['age_numeric'], inputs['gender_enc'], inputs['diag_primary_enc'],
        inputs['time_in_hospital'], inputs['num_medications'], inputs['num_procedures'],
        inputs['num_diagnoses'], inputs['num_lab_procedures'],
        inputs['number_outpatient'], inputs['number_emergency'], inputs['number_inpatient'],
        inputs['hba1c_result_enc'], inputs['glucose_serum_test_enc'],
        inputs['insulin_enc'], inputs['change_medications_enc'],
        inputs['diabetes_medication_enc'],
        risk_score,
        inputs['num_medications'] * inputs['num_diagnoses'],
        inputs['number_outpatient'] + inputs['number_emergency'] + inputs['number_inpatient']
    ]
    return np.array([row], dtype=float)

# ──────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────

st.title("🏥 Predição de Readmissão Hospitalar")
st.markdown("**Modelo de Machine Learning (Keras)** para estimativa de risco de readmissão em 30 dias")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔮 Predição Individual", "📊 Análise do Dataset", "📈 Performance do Modelo"])

# ══════════════════════════════════════
# TAB 1: Predição
# ══════════════════════════════════════
with tab1:
    st.subheader("Dados do Paciente")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Perfil**")
        age      = st.slider("Idade", 0, 100, 65)
        gender   = st.selectbox("Gênero", ["Male", "Female"])
        diag     = st.selectbox("Diagnóstico Principal", [
            "Circulatory", "Respiratory", "Digestive", "Diabetes",
            "Injury", "Musculoskeletal", "Genitourinary", "Other"
        ])

    with col2:
        st.markdown("**Internação**")
        time_hosp  = st.slider("Dias no Hospital", 1, 14, 4)
        num_meds   = st.slider("N° de Medicamentos", 1, 40, 15)
        num_proc   = st.slider("N° de Procedimentos", 0, 6, 2)
        num_diag   = st.slider("N° de Diagnósticos", 1, 16, 7)
        num_lab    = st.slider("N° de Exames Laboratoriais", 1, 120, 43)

    with col3:
        st.markdown("**Histórico & Exames**")
        n_out  = st.slider("Visitas Ambulatoriais (último ano)", 0, 5, 0)
        n_em   = st.slider("Visitas Emergência (último ano)", 0, 4, 0)
        n_in   = st.slider("Internações Anteriores (último ano)", 0, 5, 1)
        hba1c  = st.selectbox("HbA1c", ["None", "Normal", ">7", ">8"])
        gluc   = st.selectbox("Glicose Sérica", ["None", "Normal", ">200", ">300"])
        ins    = st.selectbox("Insulina", ["No", "Steady", "Up", "Down"])
        chg    = st.selectbox("Mudança de Medicamentos", ["No", "Ch"])
        diab   = st.selectbox("Medicação para Diabetes", ["Yes", "No"])

    if st.button("🔮 Calcular Risco de Readmissão", type="primary", use_container_width=True):
        inputs = {
            'age_numeric': age, 'gender': gender, 'diag_primary': diag,
            'time_in_hospital': time_hosp, 'num_medications': num_meds,
            'num_procedures': num_proc, 'num_diagnoses': num_diag,
            'num_lab_procedures': num_lab, 'number_outpatient': n_out,
            'number_emergency': n_em, 'number_inpatient': n_in,
            'hba1c_result': hba1c, 'glucose_serum_test': gluc,
            'insulin': ins, 'change_medications': chg, 'diabetes_medication': diab
        }
        X = preprocess_input(inputs)
        X_sc = scaler.transform(X)
        prob = float(model.predict(X_sc, verbose=0)[0][0])

        st.divider()
        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("Probabilidade de Readmissão", f"{prob:.1%}")
        with r2:
            if prob < 0.35:
                st.metric("Nível de Risco", "🟢 BAIXO")
            elif prob < 0.60:
                st.metric("Nível de Risco", "🟡 MODERADO")
            else:
                st.metric("Nível de Risco", "🔴 ALTO")
        with r3:
            st.metric("ROC-AUC do Modelo", f"{metrics['roc_auc']:.4f}")

        # Gauge
        fig, ax = plt.subplots(figsize=(5, 2.5))
        colors = ['#2ecc71' if i/100 < 0.35 else '#f39c12' if i/100 < 0.60 else '#e74c3c' for i in range(100)]
        ax.barh(0, 100, height=0.4, color='#ecf0f1')
        ax.barh(0, prob*100, height=0.4,
                color='#2ecc71' if prob<0.35 else '#f39c12' if prob<0.60 else '#e74c3c')
        ax.axvline(prob*100, color='#2c3e50', linewidth=2)
        ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel("Probabilidade (%)")
        ax.set_title(f"Risco estimado: {prob:.1%}", fontweight='bold')
        ax.axvline(35, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axvline(60, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        st.pyplot(fig, use_container_width=True)

        if prob < 0.35:
            st.success("**Recomendação:** Paciente com baixo risco. Alta padrão com orientações de retorno em 30 dias.")
        elif prob < 0.60:
            st.warning("**Recomendação:** Risco moderado. Acompanhamento ambulatorial em 14 dias e revisão de medicamentos.")
        else:
            st.error("**Recomendação:** Alto risco de readmissão. Plano de alta reforçado, contato ativo em 7 dias e revisão da medicação.")

# ══════════════════════════════════════
# TAB 2: Análise
# ══════════════════════════════════════
with tab2:
    st.subheader("Análise Exploratória do Dataset")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total de Pacientes", f"{len(df):,}")
    m2.metric("Taxa de Readmissão", f"{df['readmitted_30days'].mean():.1%}")
    m3.metric("Média de Medicamentos", f"{df['num_medications'].mean():.1f}")
    m4.metric("Média de Dias Internado", f"{df['time_in_hospital'].mean():.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Readmissão por diagnóstico
    ct = df.groupby('diag_primary')['readmitted_30days'].mean().sort_values(ascending=True)
    ct.plot(kind='barh', ax=axes[0], color='#1F5C99')
    axes[0].set_title('Taxa de Readmissão por\nDiagnóstico Principal')
    axes[0].set_xlabel('Taxa de Readmissão')

    # Readmissão por faixa etária
    df['age_group'] = pd.cut(df['age_numeric'], bins=[0,30,50,70,100],
                              labels=['<30', '30-50', '50-70', '>70'])
    ag = df.groupby('age_group', observed=True)['readmitted_30days'].mean()
    ag.plot(kind='bar', ax=axes[1], color='#E94560', rot=0)
    axes[1].set_title('Taxa de Readmissão\npor Faixa Etária')
    axes[1].set_ylabel('Taxa de Readmissão')

    # Distribuição HbA1c
    hba = df.groupby('hba1c_result')['readmitted_30days'].mean()
    hba.plot(kind='bar', ax=axes[2], color='#27ae60', rot=0)
    axes[2].set_title('Taxa de Readmissão\npor Resultado HbA1c')
    axes[2].set_ylabel('Taxa de Readmissão')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Distribuição das Internações Anteriores")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    readm = df[df['readmitted_30days']==1]['number_inpatient']
    no_readm = df[df['readmitted_30days']==0]['number_inpatient']
    ax2.hist(readm, bins=6, alpha=0.6, label='Readmitido', color='#e74c3c')
    ax2.hist(no_readm, bins=6, alpha=0.6, label='Não Readmitido', color='#2ecc71')
    ax2.set_xlabel('N° de Internações Anteriores')
    ax2.set_ylabel('Contagem')
    ax2.legend()
    ax2.set_title('Distribuição de Internações Anteriores por Desfecho')
    st.pyplot(fig2, use_container_width=True)

# ══════════════════════════════════════
# TAB 3: Performance
# ══════════════════════════════════════
with tab3:
    st.subheader("Performance do Modelo")

    p1, p2, p3 = st.columns(3)
    p1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    p2.metric("Average Precision", f"{metrics['average_precision']:.4f}")
    p3.metric("Amostras de Teste", f"{metrics['test_samples']:,}")

    img_path = os.path.join(MODEL_DIR, 'model_results.png')
    if os.path.exists(img_path):
        st.image(img_path, caption="Resultados do Treinamento e Avaliação", use_container_width=True)

    st.subheader("Sobre o Modelo")
    st.markdown("""
    | Componente | Detalhe |
    |---|---|
    | **Arquitetura** | Rede Neural Densa (DNN) — 4 camadas ocultas |
    | **Camadas** | Dense(128) → Dense(64) → Dense(32) → Dense(16) → Dense(1) |
    | **Regularização** | BatchNormalization + Dropout(0.3) + L2 |
    | **Otimizador** | Adam (lr=0.001 com ReduceLROnPlateau) |
    | **Loss** | Binary Crossentropy |
    | **Early Stopping** | Monitorando val_AUC (patience=10) |
    | **Framework** | TensorFlow / Keras |
    | **Dataset** | 10.000 pacientes sintéticos (baseado em UCI Diabetes 130-US) |
    """)
