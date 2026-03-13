"""
Dashboard interativo para predicao de readmissao hospitalar.
Executa localmente usando Streamlit + modelo Keras salvo.
"""

import os
import sys
import json

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import joblib
import tensorflow as tf

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, '..')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'hospital_readmission.csv')

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.preprocessing import build_feature_vector, classify_risk, RISK_LOW, RISK_MODERATE  # noqa: E402

st.set_page_config(
    page_title="Readmissão Hospitalar // ML System",
    page_icon=None,
    layout="wide"
)


# Cyberpunk theme


CP_YELLOW = '#FCEE0A'
CP_CYAN   = '#00D4FF'
CP_PINK   = '#FF003C'
CP_BG     = '#0d0d1a'
CP_CARD   = '#12122a'
CP_BORDER = '#1e1e3a'
CP_TEXT   = '#c8c8e0'
CP_DIM    = '#5a5a7a'

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {{
    font-family: 'Barlow', monospace;
    background-color: {CP_BG};
    color: {CP_TEXT};
}}
.stApp {{ background-color: {CP_BG}; }}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── Title / Headers ── */
h1 {{ font-family: 'Share Tech Mono', monospace !important;
      color: {CP_YELLOW} !important; letter-spacing: 4px;
      text-transform: uppercase; font-size: 1.6rem !important; }}
h2, h3 {{ font-family: 'Share Tech Mono', monospace !important;
           color: {CP_CYAN} !important; letter-spacing: 2px;
           text-transform: uppercase; font-size: 1rem !important; }}

/* ── Divider ── */
hr {{ border: none; border-top: 1px solid {CP_YELLOW}; opacity: 0.4; margin: 1rem 0; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid {CP_BORDER};
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Share Tech Mono', monospace;
    color: {CP_DIM};
    background: transparent;
    border: 1px solid transparent;
    letter-spacing: 1px;
    font-size: 0.78rem;
    text-transform: uppercase;
    padding: 8px 18px;
    border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
    color: {CP_YELLOW} !important;
    border: 1px solid {CP_YELLOW} !important;
    background: rgba(252,238,10,0.06) !important;
}}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.5rem; }}

/* ── Buttons ── */
.stButton button {{
    font-family: 'Share Tech Mono', monospace;
    background: transparent;
    color: {CP_YELLOW};
    border: 1px solid {CP_YELLOW};
    border-radius: 0;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-size: 0.85rem;
    transition: all 0.15s;
}}
.stButton button:hover {{
    background: {CP_YELLOW};
    color: {CP_BG};
}}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: {CP_CARD};
    border-left: 2px solid {CP_YELLOW};
    padding: 12px 16px;
}}
[data-testid="stMetricLabel"] {{
    font-family: 'Share Tech Mono', monospace;
    color: {CP_DIM} !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
[data-testid="stMetricValue"] {{
    color: {CP_YELLOW} !important;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.4rem !important;
}}

/* ── Selectbox / Slider ── */
.stSelectbox label, .stSlider label {{
    font-family: 'Share Tech Mono', monospace;
    color: {CP_DIM};
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* ── Alerts ── */
.stAlert {{
    border-radius: 0;
    border-left-width: 3px;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {CP_BORDER};
}}

/* ── Caption ── */
.stCaptionContainer {{ color: {CP_DIM} !important; font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; }}
</style>
""", unsafe_allow_html=True)

# Matplotlib dark theme

def apply_cp_style(fig, *axes):
    fig.patch.set_facecolor(CP_CARD)
    for ax in axes:
        ax.set_facecolor('#0f0f22')
        ax.tick_params(colors=CP_DIM, labelsize=8)
        ax.xaxis.label.set_color(CP_DIM)
        ax.yaxis.label.set_color(CP_DIM)
        ax.title.set_color(CP_CYAN)
        for spine in ax.spines.values():
            spine.set_edgecolor(CP_BORDER)
        ax.grid(color=CP_BORDER, linewidth=0.5, alpha=0.6)

# Artefatos


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

# Header


st.title("Readmissão Hospitalar // Prediction System")
st.markdown(
    f"<p style='font-family:Share Tech Mono,monospace;color:{CP_DIM};font-size:0.8rem;"
    f"letter-spacing:2px;'>DEEP NEURAL NETWORK + XGBOOST + ENSEMBLE &nbsp;|&nbsp; "
    f"ROC-AUC {metrics['roc_auc']:.4f} &nbsp;|&nbsp; "
    f"F1 {metrics.get('f1_score', 0):.4f}</p>",
    unsafe_allow_html=True
)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Predição Individual",
    "Análise do Dataset",
    "Performance do Modelo",
    "DNN vs XGBoost",
])

# ══════════════════════════════════════
# TAB 1: Predicao
# ══════════════════════════════════════
with tab1:
    st.subheader("Dados do Paciente")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<p style='color:{CP_YELLOW};font-family:Share Tech Mono,monospace;"
                    f"font-size:0.75rem;letter-spacing:2px;'>// PERFIL</p>", unsafe_allow_html=True)
        age      = st.slider("Idade", 0, 100, 65)
        gender   = st.selectbox("Gênero", ["Male", "Female"])
        diag     = st.selectbox("Diagnóstico Principal", [
            "Circulatory", "Respiratory", "Digestive", "Diabetes",
            "Injury", "Musculoskeletal", "Genitourinary", "Other"
        ])

    with col2:
        st.markdown(f"<p style='color:{CP_YELLOW};font-family:Share Tech Mono,monospace;"
                    f"font-size:0.75rem;letter-spacing:2px;'>// INTERNAÇÃO</p>", unsafe_allow_html=True)
        time_hosp  = st.slider("Dias no Hospital", 1, 14, 4)
        num_meds   = st.slider("N de Medicamentos", 1, 40, 15)
        num_proc   = st.slider("N de Procedimentos", 0, 6, 2)
        num_diag   = st.slider("N de Diagnósticos", 1, 16, 7)
        num_lab    = st.slider("N de Exames Laboratoriais", 1, 120, 43)

    with col3:
        st.markdown(f"<p style='color:{CP_YELLOW};font-family:Share Tech Mono,monospace;"
                    f"font-size:0.75rem;letter-spacing:2px;'>// HISTORICO & EXAMES</p>", unsafe_allow_html=True)
        n_out  = st.slider("Visitas Ambulatoriais (ultimo ano)", 0, 5, 0)
        n_em   = st.slider("Visitas Emergência (ultimo ano)", 0, 4, 0)
        n_in   = st.slider("Internações Anteriores (ultimo ano)", 0, 5, 1)
        hba1c  = st.selectbox("HbA1c", ["None", "Normal", ">7", ">8"])
        gluc   = st.selectbox("Glicose Sérica", ["None", "Normal", ">200", ">300"])
        ins    = st.selectbox("Insulina", ["No", "Steady", "Up", "Down"])
        chg    = st.selectbox("Mudança de Medicamentos", ["No", "Ch"])
        diab   = st.selectbox("Medicação para Diabetes", ["Yes", "No"])

    if st.button("Calcular Risco de Readmissão", type="primary", use_container_width=True):
        inputs = {
            'age_numeric': age, 'gender': gender, 'diag_primary': diag,
            'time_in_hospital': time_hosp, 'num_medications': num_meds,
            'num_procedures': num_proc, 'num_diagnoses': num_diag,
            'num_lab_procedures': num_lab, 'number_outpatient': n_out,
            'number_emergency': n_em, 'number_inpatient': n_in,
            'hba1c_result': hba1c, 'glucose_serum_test': gluc,
            'insulin': ins, 'change_medications': chg, 'diabetes_medication': diab
        }

        try:
            X = build_feature_vector(inputs, encoders)
            X_sc = scaler.transform(X)
            prob = float(model.predict(X_sc, verbose=0)[0][0])
        except ValueError as exc:
            st.error(f"Erro nos dados de entrada: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Erro ao processar a predição: {exc}")
            st.stop()

        st.divider()
        r1, r2, r3 = st.columns(3)

        risk_color = CP_CYAN if prob < RISK_LOW else CP_YELLOW if prob < RISK_MODERATE else CP_PINK
        risk_label = "BAIXO" if prob < RISK_LOW else "MODERADO" if prob < RISK_MODERATE else "ALTO"

        with r1:
            st.metric("Probabilidade de Readmissão", f"{prob:.1%}")
        with r2:
            st.markdown(
                f"<div style='background:{CP_CARD};border-left:2px solid {risk_color};"
                f"padding:12px 16px;'>"
                f"<p style='color:{CP_DIM};font-family:Share Tech Mono,monospace;"
                f"font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;margin:0;'>Nivel de Risco</p>"
                f"<p style='color:{risk_color};font-family:Share Tech Mono,monospace;"
                f"font-size:1.4rem;margin:0;'>{risk_label}</p></div>",
                unsafe_allow_html=True
            )
        with r3:
            st.metric("ROC-AUC do Modelo", f"{metrics['roc_auc']:.4f}")

        # Gauge
        fig, ax = plt.subplots(figsize=(6, 2))
        apply_cp_style(fig, ax)
        ax.barh(0, 100, height=0.5, color='#1e1e3a')
        ax.barh(0, prob * 100, height=0.5, color=risk_color, alpha=0.9)
        ax.axvline(prob * 100, color='#ffffff', linewidth=1.5, alpha=0.8)
        ax.axvline(RISK_LOW * 100,      color=CP_CYAN,   linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axvline(RISK_MODERATE * 100, color=CP_YELLOW, linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("Probabilidade (%)")
        ax.set_title(f"Risco estimado: {prob:.1%}", fontweight='bold', color=CP_CYAN)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        _, recommendation = classify_risk(prob)
        st.markdown(
            f"<div style='border-left:2px solid {risk_color};padding:10px 16px;"
            f"background:{CP_CARD};margin-top:1rem;'>"
            f"<p style='color:{CP_DIM};font-family:Share Tech Mono,monospace;"
            f"font-size:0.7rem;letter-spacing:1px;margin:0;'>RECOMENDACAO</p>"
            f"<p style='color:{CP_TEXT};margin:4px 0 0 0;'>{recommendation}</p></div>",
            unsafe_allow_html=True
        )

# ══════════════════════════════════════
# TAB 2: Analise
# ══════════════════════════════════════
with tab2:
    st.subheader("Análise Exploratória do Dataset")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total de Pacientes",    f"{len(df):,}")
    m2.metric("Taxa de Readmissão",    f"{df['readmitted_30days'].mean():.1%}")
    m3.metric("Media de Medicamentos", f"{df['num_medications'].mean():.1f}")
    m4.metric("Media de Dias Internado", f"{df['time_in_hospital'].mean():.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    apply_cp_style(fig, *axes)

    ct = df.groupby('diag_primary')['readmitted_30days'].mean().sort_values(ascending=True)
    ct.plot(kind='barh', ax=axes[0], color=CP_CYAN, alpha=0.85)
    axes[0].set_title('Readmissão por Diagnostico Principal')
    axes[0].set_xlabel('Taxa de Readmissão')

    df['age_group'] = pd.cut(df['age_numeric'], bins=[0, 30, 50, 70, 100],
                              labels=['<30', '30-50', '50-70', '>70'])
    ag = df.groupby('age_group', observed=True)['readmitted_30days'].mean()
    ag.plot(kind='bar', ax=axes[1], color=CP_YELLOW, alpha=0.85, rot=0)
    axes[1].set_title('Readmissão por Faixa Etaria')
    axes[1].set_ylabel('Taxa de Readmissão')

    hba = df.groupby('hba1c_result')['readmitted_30days'].mean()
    hba.plot(kind='bar', ax=axes[2], color=CP_PINK, alpha=0.85, rot=0)
    axes[2].set_title('Readmissão por HbA1c')
    axes[2].set_ylabel('Taxa de Readmissão')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Distribuicao das Internações Anteriores")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    apply_cp_style(fig2, ax2)
    readm    = df[df['readmitted_30days'] == 1]['number_inpatient']
    no_readm = df[df['readmitted_30days'] == 0]['number_inpatient']
    ax2.hist(readm,    bins=6, alpha=0.75, label='Readmitido',     color=CP_PINK)
    ax2.hist(no_readm, bins=6, alpha=0.75, label='Nao Readmitido', color=CP_CYAN)
    ax2.set_xlabel('N de Internações Anteriores')
    ax2.set_ylabel('Contagem')
    ax2.legend(facecolor=CP_CARD, edgecolor=CP_BORDER, labelcolor=CP_TEXT)
    ax2.set_title('Distribuicao de Internações Anteriores por Desfecho')
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ══════════════════════════════════════
# TAB 3: Performance
# ══════════════════════════════════════
with tab3:
    st.subheader("Performance do Modelo")

    p1, p2, p3 = st.columns(3)
    p1.metric("ROC-AUC",           f"{metrics['roc_auc']:.4f}")
    p2.metric("Average Precision", f"{metrics['average_precision']:.4f}")
    p3.metric("Amostras de Teste", f"{metrics['test_samples']:,}")

    if metrics.get('f1_score'):
        q1, q2, q3 = st.columns(3)
        q1.metric("F1-Score",              f"{metrics['f1_score']:.4f}")
        q2.metric("Precisao",              f"{metrics['precision']:.4f}")
        q3.metric("Recall (Sensibilidade)", f"{metrics['recall']:.4f}")

    img_path = os.path.join(MODEL_DIR, 'model_results.png')
    if os.path.exists(img_path):
        st.image(img_path, caption="Resultados do Treinamento e Avaliação", use_container_width=True)
    else:
        st.info("Imagem nao encontrada. Execute: python model/train.py")

    if metrics.get('model_version'):
        st.caption(f"v{metrics['model_version']} | treinado: {metrics.get('trained_at', 'N/A')}")

    st.subheader("Arquitetura do Modelo")
    st.markdown("""
    | Componente | Detalhe |
    |---|---|
    | **Arquitetura** | Rede Neural Densa (DNN) — 4 camadas ocultas |
    | **Camadas** | Dense(128) -> Dense(64) -> Dense(32) -> Dense(16) -> Dense(1) |
    | **Regularizacao** | BatchNormalization + Dropout(0.3) + L2 |
    | **Otimizador** | Adam (lr=0.001 com ReduceLROnPlateau) |
    | **Loss** | Binary Crossentropy |
    | **Early Stopping** | val_AUC, patience=10 |
    | **Framework** | TensorFlow / Keras |
    | **Dataset** | 10.000 pacientes sinteticos (UCI Diabetes 130-US) |
    """)

# ══════════════════════════════════════
# TAB 4: DNN vs XGBoost
# ══════════════════════════════════════
with tab4:
    st.subheader("Comparacao: DNN vs XGBoost")

    xgb_metrics_path = os.path.join(MODEL_DIR, 'metrics_xgb.json')
    xgb_img_path     = os.path.join(MODEL_DIR, 'model_results_xgb.png')
    xgb_model_path   = os.path.join(MODEL_DIR, 'best_model_xgb.pkl')

    if not os.path.exists(xgb_metrics_path):
        st.info("Modelo XGBoost nao encontrado. Execute: python model/train_xgboost.py")
    else:
        with open(xgb_metrics_path) as f:
            xgb_metrics = json.load(f)

        comparison = {
            'Metrica':     ['ROC-AUC', 'Average Precision', 'F1-Score', 'Precisao', 'Recall'],
            'DNN (Keras)': [metrics.get('roc_auc', '-'), metrics.get('average_precision', '-'),
                            metrics.get('f1_score', '-'), metrics.get('precision', '-'),
                            metrics.get('recall', '-')],
            'XGBoost':     [xgb_metrics.get('roc_auc', '-'), xgb_metrics.get('average_precision', '-'),
                            xgb_metrics.get('f1_score', '-'), xgb_metrics.get('precision', '-'),
                            xgb_metrics.get('recall', '-')],
        }
        df_comp = pd.DataFrame(comparison)

        def highlight_best(row):
            dnn_val = row['DNN (Keras)']
            xgb_val = row['XGBoost']
            if isinstance(dnn_val, float) and isinstance(xgb_val, float):
                if dnn_val > xgb_val:
                    return ['', f'font-weight:bold;color:{CP_CYAN}', '']
                elif xgb_val > dnn_val:
                    return ['', '', f'font-weight:bold;color:{CP_YELLOW}']
            return ['', '', '']

        st.dataframe(
            df_comp.style.apply(highlight_best, axis=1),
            use_container_width=True, hide_index=True
        )

        c1, c2, c3 = st.columns(3)
        c1.caption(f"DNN treinado: {metrics.get('trained_at', 'N/A')[:10]}")
        c2.caption(f"XGBoost treinado: {xgb_metrics.get('trained_at', 'N/A')[:10]}")
        c3.caption(f"XGBoost melhor iteracao: {xgb_metrics.get('best_iteration', 'N/A')}")

        if os.path.exists(xgb_img_path):
            st.image(xgb_img_path, caption="XGBoost — ROC, Matriz de Confusao, Feature Importance",
                     use_container_width=True)

        if os.path.exists(xgb_model_path):
            st.subheader("Importancia das Features (XGBoost)")
            xgb_model  = joblib.load(xgb_model_path)
            importance = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=True)

            fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
            apply_cp_style(fig_fi, ax_fi)
            colors = [CP_YELLOW if v >= importance.quantile(0.75) else CP_CYAN for v in importance]
            importance.plot(kind='barh', ax=ax_fi, color=colors, alpha=0.85)
            ax_fi.set_title('Importancia das Features — XGBoost')
            ax_fi.set_xlabel('Importancia (ganho medio)')
            ax_fi.axvline(importance.mean(), color=CP_PINK, linestyle='--',
                          alpha=0.8, label=f'Media ({importance.mean():.4f})')
            ax_fi.legend(facecolor=CP_CARD, edgecolor=CP_BORDER, labelcolor=CP_TEXT)
            st.pyplot(fig_fi, use_container_width=True)
            plt.close(fig_fi)

            top3 = importance.nlargest(3)
            st.info(
                f"Top 3 features: {top3.index[0]} ({top3.iloc[0]:.4f}), "
                f"{top3.index[1]} ({top3.iloc[1]:.4f}), "
                f"{top3.index[2]} ({top3.iloc[2]:.4f})"
            )

        # ── Ensemble ──────────────────────────────────
        ens_metrics_path = os.path.join(MODEL_DIR, 'metrics_ensemble.json')
        ens_img_path     = os.path.join(MODEL_DIR, 'model_results_ensemble.png')

        st.divider()
        if not os.path.exists(ens_metrics_path):
            st.info("Ensemble nao treinado. Execute: python model/train_ensemble.py")
        else:
            with open(ens_metrics_path) as f:
                ens_metrics = json.load(f)

            st.subheader("Ensemble: DNN + XGBoost")

            e1, e2, e3 = st.columns(3)
            e1.metric("Peso DNN",          f"{ens_metrics.get('dnn_weight', 0):.0%}")
            e2.metric("Peso XGBoost",      f"{ens_metrics.get('xgb_weight', 0):.0%}")
            e3.metric("ROC-AUC Ensemble",  f"{ens_metrics.get('roc_auc', 0):.4f}")

            comp3 = {
                'Metrica':  ['ROC-AUC', 'Avg Precision', 'F1-Score', 'Precisao', 'Recall'],
                'DNN':      [metrics.get(k, '-') for k in ['roc_auc', 'average_precision', 'f1_score', 'precision', 'recall']],
                'XGBoost':  [xgb_metrics.get(k, '-') for k in ['roc_auc', 'average_precision', 'f1_score', 'precision', 'recall']],
                'Ensemble': [ens_metrics.get(k, '-') for k in ['roc_auc', 'average_precision', 'f1_score', 'precision', 'recall']],
            }
            df_comp3 = pd.DataFrame(comp3)

            def highlight_best3(row):
                vals = [row['DNN'], row['XGBoost'], row['Ensemble']]
                numeric = [v for v in vals if isinstance(v, float)]
                if not numeric:
                    return [''] * 4
                best = max(numeric)
                return [''] + [f'font-weight:bold;color:{CP_YELLOW}' if v == best else '' for v in vals]

            st.dataframe(
                df_comp3.style.apply(highlight_best3, axis=1),
                use_container_width=True, hide_index=True
            )

            if os.path.exists(ens_img_path):
                st.image(ens_img_path, caption="Ensemble — ROC e Comparacao de Metricas",
                         use_container_width=True)
