"""
Dashboard interativo para predição de readmissão hospitalar.
Tab 1 consome a API REST (autenticação JWT).
Tabs 2-4 carregam dados locais para análise e comparação de modelos.
"""

import os
import sys
import json

import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import joblib

# ──────────────────────────────────────────────
# Config de paths
# ──────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, '..')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'hospital_readmission.csv')

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.preprocessing import RISK_LOW, RISK_MODERATE

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Readmissão Hospitalar // ML System",
    page_icon=None,
    layout="wide"
)

# ──────────────────────────────────────────────
# Cyberpunk theme
# ──────────────────────────────────────────────

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

html, body, [class*="css"] {{
    font-family: 'Barlow', monospace;
    background-color: {CP_BG};
    color: {CP_TEXT};
}}
.stApp {{ background-color: {CP_BG}; }}
#MainMenu, footer, header {{ visibility: hidden; }}

h1 {{ font-family: 'Share Tech Mono', monospace !important;
      color: {CP_YELLOW} !important; letter-spacing: 4px;
      text-transform: uppercase; font-size: 1.6rem !important; }}
h2, h3 {{ font-family: 'Share Tech Mono', monospace !important;
           color: {CP_CYAN} !important; letter-spacing: 2px;
           text-transform: uppercase; font-size: 1rem !important; }}

hr {{ border: none; border-top: 1px solid {CP_YELLOW}; opacity: 0.4; margin: 1rem 0; }}

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

.stSelectbox label, .stSlider label {{
    font-family: 'Share Tech Mono', monospace;
    color: {CP_DIM};
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.stAlert {{ border-radius: 0; border-left-width: 3px; }}
[data-testid="stDataFrame"] {{ border: 1px solid {CP_BORDER}; }}
.stCaptionContainer {{ color: {CP_DIM} !important; font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; }}
</style>
""", unsafe_allow_html=True)


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


# ──────────────────────────────────────────────
# Dados locais (tabs 2-4, sem auth)
# ──────────────────────────────────────────────

@st.cache_data
def load_local_metrics():
    ens_path = os.path.join(MODEL_DIR, 'metrics_ensemble.json')
    if os.path.exists(ens_path):
        with open(ens_path) as f:
            return json.load(f)
    with open(os.path.join(MODEL_DIR, 'metrics.json')) as f:
        return json.load(f)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_xgb_for_importance():
    """Carrega XGBoost apenas para visualização de feature importance (Tab 4)."""
    path = os.path.join(MODEL_DIR, 'best_model_xgb.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_feature_cols():
    return joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))

metrics      = load_local_metrics()
df           = load_data()
feature_cols = load_feature_cols()

# ──────────────────────────────────────────────
# Sidebar — Login
# ──────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")

with st.sidebar:
    st.markdown(
        f"<p style='font-family:Share Tech Mono,monospace;color:{CP_YELLOW};"
        f"font-size:0.8rem;letter-spacing:2px;'>// ACESSO</p>",
        unsafe_allow_html=True
    )

    if "token" not in st.session_state:
        st.session_state.token = None
        st.session_state.username = None

    if st.session_state.token is None:
        username = st.text_input("Usuário", placeholder="seu_usuario")
        password = st.text_input("Senha", type="password", placeholder="••••••••")

        if st.button("Entrar", type="primary"):
            try:
                resp = requests.post(
                    f"{API_URL}/token",
                    data={"username": username, "password": password},
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.token    = data["access_token"]
                    st.session_state.username = username
                    st.session_state.role     = data.get("role", "")
                    st.rerun()
                else:
                    st.error("Usuário ou senha incorretos.")
            except requests.exceptions.ConnectionError:
                st.error(f"API offline. Inicie com:\nuvicorn api.main:app --port 8000")
    else:
        st.markdown(
            f"<p style='font-family:Share Tech Mono,monospace;color:{CP_CYAN};"
            f"font-size:0.75rem;'>● {st.session_state.username} "
            f"[{st.session_state.get('role','')}]</p>",
            unsafe_allow_html=True
        )
        if st.button("Sair"):
            st.session_state.token    = None
            st.session_state.username = None
            st.rerun()

    st.divider()
    st.markdown(
        f"<p style='font-family:Share Tech Mono,monospace;color:{CP_DIM};"
        f"font-size:0.7rem;'>API: {API_URL}</p>",
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

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
# TAB 1: Predição via API
# ══════════════════════════════════════
with tab1:
    if st.session_state.token is None:
        st.markdown(
            f"<div style='border-left:2px solid {CP_YELLOW};padding:16px;background:{CP_CARD};'>"
            f"<p style='font-family:Share Tech Mono,monospace;color:{CP_YELLOW};"
            f"font-size:0.85rem;letter-spacing:2px;'>// AUTENTICAÇÃO NECESSÁRIA</p>"
            f"<p style='color:{CP_TEXT};margin:8px 0 0 0;'>"
            f"Faça login na barra lateral para realizar predições.</p>"
            f"<p style='color:{CP_DIM};font-size:0.8rem;margin:4px 0 0 0;'>"
            f"Todas as predições são registradas conforme LGPD Art. 11, II, f.</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.subheader("Dados do Paciente")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"<p style='color:{CP_YELLOW};font-family:Share Tech Mono,monospace;"
                f"font-size:0.75rem;letter-spacing:2px;'>// PERFIL</p>",
                unsafe_allow_html=True
            )
            age    = st.slider("Idade", 0, 100, 65)
            gender = st.selectbox("Gênero", ["Male", "Female"])
            diag   = st.selectbox("Diagnóstico Principal", [
                "Circulatory", "Respiratory", "Digestive", "Diabetes",
                "Injury", "Musculoskeletal", "Genitourinary", "Other"
            ])

        with col2:
            st.markdown(
                f"<p style='color:{CP_YELLOW};font-family:Share Tech Mono,monospace;"
                f"font-size:0.75rem;letter-spacing:2px;'>// INTERNAÇÃO</p>",
                unsafe_allow_html=True
            )
            time_hosp = st.slider("Dias no Hospital", 1, 14, 4)
            num_meds  = st.slider("Nº de Medicamentos", 1, 40, 15)
            num_proc  = st.slider("Nº de Procedimentos", 0, 6, 2)
            num_diag  = st.slider("Nº de Diagnósticos", 1, 16, 7)
            num_lab   = st.slider("Nº de Exames Laboratoriais", 1, 120, 43)

        with col3:
            st.markdown(
                f"<p style='color:{CP_YELLOW};font-family:Share Tech Mono,monospace;"
                f"font-size:0.75rem;letter-spacing:2px;'>// HISTÓRICO & EXAMES</p>",
                unsafe_allow_html=True
            )
            n_out = st.slider("Visitas Ambulatoriais (último ano)", 0, 5, 0)
            n_em  = st.slider("Visitas Emergência (último ano)", 0, 4, 0)
            n_in  = st.slider("Internações Anteriores (último ano)", 0, 5, 1)
            hba1c = st.selectbox("HbA1c", ["None", "Normal", ">7", ">8"])
            gluc  = st.selectbox("Glicose Sérica", ["None", "Normal", ">200", ">300"])
            ins   = st.selectbox("Insulina", ["No", "Steady", "Up", "Down"])
            chg   = st.selectbox("Mudança de Medicamentos", ["No", "Ch"])
            diab  = st.selectbox("Medicação para Diabetes", ["Yes", "No"])

        payload = {
            "age_numeric": age, "gender": gender, "diag_primary": diag,
            "time_in_hospital": time_hosp, "num_medications": num_meds,
            "num_procedures": num_proc, "num_diagnoses": num_diag,
            "num_lab_procedures": num_lab, "number_outpatient": n_out,
            "number_emergency": n_em, "number_inpatient": n_in,
            "hba1c_result": hba1c, "glucose_serum_test": gluc,
            "insulin": ins, "change_medications": chg, "diabetes_medication": diab,
        }
        headers = {"Authorization": f"Bearer {st.session_state.token}"}

        if st.button("Calcular Risco de Readmissão", type="primary"):
            with st.spinner("Consultando API..."):
                try:
                    r_pred = requests.post(
                        f"{API_URL}/predict", json=payload, headers=headers, timeout=10
                    )
                    r_shap = requests.post(
                        f"{API_URL}/explain", json=payload, headers=headers, timeout=15
                    )
                except requests.exceptions.ConnectionError:
                    st.error("API offline. Inicie com: uvicorn api.main:app --port 8000")
                    st.stop()

            if r_pred.status_code == 401:
                st.warning("Sessão expirada. Faça login novamente.")
                st.session_state.token = None
                st.rerun()
            elif r_pred.status_code != 200:
                st.error(f"Erro na API ({r_pred.status_code}): {r_pred.text}")
                st.stop()

            pred = r_pred.json()
            prob       = pred["readmission_probability"]
            risk_label = pred["risk_level"].upper()
            prediction = pred["prediction"]
            threshold  = pred["threshold_used"]
            calibrated = pred.get("calibrated", False)

            risk_color = CP_CYAN if prob < RISK_LOW else CP_YELLOW if prob < RISK_MODERATE else CP_PINK

            st.divider()
            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric("Probabilidade de Readmissão", f"{prob:.1%}")
                if calibrated:
                    st.caption("✓ Probabilidade calibrada (isotônica)")
            with r2:
                st.markdown(
                    f"<div style='background:{CP_CARD};border-left:2px solid {risk_color};"
                    f"padding:12px 16px;'>"
                    f"<p style='color:{CP_DIM};font-family:Share Tech Mono,monospace;"
                    f"font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;margin:0;'>Nível de Risco</p>"
                    f"<p style='color:{risk_color};font-family:Share Tech Mono,monospace;"
                    f"font-size:1.4rem;margin:0;'>{risk_label}</p></div>",
                    unsafe_allow_html=True
                )
            with r3:
                st.metric("ROC-AUC do Modelo", f"{pred['model_auc']:.4f}")
                st.caption(f"Threshold: {threshold:.2f} | {'Readmissão' if prediction else 'Sem Readmissão'}")

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
            st.pyplot(fig)
            plt.close(fig)

            st.markdown(
                f"<div style='border-left:2px solid {risk_color};padding:10px 16px;"
                f"background:{CP_CARD};margin-top:1rem;'>"
                f"<p style='color:{CP_DIM};font-family:Share Tech Mono,monospace;"
                f"font-size:0.7rem;letter-spacing:1px;margin:0;'>RECOMENDAÇÃO</p>"
                f"<p style='color:{CP_TEXT};margin:4px 0 0 0;'>{pred['recommendation']}</p></div>",
                unsafe_allow_html=True
            )

            # SHAP
            if r_shap.status_code == 200:
                shap_data    = r_shap.json()
                contributions = shap_data["feature_contributions"]
                shap_series  = pd.Series(contributions).reindex(feature_cols).sort_values()
                colors_shap  = [CP_PINK if v > 0 else CP_CYAN for v in shap_series]

                st.divider()
                st.markdown(
                    f"<p style='color:{CP_YELLOW};font-family:Share Tech Mono,monospace;"
                    f"font-size:0.75rem;letter-spacing:2px;'>// EXPLICABILIDADE — SHAP VALUES</p>",
                    unsafe_allow_html=True
                )
                st.caption("Contribuição de cada feature para esta predição (componente XGBoost). "
                           "Positivo = aumenta o risco | Negativo = reduz o risco")

                fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
                apply_cp_style(fig_shap, ax_shap)
                ax_shap.barh(shap_series.index, shap_series.values, color=colors_shap, alpha=0.85)
                ax_shap.axvline(0, color=CP_DIM, linewidth=0.8)
                ax_shap.set_title(f'SHAP Values — Paciente (prob={prob:.1%})')
                ax_shap.set_xlabel('Contribuição SHAP')
                fig_shap.tight_layout()
                st.pyplot(fig_shap)
                plt.close(fig_shap)

                top_risk = shap_data.get("top_risk_factors", [])
                top_prot = shap_data.get("top_protective_factors", [])
                if top_risk:
                    st.markdown(
                        f"<p style='color:{CP_PINK};font-family:Share Tech Mono,monospace;font-size:0.75rem;'>"
                        f"FATORES DE RISCO: {', '.join(top_risk[:3])}</p>",
                        unsafe_allow_html=True
                    )
                if top_prot:
                    st.markdown(
                        f"<p style='color:{CP_CYAN};font-family:Share Tech Mono,monospace;font-size:0.75rem;'>"
                        f"FATORES PROTETORES: {', '.join(top_prot[:3])}</p>",
                        unsafe_allow_html=True
                    )

            st.caption(
                "⚠ Esta predição é auxílio à decisão clínica. "
                "O profissional de saúde retém a responsabilidade pela decisão. "
                "Registro gerado conforme LGPD Art. 11, II, f."
            )

# ══════════════════════════════════════
# TAB 2: Análise do Dataset
# ══════════════════════════════════════
with tab2:
    st.subheader("Análise Exploratória do Dataset")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total de Pacientes",      f"{len(df):,}")
    m2.metric("Taxa de Readmissão",      f"{df['readmitted_30days'].mean():.1%}")
    m3.metric("Média de Medicamentos",   f"{df['num_medications'].mean():.1f}")
    m4.metric("Média de Dias Internado", f"{df['time_in_hospital'].mean():.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    apply_cp_style(fig, *axes)

    ct = df.groupby('diag_primary')['readmitted_30days'].mean().sort_values(ascending=True)
    ct.plot(kind='barh', ax=axes[0], color=CP_CYAN, alpha=0.85)
    axes[0].set_title('Readmissão por Diagnóstico Principal')
    axes[0].set_xlabel('Taxa de Readmissão')

    df['age_group'] = pd.cut(df['age_numeric'], bins=[0, 30, 50, 70, 100],
                              labels=['<30', '30-50', '50-70', '>70'])
    ag = df.groupby('age_group', observed=True)['readmitted_30days'].mean()
    ag.plot(kind='bar', ax=axes[1], color=CP_YELLOW, alpha=0.85, rot=0)
    axes[1].set_title('Readmissão por Faixa Etária')
    axes[1].set_ylabel('Taxa de Readmissão')

    hba = df.groupby('hba1c_result')['readmitted_30days'].mean()
    hba.plot(kind='bar', ax=axes[2], color=CP_PINK, alpha=0.85, rot=0)
    axes[2].set_title('Readmissão por HbA1c')
    axes[2].set_ylabel('Taxa de Readmissão')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Distribuição das Internações Anteriores")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    apply_cp_style(fig2, ax2)
    readm    = df[df['readmitted_30days'] == 1]['number_inpatient']
    no_readm = df[df['readmitted_30days'] == 0]['number_inpatient']
    ax2.hist(readm,    bins=6, alpha=0.75, label='Readmitido',     color=CP_PINK)
    ax2.hist(no_readm, bins=6, alpha=0.75, label='Não Readmitido', color=CP_CYAN)
    ax2.set_xlabel('Nº de Internações Anteriores')
    ax2.set_ylabel('Contagem')
    ax2.legend(facecolor=CP_CARD, edgecolor=CP_BORDER, labelcolor=CP_TEXT)
    ax2.set_title('Distribuição de Internações Anteriores por Desfecho')
    st.pyplot(fig2)
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
        q1.metric("F1-Score",               f"{metrics['f1_score']:.4f}")
        q2.metric("Precisão",               f"{metrics['precision']:.4f}")
        q3.metric("Recall (Sensibilidade)",  f"{metrics['recall']:.4f}")

    img_path = os.path.join(MODEL_DIR, 'model_results.png')
    if os.path.exists(img_path):
        st.image(img_path, caption="Resultados do Treinamento e Avaliação")
    else:
        st.info("Imagem não encontrada. Execute: python model/train.py")

    if metrics.get('model_version'):
        st.caption(f"v{metrics['model_version']} | treinado: {metrics.get('trained_at', 'N/A')}")

    st.subheader("Arquitetura do Modelo")
    st.markdown("""
    | Componente | Detalhe |
    |---|---|
    | **Arquitetura** | Rede Neural Densa (DNN) — 4 camadas ocultas |
    | **Camadas** | Dense(128) → Dense(64) → Dense(32) → Dense(16) → Dense(1) |
    | **Regularização** | BatchNormalization + Dropout(0.3) + L2 |
    | **Otimizador** | Adam (lr=0.001 com ReduceLROnPlateau) |
    | **Loss** | Binary Crossentropy |
    | **Early Stopping** | val_AUC, patience=10 |
    | **Framework** | TensorFlow / Keras |
    | **Dataset** | 10.000 pacientes sintéticos (UCI Diabetes 130-US) |
    """)

# ══════════════════════════════════════
# TAB 4: DNN vs XGBoost
# ══════════════════════════════════════
with tab4:
    st.subheader("Comparação: DNN vs XGBoost")

    xgb_metrics_path = os.path.join(MODEL_DIR, 'metrics_xgb.json')
    xgb_img_path     = os.path.join(MODEL_DIR, 'model_results_xgb.png')

    if not os.path.exists(xgb_metrics_path):
        st.info("Modelo XGBoost não encontrado. Execute: python model/train_xgboost.py")
    else:
        with open(xgb_metrics_path) as f:
            xgb_metrics = json.load(f)

        comparison = {
            'Métrica':     ['ROC-AUC', 'Average Precision', 'F1-Score', 'Precisão', 'Recall'],
            'DNN (Keras)': [metrics.get(k, '-') for k in ['roc_auc', 'average_precision', 'f1_score', 'precision', 'recall']],
            'XGBoost':     [xgb_metrics.get(k, '-') for k in ['roc_auc', 'average_precision', 'f1_score', 'precision', 'recall']],
        }
        df_comp = pd.DataFrame(comparison)

        def highlight_best(row):
            a, b = row['DNN (Keras)'], row['XGBoost']
            if isinstance(a, float) and isinstance(b, float):
                if a > b: return ['', f'font-weight:bold;color:{CP_CYAN}', '']
                if b > a: return ['', '', f'font-weight:bold;color:{CP_YELLOW}']
            return ['', '', '']

        st.dataframe(df_comp.style.apply(highlight_best, axis=1), hide_index=True)

        c1, c2, c3 = st.columns(3)
        c1.caption(f"DNN treinado: {metrics.get('trained_at', 'N/A')[:10]}")
        c2.caption(f"XGBoost treinado: {xgb_metrics.get('trained_at', 'N/A')[:10]}")
        c3.caption(f"XGBoost melhor iteração: {xgb_metrics.get('best_iteration', 'N/A')}")

        if os.path.exists(xgb_img_path):
            st.image(xgb_img_path, caption="XGBoost — ROC, Matriz de Confusão, Feature Importance")

        # Feature importance
        xgb_model_loaded = load_xgb_for_importance()
        if xgb_model_loaded is not None:
            st.subheader("Importância das Features (XGBoost)")
            importance = pd.Series(
                xgb_model_loaded.feature_importances_, index=feature_cols
            ).sort_values(ascending=True)

            fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
            apply_cp_style(fig_fi, ax_fi)
            colors = [CP_YELLOW if v >= importance.quantile(0.75) else CP_CYAN for v in importance]
            importance.plot(kind='barh', ax=ax_fi, color=colors, alpha=0.85)
            ax_fi.set_title('Importância das Features — XGBoost')
            ax_fi.set_xlabel('Importância (ganho médio)')
            ax_fi.axvline(importance.mean(), color=CP_PINK, linestyle='--',
                          alpha=0.8, label=f'Média ({importance.mean():.4f})')
            ax_fi.legend(facecolor=CP_CARD, edgecolor=CP_BORDER, labelcolor=CP_TEXT)
            st.pyplot(fig_fi)
            plt.close(fig_fi)

            top3 = importance.nlargest(3)
            st.info(
                f"Top 3 features: {top3.index[0]} ({top3.iloc[0]:.4f}), "
                f"{top3.index[1]} ({top3.iloc[1]:.4f}), "
                f"{top3.index[2]} ({top3.iloc[2]:.4f})"
            )

        # Ensemble
        ens_metrics_path = os.path.join(MODEL_DIR, 'metrics_ensemble.json')
        ens_img_path     = os.path.join(MODEL_DIR, 'model_results_ensemble.png')

        st.divider()
        if os.path.exists(ens_metrics_path):
            with open(ens_metrics_path) as f:
                ens_metrics = json.load(f)

            st.subheader("Ensemble: DNN + XGBoost")
            e1, e2, e3 = st.columns(3)
            e1.metric("Peso DNN",         f"{ens_metrics.get('dnn_weight', 0):.0%}")
            e2.metric("Peso XGBoost",     f"{ens_metrics.get('xgb_weight', 0):.0%}")
            e3.metric("ROC-AUC Ensemble", f"{ens_metrics.get('roc_auc', 0):.4f}")

            comp3 = {
                'Métrica':  ['ROC-AUC', 'Avg Precision', 'F1-Score', 'Precisão', 'Recall'],
                'DNN':      [metrics.get(k, '-') for k in ['roc_auc', 'average_precision', 'f1_score', 'precision', 'recall']],
                'XGBoost':  [xgb_metrics.get(k, '-') for k in ['roc_auc', 'average_precision', 'f1_score', 'precision', 'recall']],
                'Ensemble': [ens_metrics.get(k, '-') for k in ['roc_auc', 'average_precision', 'f1_score', 'precision', 'recall']],
            }
            df_comp3 = pd.DataFrame(comp3)

            def highlight_best3(row):
                vals = [row['DNN'], row['XGBoost'], row['Ensemble']]
                numeric = [v for v in vals if isinstance(v, float)]
                if not numeric: return [''] * 4
                best = max(numeric)
                return [''] + [f'font-weight:bold;color:{CP_YELLOW}' if v == best else '' for v in vals]

            st.dataframe(df_comp3.style.apply(highlight_best3, axis=1), hide_index=True)

            if os.path.exists(ens_img_path):
                st.image(ens_img_path, caption="Ensemble — ROC e Comparação de Métricas")
