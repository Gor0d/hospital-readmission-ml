# Hospital Readmission Prediction — ML with Keras

Projeto de **Ciência de Dados** para predição de risco de readmissão hospitalar em 30 dias, utilizando uma rede neural profunda treinada com Keras/TensorFlow.

[![Python](https://img.shields.io/badge/Python-3.13+-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)

---

## Problema

Readmissões hospitalares dentro de 30 dias representam um dos principais indicadores de qualidade assistencial e um dos maiores custos evitáveis no setor de saúde. Identificar pacientes em alto risco **antes da alta** permite intervenções preventivas, reduzindo readmissões e custos operacionais.

---

## Dataset

Dataset sintético de **10.000 pacientes**, gerado com base nas características do [Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008) (UCI Machine Learning Repository).

**Features utilizadas (19):**
- Dados demográficos: idade, gênero
- Diagnóstico principal: Circulatório, Respiratório, Diabetes etc.
- Internação: dias no hospital, n° de medicamentos, procedimentos, exames laboratoriais
- Histórico: internações, emergências e visitas ambulatoriais no último ano
- Exames clínicos: HbA1c, glicose sérica
- Medicação: insulina, mudança de medicamentos
- Features derivadas: `risk_score`, `medication_complexity`, `hospital_utilization`

**Variável alvo:** `readmitted_30days` (binário: 0 = não readmitido, 1 = readmitido)

---

## Arquitetura do Modelo

```
Input (19 features)
    ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(64)  → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(32)  → BatchNorm → ReLU → Dropout(0.15)
    ↓
Dense(16)  → ReLU
    ↓
Dense(1)   → Sigmoid → P(readmissão)
```

**Configuração de treinamento:**
- Otimizador: Adam (lr=0.001 com ReduceLROnPlateau)
- Loss: Binary Crossentropy
- Regularização: L2 + Dropout + BatchNormalization
- Early Stopping: monitor=val_AUC, patience=10
- Class Weights: balanceamento da classe minoritária

---

## Resultados

| Métrica | Valor |
|---------|-------|
| **ROC-AUC** | 0.6848 |
| **Average Precision** | 0.7575 |
| **F1-Score** | 0.6826 |
| **Precisão** | 0.7116 |
| **Recall** | 0.6558 |

> ROC-AUC de ~0.68 é consistente com benchmarks da literatura para esse problema clínico com dados tabulares.

---

## Estrutura do Projeto

```
hospital-readmission/
├── data/
│   ├── generate_data.py           # Geração do dataset sintético
│   └── hospital_readmission.csv
├── model/
│   ├── train.py                   # Treinamento da rede neural (Keras)
│   ├── best_model.keras           # Modelo salvo
│   ├── scaler.pkl                 # StandardScaler
│   ├── encoders.pkl               # LabelEncoders categóricos
│   ├── feature_cols.pkl           # Lista de features
│   ├── metrics.json               # Métricas de avaliação
│   └── model_results.png          # Curvas de treino e avaliação
├── api/
│   └── main.py                    # API REST (FastAPI)
├── dashboard/
│   └── app.py                     # Dashboard interativo (Streamlit)
├── utils/
│   └── preprocessing.py           # Pré-processamento compartilhado (API + Dashboard)
├── requirements.txt
└── README.md
```

---

## Como Executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

> **Windows:** Se ocorrer erro de caminho longo, habilite o suporte a Long Paths:
> ```powershell
> # PowerShell como Administrador
> New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
> ```

### 2. Gerar dataset e treinar modelo

```bash
python data/generate_data.py
python model/train.py
```

### 3. Iniciar a API

```bash
python api/main.py
```
Acesse: [http://localhost:8000/docs](http://localhost:8000/docs) — Swagger UI automático

> Para restringir origens do CORS em produção, defina a variável de ambiente:
> ```bash
> set ALLOWED_ORIGINS=https://meudominio.com
> ```

### 4. Iniciar o Dashboard

```bash
python -m streamlit run dashboard/app.py
```
Acesse: [http://localhost:8501](http://localhost:8501)

---

## Exemplo de Uso da API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age_numeric": 72,
    "gender": "Female",
    "diag_primary": "Circulatory",
    "time_in_hospital": 5,
    "num_medications": 18,
    "num_procedures": 2,
    "num_diagnoses": 8,
    "num_lab_procedures": 45,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
    "hba1c_result": ">8",
    "glucose_serum_test": ">200",
    "insulin": "Up",
    "change_medications": "Ch",
    "diabetes_medication": "Yes"
  }'
```

**Resposta:**
```json
{
  "readmission_probability": 0.7823,
  "risk_level": "Alto",
  "prediction": 1,
  "recommendation": "Alto risco de readmissão. Recomendar acompanhamento intensivo, revisão do plano de alta e contato ativo em 7 dias.",
  "model_auc": 0.6848
}
```

---

## Tecnologias

| Categoria | Tecnologias |
|-----------|-------------|
| **ML / DL** | TensorFlow 2.20+, Keras, Scikit-learn |
| **Dados** | Pandas, NumPy |
| **API** | FastAPI, Uvicorn, Pydantic v2 |
| **Dashboard** | Streamlit, Matplotlib, Seaborn |
| **Serialização** | Joblib |

---

## Sobre

Projeto desenvolvido por **Emerson Guimarães** como parte do portfólio de Ciência de Dados.
Contexto: 9+ anos de experiência em ambientes hospitalares, aplicando ML ao problema real de gestão de readmissões.

- LinkedIn: [linkedin.com/in/emersongsguimaraes](https://linkedin.com/in/emersongsguimaraes)
- GitHub: [github.com/Gor0d](https://github.com/Gor0d)
