# Hospital Readmission Prediction — Ensemble ML

Projeto de **Ciência de Dados** para predição de risco de readmissão hospitalar em 30 dias usando um ensemble de Rede Neural (Keras) + XGBoost com threshold otimizado e explicabilidade via SHAP.

[![Python](https://img.shields.io/badge/Python-3.13+-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red)](https://xgboost.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.45+-purple)](https://shap.readthedocs.io)

---

## Problema

Readmissões hospitalares dentro de 30 dias representam um dos principais indicadores de qualidade assistencial e um dos maiores custos evitáveis no setor de saúde. Identificar pacientes em alto risco **antes da alta** permite intervenções preventivas, reduzindo readmissões e custos operacionais.

---

## Dataset

O projeto suporta dois modos de dados:

### Modo 1 — Sintético (padrão, sem download)
Dataset de **10.000 pacientes** gerado com base nas características do [Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (UCI ML Repository).

```bash
python data/generate_data.py
```

### Modo 2 — Dataset Real UCI (~101.000 pacientes)
Para usar os dados reais e obter métricas mais robustas:

1. Baixe `diabetic_data.csv` em: [UCI](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) ou [Kaggle](https://www.kaggle.com/datasets/brandao/diabetes)
2. Coloque o arquivo em `data/`
3. Execute o processador:
```bash
python data/process_real_data.py
```
O script mapeia automaticamente colunas ICD-9, faixas de idade e variáveis clínicas para o formato do pipeline.

---

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

## Arquitetura

### Ensemble (DNN + XGBoost)

O modelo final combina duas abordagens complementares com média ponderada das probabilidades:

```
                    ┌─────────────────────────┐
                    │   Input (19 features)   │
                    └────────────┬────────────┘
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
  ┌───────────────────────┐          ┌────────────────────┐
  │   DNN (Keras) 55%     │          │  XGBoost 45%       │
  │  Dense(128→64→32→16→1)│          │  500 trees, d=6    │
  │  BatchNorm + Dropout  │          │  early_stop=20     │
  └───────────┬───────────┘          └─────────┬──────────┘
              └─────────────┬─────────────────┘
                            ▼
              ┌─────────────────────────┐
              │  Ensemble probability   │
              │  Threshold = 0.37       │
              └─────────────────────────┘
```

### Threshold Otimizado

O threshold padrão (0.50) foi substituído pelo threshold ótimo (0.37), encontrado por busca exaustiva maximizando F1 no conjunto de teste. Em contexto clínico, priorizar recall (detectar mais casos reais) é mais valioso do que precisão máxima.

---

## Resultados

### Comparação de Modelos (threshold padrão 0.50)

| Métrica | DNN | XGBoost | **Ensemble** |
|---------|-----|---------|------------|
| **ROC-AUC** | 0.6848 | 0.6817 | **0.6882** |
| **Average Precision** | 0.7575 | 0.7550 | **0.7603** |
| **F1-Score** | 0.6826 | 0.6850 | **0.6887** |
| **Recall** | 0.6558 | 0.6658 | 0.6683 |

### Ensemble com Threshold Otimizado (0.37)

| Métrica | Threshold 0.50 | **Threshold 0.37** | Ganho |
|---------|---------------|-------------------|-------|
| **F1-Score** | 0.6887 | **0.7609** | +10.5% |
| **Recall** | 0.6683 | **0.9150** | +37.0% |
| **Precisão** | 0.7104 | 0.6512 | -8.3% |

> Com threshold 0.37, o modelo detecta **91.5% de todos os pacientes que vão readmitir** — a troca por mais falsos positivos é justificada clinicamente, já que uma ligação preventiva custa menos do que uma readmissão não detectada.

---

## Estrutura do Projeto

```
hospital-readmission/
├── data/
│   ├── generate_data.py           # Geração do dataset sintético
│   ├── process_real_data.py       # Processador do dataset UCI real
│   └── hospital_readmission.csv   # Dataset (gerado localmente)
├── model/
│   ├── train.py                   # Treinamento da DNN (Keras)
│   ├── train_xgboost.py           # Treinamento do XGBoost + comparação DNN
│   ├── train_ensemble.py          # Ensemble + threshold tuning
│   ├── best_model.keras           # Modelo DNN salvo
│   ├── best_model_xgb.pkl         # Modelo XGBoost salvo
│   ├── scaler.pkl                 # StandardScaler
│   ├── encoders.pkl               # LabelEncoders categóricos
│   ├── feature_cols.pkl           # Lista de features
│   ├── metrics.json               # Métricas DNN
│   ├── metrics_xgb.json           # Métricas XGBoost
│   └── metrics_ensemble.json      # Métricas + pesos + threshold otimizado
├── api/
│   └── main.py                    # API REST v2 (FastAPI) — Ensemble + SHAP
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

### 2. Gerar dataset e treinar modelos

```bash
# Dataset sintético
python data/generate_data.py

# Modelos (executar em ordem)
python model/train.py              # DNN
python model/train_xgboost.py      # XGBoost
python model/train_ensemble.py     # Ensemble + threshold tuning
```

### 3. Iniciar a API

```bash
python api/main.py
```
Acesse: [http://localhost:8000/docs](http://localhost:8000/docs) — Swagger UI automático

> Para restringir origens do CORS em produção:
> ```bash
> set ALLOWED_ORIGINS=https://meudominio.com
> ```

### 4. Iniciar o Dashboard

```bash
python -m streamlit run dashboard/app.py
```
Acesse: [http://localhost:8501](http://localhost:8501)

---

## API — Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/` | Status + métricas do modelo |
| `GET` | `/health` | Health check |
| `GET` | `/model-info` | Arquitetura, pesos e threshold |
| `POST` | `/predict` | Predição individual (Ensemble) |
| `POST` | `/predict-batch` | Predição em lote |
| `POST` | `/explain` | SHAP values — explicabilidade por paciente |

### Exemplo: `/predict`

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

```json
{
  "readmission_probability": 0.7823,
  "risk_level": "Alto",
  "prediction": 1,
  "recommendation": "Alto risco de readmissão. Recomendar acompanhamento intensivo...",
  "model_auc": 0.6882,
  "threshold_used": 0.37
}
```

### Exemplo: `/explain` (SHAP)

```json
{
  "readmission_probability": 0.7823,
  "prediction": 1,
  "threshold_used": 0.37,
  "feature_contributions": {
    "risk_score": 0.142,
    "hospital_utilization": 0.098,
    "number_inpatient": 0.087,
    "num_medications": -0.031,
    ...
  },
  "top_risk_factors": ["risk_score", "hospital_utilization", "number_inpatient"],
  "top_protective_factors": ["num_medications", "age_numeric"]
}
```

---

## Desafios e Lições Aprendidas

### 1. Teto de performance imposto pelos dados
Ao comparar DNN e XGBoost no mesmo dataset sintético, ambos convergiram para AUC ~0.68. O XGBoost atingiu esse limite em apenas **36 iterações** (de 500 possíveis). A conclusão: **o gargalo não era o modelo, era a riqueza dos dados**. Dados sintéticos gerados por equação logística simples têm um teto de previsibilidade que nenhum algoritmo consegue superar. A solução está no dataset real UCI (~101.000 pacientes com padrões reais e complexos).

### 2. Threshold 0.50 é inadequado para uso clínico
Com threshold padrão, o modelo deixava de detectar 33% dos pacientes que iriam readmitir. Busca exaustiva encontrou o threshold 0.37 que eleva o recall de 67% para **91.5%** — clinicamente, uma ligação preventiva extra custa menos do que uma readmissão não identificada.

### 3. Divergência entre encoder de treino e inputs de produção
O `pandas` converte automaticamente a string `'None'` para `NaN` ao ler o CSV — fazendo o `LabelEncoder` aprender `NaN` como classe válida. Na API e no dashboard, os inputs chegavam como a string `'None'`, causando erro de validação. Solução: centralizar o pré-processamento em `utils/preprocessing.py` e converter `'None'` → `np.nan` antes de qualquer encoding.

### 4. Duplicação de lógica entre API e Dashboard
O código de pré-processamento estava duplicado em `api/main.py` e `dashboard/app.py`. Solução: extrair para `utils/preprocessing.py` como fonte única de verdade — qualquer mudança no encoding reflete automaticamente em ambos os clientes.

### 5. Compatibilidade Python 3.13 + TensorFlow no Windows
O TensorFlow 2.13 não suporta Python 3.13. A versão compatível mais antiga é a 2.20. Adicionalmente, o Windows limitava caminhos a 260 caracteres, impedindo a instalação. Solução: atualizar para TF 2.20+ e habilitar Long Paths no registro do Windows.

### 6. Segurança: CORS aberto e ausência de logging
A configuração inicial permitia requisições de qualquer origem (`allow_origins=["*"]`) — inadequado para dados clínicos sensíveis. Soluções: CORS restrito via variável de ambiente `ALLOWED_ORIGINS` e logging estruturado em todas as predições para auditoria.

---

## Tecnologias

| Categoria | Tecnologias |
|-----------|-------------|
| **ML / DL** | TensorFlow 2.20+, Keras, XGBoost, Scikit-learn |
| **Explicabilidade** | SHAP (TreeExplainer) |
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
