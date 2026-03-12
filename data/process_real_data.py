"""
Processamento do dataset real UCI Diabetes 130-US Hospitals.
Converte o CSV original para o mesmo formato usado pelo pipeline de treino.

Como obter o dataset:
  1. Acesse: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
     ou Kaggle: https://www.kaggle.com/datasets/brandao/diabetes
  2. Baixe e extraia o arquivo
  3. Coloque 'diabetic_data.csv' nesta pasta (data/)
  4. Execute: python data/process_real_data.py

Saída: data/hospital_readmission.csv  (substitui o sintético)
"""

import os
import numpy as np
import pandas as pd

DATA_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(DATA_DIR, 'diabetic_data.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'hospital_readmission.csv')

# ──────────────────────────────────────────────
# 1. CARREGAR
# ──────────────────────────────────────────────

print("Carregando dataset UCI...")
df = pd.read_csv(INPUT_CSV, na_values=['?'])
print(f"Shape original: {df.shape}")

# ──────────────────────────────────────────────
# 2. VARIÁVEL ALVO
# readmitted: '<30' → 1 (readmitido em 30 dias), demais → 0
# ──────────────────────────────────────────────

df['readmitted_30days'] = (df['readmitted'] == '<30').astype(int)
print(f"Taxa de readmissão em 30 dias: {df['readmitted_30days'].mean():.1%}")

# ──────────────────────────────────────────────
# 3. IDADE → NUMÉRICA
# Faixas como '[60-70)' → midpoint = 65
# ──────────────────────────────────────────────

def age_to_numeric(age_str):
    if pd.isna(age_str):
        return np.nan
    low = int(age_str.strip('[]()').split('-')[0])
    return low + 5

df['age_numeric'] = df['age'].apply(age_to_numeric)

# ──────────────────────────────────────────────
# 4. DIAGNÓSTICO PRINCIPAL → CATEGORIA
# Mapeamento ICD-9 → categorias do projeto
# ──────────────────────────────────────────────

def icd9_to_category(code):
    if pd.isna(code):
        return 'Other'
    code = str(code).strip()
    # Diabetes (250.xx)
    if code.startswith('250'):
        return 'Diabetes'
    # Códigos V e E → Other
    if code.startswith('V') or code.startswith('E'):
        return 'Other'
    try:
        num = float(code)
    except ValueError:
        return 'Other'
    if 390 <= num <= 459 or num == 785:
        return 'Circulatory'
    if 460 <= num <= 519 or num == 786:
        return 'Respiratory'
    if 520 <= num <= 579 or num == 787:
        return 'Digestive'
    if 800 <= num <= 999:
        return 'Injury'
    if 710 <= num <= 739:
        return 'Musculoskeletal'
    if 580 <= num <= 629 or num == 788:
        return 'Genitourinary'
    return 'Other'

df['diag_primary'] = df['diag_1'].apply(icd9_to_category)

# ──────────────────────────────────────────────
# 5. EXAMES CLÍNICOS
# ──────────────────────────────────────────────

# HbA1c: 'Norm' → 'Normal', demais iguais
df['hba1c_result'] = df['A1Cresult'].replace({'Norm': 'Normal'}).fillna('None')

# Glicose sérica: 'Norm' → 'Normal', demais iguais
df['glucose_serum_test'] = df['max_glu_serum'].replace({'Norm': 'Normal'}).fillna('None')

# ──────────────────────────────────────────────
# 6. RENOMEAR / MAPEAR COLUNAS DIRETAS
# ──────────────────────────────────────────────

df['gender']             = df['gender'].replace({'Unknown/Invalid': np.nan})
df['change_medications'] = df['change']
df['diabetes_medication']= df['diabetesMed']
df['num_diagnoses']      = df['number_diagnoses']

# ──────────────────────────────────────────────
# 7. SELECIONAR E LIMPAR
# ──────────────────────────────────────────────

cols_needed = [
    'age_numeric', 'gender', 'diag_primary',
    'time_in_hospital', 'num_medications', 'num_procedures',
    'num_diagnoses', 'num_lab_procedures',
    'number_outpatient', 'number_emergency', 'number_inpatient',
    'hba1c_result', 'glucose_serum_test',
    'insulin', 'change_medications', 'diabetes_medication',
    'readmitted_30days'
]

df_out = df[cols_needed].copy()

# Remover linhas com valores críticos ausentes
before = len(df_out)
df_out = df_out.dropna(subset=['age_numeric', 'gender'])
print(f"Removidas {before - len(df_out)} linhas com idade/gênero ausentes")

# Garantir valores válidos de gênero
df_out = df_out[df_out['gender'].isin(['Male', 'Female'])]

# Clipar valores numéricos nos mesmos limites do pipeline
df_out['age_numeric']      = df_out['age_numeric'].clip(0, 100).astype(int)
df_out['time_in_hospital'] = df_out['time_in_hospital'].clip(1, 14).astype(int)
df_out['num_medications']  = df_out['num_medications'].clip(1, 40).astype(int)
df_out['num_procedures']   = df_out['num_procedures'].clip(0, 6).astype(int)
df_out['num_diagnoses']    = df_out['num_diagnoses'].clip(1, 16).astype(int)
df_out['num_lab_procedures'] = df_out['num_lab_procedures'].clip(1, 120).astype(int)
df_out['number_outpatient']  = df_out['number_outpatient'].clip(0, 5).astype(int)
df_out['number_emergency']   = df_out['number_emergency'].clip(0, 4).astype(int)
df_out['number_inpatient']   = df_out['number_inpatient'].clip(0, 5).astype(int)

# Garantir valores categóricos válidos
valid_insulin  = ['No', 'Steady', 'Up', 'Down']
valid_change   = ['No', 'Ch']
valid_diab_med = ['Yes', 'No']

df_out['insulin']             = df_out['insulin'].where(df_out['insulin'].isin(valid_insulin), 'No')
df_out['change_medications']  = df_out['change_medications'].where(df_out['change_medications'].isin(valid_change), 'No')
df_out['diabetes_medication'] = df_out['diabetes_medication'].where(df_out['diabetes_medication'].isin(valid_diab_med), 'No')

# ──────────────────────────────────────────────
# 8. SALVAR
# ──────────────────────────────────────────────

df_out = df_out.reset_index(drop=True)
df_out.to_csv(OUTPUT_CSV, index=False)

print(f"\nDataset processado: {len(df_out):,} pacientes")
print(f"Taxa de readmissão em 30 dias: {df_out['readmitted_30days'].mean():.1%}")
print(f"\nDistribuição — Diagnóstico Principal:")
print(df_out['diag_primary'].value_counts())
print(f"\nSalvo em: {OUTPUT_CSV}")
print("\nPróximos passos:")
print("  python model/train.py")
print("  python model/train_xgboost.py")
