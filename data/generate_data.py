"""
Geração de dataset sintético realista para predição de readmissão hospitalar.
Baseado nas características do dataset 'Diabetes 130-US hospitals' (UCI ML Repository).
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

np.random.seed(42)
N = 10000

def generate_dataset(n=N):
    age_groups = np.random.choice(
        ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
         '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
        size=n,
        p=[0.01, 0.02, 0.04, 0.07, 0.12, 0.18, 0.22, 0.20, 0.11, 0.03]
    )
    age_numeric = np.array([int(a.strip('[]()').split('-')[0]) + 5 for a in age_groups])

    gender = np.random.choice(['Male', 'Female'], size=n, p=[0.47, 0.53])
    gender_num = (gender == 'Female').astype(int)

    diag_primary = np.random.choice(
        ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes',
         'Injury', 'Musculoskeletal', 'Genitourinary', 'Other'],
        size=n,
        p=[0.30, 0.15, 0.12, 0.18, 0.07, 0.06, 0.06, 0.06]
    )

    time_in_hospital = np.random.choice(range(1, 15), size=n,
        p=[0.12, 0.14, 0.13, 0.12, 0.10, 0.09, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01])

    num_medications = np.clip(
        np.random.poisson(lam=15, size=n) + (age_numeric // 20), 1, 40
    ).astype(int)

    num_procedures = np.random.choice(range(0, 7), size=n,
        p=[0.30, 0.28, 0.20, 0.12, 0.06, 0.03, 0.01])

    num_diagnoses = np.clip(np.random.poisson(lam=7, size=n), 1, 16).astype(int)

    num_lab_procedures = np.clip(np.random.poisson(lam=43, size=n), 1, 120).astype(int)

    number_outpatient = np.random.choice(range(0, 6), size=n,
        p=[0.60, 0.18, 0.10, 0.07, 0.03, 0.02])

    number_emergency = np.random.choice(range(0, 5), size=n,
        p=[0.65, 0.20, 0.09, 0.04, 0.02])

    number_inpatient = np.random.choice(range(0, 6), size=n,
        p=[0.55, 0.22, 0.12, 0.07, 0.03, 0.01])

    hba1c = np.random.choice(
        ['None', 'Normal', '>7', '>8'],
        size=n, p=[0.50, 0.17, 0.18, 0.15]
    )

    glucose_serum = np.random.choice(
        ['None', 'Normal', '>200', '>300'],
        size=n, p=[0.55, 0.20, 0.15, 0.10]
    )

    insulin = np.random.choice(
        ['No', 'Steady', 'Up', 'Down'],
        size=n, p=[0.47, 0.32, 0.13, 0.08]
    )

    change_meds = np.random.choice(['No', 'Ch'], size=n, p=[0.54, 0.46])
    diabetesMed = np.random.choice(['Yes', 'No'], size=n, p=[0.77, 0.23])

    # Readmission probability (logistic model)
    logit = (
        -2.5
        + 0.025 * (age_numeric - 50)
        + 0.10 * time_in_hospital
        + 0.08 * num_medications
        + 0.12 * number_inpatient
        + 0.10 * number_emergency
        + 0.06 * num_diagnoses
        - 0.05 * num_procedures
        + 0.30 * (hba1c == '>8').astype(int)
        + 0.20 * (hba1c == '>7').astype(int)
        + 0.25 * (glucose_serum == '>300').astype(int)
        + 0.15 * (glucose_serum == '>200').astype(int)
        + 0.20 * (insulin == 'Up').astype(int)
        + 0.15 * (change_meds == 'Ch').astype(int)
        + 0.20 * (diag_primary == 'Circulatory').astype(int)
        + 0.15 * (diag_primary == 'Diabetes').astype(int)
        + np.random.normal(0, 0.3, size=n)
    )
    prob = 1 / (1 + np.exp(-logit))
    readmitted = (np.random.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame({
        'age': age_groups,
        'age_numeric': age_numeric,
        'gender': gender,
        'diag_primary': diag_primary,
        'time_in_hospital': time_in_hospital,
        'num_medications': num_medications,
        'num_procedures': num_procedures,
        'num_diagnoses': num_diagnoses,
        'num_lab_procedures': num_lab_procedures,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'hba1c_result': hba1c,
        'glucose_serum_test': glucose_serum,
        'insulin': insulin,
        'change_medications': change_meds,
        'diabetes_medication': diabetesMed,
        'readmitted_30days': readmitted
    })

    return shuffle(df, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("hospital_readmission.csv", index=False)
    print(f"Dataset gerado: {df.shape}")
    print(f"Taxa de readmissão: {df['readmitted_30days'].mean():.2%}")
    print(df.head())
