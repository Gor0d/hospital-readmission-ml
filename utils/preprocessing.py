"""
Utilitários de pré-processamento compartilhados entre API e Dashboard.
Centraliza a lógica de feature engineering e classificação de risco.
"""

import numpy as np

CAT_COLS = [
    'diag_primary', 'hba1c_result', 'glucose_serum_test',
    'insulin', 'change_medications', 'diabetes_medication', 'gender'
]

# Limiares de risco (usados em API e Dashboard)
RISK_LOW      = 0.35
RISK_MODERATE = 0.60


def build_feature_vector(data: dict, encoders: dict) -> np.ndarray:
    """
    Constrói o vetor de features a partir de um dicionário de dados do paciente.

    Args:
        data:     dict com os campos do paciente (mesmas chaves do PatientInput)
        encoders: dict de LabelEncoders indexados pelo nome da coluna

    Returns:
        numpy array de shape (1, 19)

    Raises:
        ValueError: se um valor categórico não for reconhecido pelo encoder
    """
    encoded = {}
    for col in CAT_COLS:
        le  = encoders[col]
        val = data[col]
        # Pandas converte 'None' → NaN ao ler o CSV, então o encoder foi treinado com NaN.
        # Aqui replicamos essa conversão para inputs vindos da API/Dashboard.
        if val == 'None':
            val = np.nan
        if val not in le.classes_:
            raise ValueError(
                f"Valor inválido para '{col}': '{val}'. "
                f"Valores aceitos: {list(le.classes_)}"
            )
        encoded[col + '_enc'] = int(le.transform([val])[0])

    risk_score = (
        data['number_inpatient'] * 2 +
        data['number_emergency'] +
        (1 if data['hba1c_result'] in ['>7', '>8'] else 0) +
        (1 if data['glucose_serum_test'] in ['>200', '>300'] else 0)
    )

    row = [
        data['age_numeric'],          encoded['gender_enc'],
        encoded['diag_primary_enc'],  data['time_in_hospital'],
        data['num_medications'],      data['num_procedures'],
        data['num_diagnoses'],        data['num_lab_procedures'],
        data['number_outpatient'],    data['number_emergency'],
        data['number_inpatient'],     encoded['hba1c_result_enc'],
        encoded['glucose_serum_test_enc'], encoded['insulin_enc'],
        encoded['change_medications_enc'], encoded['diabetes_medication_enc'],
        risk_score,
        data['num_medications'] * data['num_diagnoses'],
        data['number_outpatient'] + data['number_emergency'] + data['number_inpatient'],
    ]
    return np.array([row], dtype=float)


def classify_risk(prob: float) -> tuple:
    """
    Classifica o nível de risco e retorna uma recomendação clínica.

    Returns:
        (risk_level: str, recommendation: str)
    """
    if prob < RISK_LOW:
        return (
            "Baixo",
            "Paciente com baixo risco de readmissão. Seguir protocolo de alta padrão."
        )
    elif prob < RISK_MODERATE:
        return (
            "Moderado",
            "Risco moderado. Considerar acompanhamento ambulatorial em 14 dias e revisão de medicamentos."
        )
    else:
        return (
            "Alto",
            "Alto risco de readmissão. Recomendar acompanhamento intensivo, "
            "revisão do plano de alta e contato ativo em 7 dias."
        )
