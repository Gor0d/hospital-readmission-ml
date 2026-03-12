"""
API REST para predição de risco de readmissão hospitalar.
Modelo: Rede Neural (Keras) treinada sobre dataset de pacientes hospitalares.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
import joblib
import json
import os
import tensorflow as tf

# ──────────────────────────────────────────────
# Configuração
# ──────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')

model        = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best_model.keras'))
scaler       = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
encoders     = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))

with open(os.path.join(MODEL_DIR, 'metrics.json')) as f:
    model_metrics = json.load(f)

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="Prediz o risco de readmissão hospitalar em 30 dias usando uma rede neural treinada com Keras.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ──────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────

class PatientInput(BaseModel):
    age_numeric: int = Field(..., ge=0, le=100, description="Idade do paciente em anos")
    gender: Literal["Male", "Female"]
    diag_primary: Literal[
        "Circulatory", "Respiratory", "Digestive", "Diabetes",
        "Injury", "Musculoskeletal", "Genitourinary", "Other"
    ]
    time_in_hospital: int = Field(..., ge=1, le=14)
    num_medications: int = Field(..., ge=1, le=40)
    num_procedures: int = Field(..., ge=0, le=6)
    num_diagnoses: int = Field(..., ge=1, le=16)
    num_lab_procedures: int = Field(..., ge=1, le=120)
    number_outpatient: int = Field(..., ge=0, le=5)
    number_emergency: int = Field(..., ge=0, le=4)
    number_inpatient: int = Field(..., ge=0, le=5)
    hba1c_result: Literal["None", "Normal", ">7", ">8"]
    glucose_serum_test: Literal["None", "Normal", ">200", ">300"]
    insulin: Literal["No", "Steady", "Up", "Down"]
    change_medications: Literal["No", "Ch"]
    diabetes_medication: Literal["Yes", "No"]

    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }

class PredictionResponse(BaseModel):
    readmission_probability: float
    risk_level: str
    prediction: int
    recommendation: str
    model_auc: float

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def preprocess(data: PatientInput) -> np.ndarray:
    cat_map = {
        'diag_primary':        data.diag_primary,
        'hba1c_result':        data.hba1c_result,
        'glucose_serum_test':  data.glucose_serum_test,
        'insulin':             data.insulin,
        'change_medications':  data.change_medications,
        'diabetes_medication': data.diabetes_medication,
        'gender':              data.gender,
    }
    encoded = {}
    for col, val in cat_map.items():
        le = encoders[col]
        if val not in le.classes_:
            raise HTTPException(status_code=422, detail=f"Valor inválido para '{col}': {val}")
        encoded[col + '_enc'] = int(le.transform([val])[0])

    risk_score = (
        data.number_inpatient * 2 +
        data.number_emergency +
        (1 if data.hba1c_result in ['>7', '>8'] else 0) +
        (1 if data.glucose_serum_test in ['>200', '>300'] else 0)
    )

    row = [
        data.age_numeric, encoded['gender_enc'], encoded['diag_primary_enc'],
        data.time_in_hospital, data.num_medications, data.num_procedures,
        data.num_diagnoses, data.num_lab_procedures,
        data.number_outpatient, data.number_emergency, data.number_inpatient,
        encoded['hba1c_result_enc'], encoded['glucose_serum_test_enc'],
        encoded['insulin_enc'], encoded['change_medications_enc'],
        encoded['diabetes_medication_enc'],
        risk_score,
        data.num_medications * data.num_diagnoses,
        data.number_outpatient + data.number_emergency + data.number_inpatient
    ]
    return np.array([row], dtype=float)

def classify_risk(prob: float) -> tuple[str, str]:
    if prob < 0.35:
        return "Baixo", "Paciente com baixo risco de readmissão. Seguir protocolo de alta padrão."
    elif prob < 0.60:
        return "Moderado", "Risco moderado. Considerar acompanhamento ambulatorial em 14 dias e revisão de medicamentos."
    else:
        return "Alto", "Alto risco de readmissão. Recomendar acompanhamento intensivo, revisão do plano de alta e contato ativo em 7 dias."

# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "model": "Hospital Readmission Predictor v1.0",
        "metrics": model_metrics
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientInput):
    """
    Recebe os dados clínicos e administrativos de um paciente e retorna
    a probabilidade estimada de readmissão hospitalar em 30 dias.
    """
    X = preprocess(patient)
    X_scaled = scaler.transform(X)
    prob = float(model.predict(X_scaled, verbose=0)[0][0])
    risk, recommendation = classify_risk(prob)

    return PredictionResponse(
        readmission_probability=round(prob, 4),
        risk_level=risk,
        prediction=int(prob >= 0.5),
        recommendation=recommendation,
        model_auc=model_metrics['roc_auc']
    )

@app.get("/model-info", tags=["Model"])
def model_info():
    """Retorna informações sobre o modelo em produção."""
    return {
        "model_type": "Deep Neural Network (Keras)",
        "architecture": "4 Dense layers (128→64→32→16→1) com BatchNorm e Dropout",
        "features": feature_cols,
        "metrics": model_metrics,
        "target": "readmitted_30days (binário: 0=Não, 1=Sim)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
