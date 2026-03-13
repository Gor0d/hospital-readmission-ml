"""
API REST para predição de risco de readmissão hospitalar.
Modelo: Rede Neural (Keras) treinada sobre dataset de pacientes hospitalares.
"""

import logging
import os
import sys
import json

import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List

# ──────────────────────────────────────────────
# Path para importar utils do pacote raiz
# ──────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, '..')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.preprocessing import build_feature_vector, classify_risk  # noqa: E402

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Artefatos do modelo
# ──────────────────────────────────────────────

model        = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best_model.keras'))
scaler       = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
encoders     = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))

with open(os.path.join(MODEL_DIR, 'metrics.json')) as f:
    model_metrics = json.load(f)

logger.info("Modelo e artefatos carregados com sucesso.")

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="Prediz o risco de readmissão hospitalar em 30 dias usando uma rede neural treinada com Keras.",
    version="1.0.0"
)

# CORS: restringe origens via variável de ambiente ALLOWED_ORIGINS.
# Padrão: localhost (Streamlit + docs interativos).
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:8000")
allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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
# Endpoints
# ──────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "model": "Hospital Readmission Predictor v1.0",
        "metrics": model_metrics,
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
    try:
        X = build_feature_vector(patient.model_dump(), encoders)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        X_scaled = scaler.transform(X)
        prob = float(model.predict(X_scaled, verbose=0)[0][0])
    except Exception as exc:
        logger.error("Erro na inferência do modelo: %s", exc)
        raise HTTPException(status_code=500, detail="Erro interno ao processar a predição.")

    risk, recommendation = classify_risk(prob)
    logger.info("Predição | risco=%s | prob=%.4f | diag=%s | idade=%d",
                risk, prob, patient.diag_primary, patient.age_numeric)

    return PredictionResponse(
        readmission_probability=round(prob, 4),
        risk_level=risk,
        prediction=int(prob >= 0.5),
        recommendation=recommendation,
        model_auc=model_metrics['roc_auc'],
    )


@app.post("/predict-batch", response_model=List[PredictionResponse], tags=["Prediction"])
def predict_batch(patients: List[PatientInput]):
    """
    Recebe uma lista de pacientes e retorna a probabilidade de readmissão para cada um.
    Útil para integração hospitalar com múltiplos registros simultâneos.
    """
    results = []
    for i, patient in enumerate(patients):
        try:
            X = build_feature_vector(patient.model_dump(), encoders)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Paciente {i}: {exc}")

        try:
            X_scaled = scaler.transform(X)
            prob = float(model.predict(X_scaled, verbose=0)[0][0])
        except Exception as exc:
            logger.error("Erro na inferência do paciente %d: %s", i, exc)
            raise HTTPException(status_code=500, detail=f"Erro ao processar paciente {i}.")

        risk, recommendation = classify_risk(prob)
        results.append(PredictionResponse(
            readmission_probability=round(prob, 4),
            risk_level=risk,
            prediction=int(prob >= 0.5),
            recommendation=recommendation,
            model_auc=model_metrics['roc_auc'],
        ))

    logger.info("Batch concluído | %d pacientes | alto_risco=%d",
                len(results), sum(1 for r in results if r.risk_level == "Alto"))
    return results


@app.get("/model-info", tags=["Model"])
def model_info():
    """Retorna informações sobre o modelo em produção."""
    return {
        "model_type": "Deep Neural Network (Keras)",
        "architecture": "4 Dense layers (128→64→32→16→1) com BatchNorm e Dropout",
        "features": feature_cols,
        "metrics": model_metrics,
        "target": "readmitted_30days (binário: 0=Não, 1=Sim)",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
