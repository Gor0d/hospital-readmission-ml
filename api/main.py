"""
API REST para predição de risco de readmissão hospitalar.
Modelo: Ensemble (DNN + XGBoost) com threshold otimizado.
"""

import logging
import os
import sys
import json

import numpy as np
import joblib
import shap
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List, Dict

# ──────────────────────────────────────────────
# Paths
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
# Artefatos
# ──────────────────────────────────────────────

dnn_model    = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best_model.keras'))
scaler       = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
encoders     = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))
xgb_model    = joblib.load(os.path.join(MODEL_DIR, 'best_model_xgb.pkl'))

# Configuração do ensemble (pesos + threshold otimizado)
_ens_path = os.path.join(MODEL_DIR, 'metrics_ensemble.json')
if os.path.exists(_ens_path):
    with open(_ens_path) as f:
        ensemble_config = json.load(f)
    DNN_WEIGHT    = ensemble_config['dnn_weight']
    XGB_WEIGHT    = ensemble_config['xgb_weight']
    THRESHOLD     = ensemble_config['best_threshold']
    model_metrics = ensemble_config
    MODEL_TYPE    = 'Ensemble (DNN + XGBoost)'
else:
    with open(os.path.join(MODEL_DIR, 'metrics.json')) as f:
        model_metrics = json.load(f)
    DNN_WEIGHT    = 1.0
    XGB_WEIGHT    = 0.0
    THRESHOLD     = 0.5
    MODEL_TYPE    = 'DNN (Keras)'

# SHAP explainer — TreeExplainer sobre XGBoost (rápido para real-time)
shap_explainer = shap.TreeExplainer(xgb_model)

logger.info("Modelos carregados | tipo=%s | threshold=%.2f", MODEL_TYPE, THRESHOLD)

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="Prediz o risco de readmissão hospitalar em 30 dias usando Ensemble (DNN + XGBoost) com threshold otimizado.",
    version="2.0.0"
)

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
    age_numeric: int = Field(..., ge=0, le=100)
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
                "age_numeric": 72, "gender": "Female", "diag_primary": "Circulatory",
                "time_in_hospital": 5, "num_medications": 18, "num_procedures": 2,
                "num_diagnoses": 8, "num_lab_procedures": 45, "number_outpatient": 0,
                "number_emergency": 1, "number_inpatient": 2, "hba1c_result": ">8",
                "glucose_serum_test": ">200", "insulin": "Up",
                "change_medications": "Ch", "diabetes_medication": "Yes"
            }
        }


class PredictionResponse(BaseModel):
    readmission_probability: float
    risk_level: str
    prediction: int
    recommendation: str
    model_auc: float
    threshold_used: float


class ExplainResponse(BaseModel):
    readmission_probability: float
    prediction: int
    threshold_used: float
    feature_contributions: Dict[str, float]
    top_risk_factors: List[str]
    top_protective_factors: List[str]


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _predict_proba(X: np.ndarray) -> float:
    X_scaled = scaler.transform(X)
    prob_dnn  = float(dnn_model.predict(X_scaled, verbose=0)[0][0])
    prob_xgb  = float(xgb_model.predict_proba(X)[:, 1][0])
    return DNN_WEIGHT * prob_dnn + XGB_WEIGHT * prob_xgb


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "model": MODEL_TYPE,
        "threshold": THRESHOLD,
        "metrics": model_metrics,
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_loaded": True, "model_type": MODEL_TYPE}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientInput):
    """
    Retorna a probabilidade de readmissão hospitalar em 30 dias
    usando o Ensemble (DNN + XGBoost) com threshold otimizado.
    """
    try:
        X = build_feature_vector(patient.model_dump(), encoders)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        prob = _predict_proba(X)
    except Exception as exc:
        logger.error("Erro na inferência: %s", exc)
        raise HTTPException(status_code=500, detail="Erro interno ao processar a predição.")

    risk, recommendation = classify_risk(prob)
    logger.info("Predição | risco=%s | prob=%.4f | threshold=%.2f | diag=%s | idade=%d",
                risk, prob, THRESHOLD, patient.diag_primary, patient.age_numeric)

    return PredictionResponse(
        readmission_probability=round(prob, 4),
        risk_level=risk,
        prediction=int(prob >= THRESHOLD),
        recommendation=recommendation,
        model_auc=model_metrics['roc_auc'],
        threshold_used=THRESHOLD,
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain(patient: PatientInput):
    """
    Retorna a contribuição de cada feature na predição via SHAP values
    (TreeExplainer sobre o componente XGBoost do ensemble).
    Valores positivos aumentam o risco; negativos reduzem.
    """
    try:
        X = build_feature_vector(patient.model_dump(), encoders)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        prob = _predict_proba(X)
        sv   = shap_explainer.shap_values(X)[0]
    except Exception as exc:
        logger.error("Erro no SHAP: %s", exc)
        raise HTTPException(status_code=500, detail="Erro ao calcular explicabilidade.")

    contributions = {feat: round(float(v), 6) for feat, v in zip(feature_cols, sv)}
    sorted_contribs = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))

    top_risk       = [f for f, v in sorted_contribs.items() if v > 0][:5]
    top_protective = [f for f, v in sorted_contribs.items() if v < 0][:5]

    logger.info("SHAP | prob=%.4f | top_risk=%s", prob, top_risk[:3])

    return ExplainResponse(
        readmission_probability=round(prob, 4),
        prediction=int(prob >= THRESHOLD),
        threshold_used=THRESHOLD,
        feature_contributions=sorted_contribs,
        top_risk_factors=top_risk,
        top_protective_factors=top_protective,
    )


@app.post("/predict-batch", response_model=List[PredictionResponse], tags=["Prediction"])
def predict_batch(patients: List[PatientInput]):
    """
    Recebe uma lista de pacientes e retorna a probabilidade de readmissão para cada um.
    """
    results = []
    for i, patient in enumerate(patients):
        try:
            X = build_feature_vector(patient.model_dump(), encoders)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Paciente {i}: {exc}")

        try:
            prob = _predict_proba(X)
        except Exception as exc:
            logger.error("Erro na inferência do paciente %d: %s", i, exc)
            raise HTTPException(status_code=500, detail=f"Erro ao processar paciente {i}.")

        risk, recommendation = classify_risk(prob)
        results.append(PredictionResponse(
            readmission_probability=round(prob, 4),
            risk_level=risk,
            prediction=int(prob >= THRESHOLD),
            recommendation=recommendation,
            model_auc=model_metrics['roc_auc'],
            threshold_used=THRESHOLD,
        ))

    logger.info("Batch | %d pacientes | alto_risco=%d | threshold=%.2f",
                len(results), sum(1 for r in results if r.risk_level == "Alto"), THRESHOLD)
    return results


@app.get("/model-info", tags=["Model"])
def model_info():
    """Retorna informações sobre o modelo em produção."""
    return {
        "model_type": MODEL_TYPE,
        "dnn_weight": DNN_WEIGHT,
        "xgb_weight": XGB_WEIGHT,
        "threshold": THRESHOLD,
        "features": feature_cols,
        "metrics": model_metrics,
        "target": "readmitted_30days (binário: 0=Não, 1=Sim)",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
