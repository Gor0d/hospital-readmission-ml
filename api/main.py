"""
API REST para predição de risco de readmissão hospitalar.
Modelo: Ensemble (DNN + XGBoost) com threshold otimizado.

Segurança: JWT (roles: admin, clinician, viewer)
Auditoria: LGPD Art. 11, II, f — registros sem PII
Rate limiting: slowapi
"""

import logging
import os
import sys
import json
import time
from contextlib import asynccontextmanager
from typing import Literal, List, Dict, Optional

import numpy as np
import joblib
import shap
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ──────────────────────────────────────────────
# Paths & sys.path
# ──────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.config import settings
from api.auth import (
    Token, authenticate_user, create_access_token,
    get_current_user, require_role,
    init_users_table, ensure_default_admin,
)
from api.audit import init_audit_table, log_prediction, get_audit_summary, export_audit_csv
from api.monitoring import (
    verify_model_integrity, update_realtime_gauges,
    PREDICTIONS_TOTAL, PREDICTION_LATENCY, SHAP_LATENCY,
    generate_latest, CONTENT_TYPE_LATEST,
)
from utils.preprocessing import build_feature_vector, classify_risk

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Carregamento dos artefatos
# ──────────────────────────────────────────────

MODEL_DIR = settings.model_dir

dnn_model    = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))
scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
encoders     = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
xgb_model    = joblib.load(os.path.join(MODEL_DIR, "best_model_xgb.pkl"))

# Modelo calibrado (opcional — ativado via USE_CALIBRATED_MODEL=true no .env)
calibrated_model = None
if settings.use_calibrated_model:
    _cal_path = os.path.join(MODEL_DIR, "calibrated_isotonic.pkl")
    if os.path.exists(_cal_path):
        calibrated_model = joblib.load(_cal_path)
        logger.info("Modelo calibrado carregado: %s", _cal_path)
    else:
        logger.warning("USE_CALIBRATED_MODEL=true mas arquivo não encontrado: %s", _cal_path)

# Configuração do ensemble
_ens_path = os.path.join(MODEL_DIR, "metrics_ensemble.json")
if os.path.exists(_ens_path):
    with open(_ens_path) as f:
        ensemble_config = json.load(f)
    DNN_WEIGHT    = ensemble_config["dnn_weight"]
    XGB_WEIGHT    = ensemble_config["xgb_weight"]
    THRESHOLD     = ensemble_config["best_threshold"]
    model_metrics = ensemble_config
    MODEL_TYPE    = "Ensemble (DNN + XGBoost)"
    MODEL_VERSION = ensemble_config.get("model_version", "1.0.0")
else:
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        model_metrics = json.load(f)
    DNN_WEIGHT    = 1.0
    XGB_WEIGHT    = 0.0
    THRESHOLD     = 0.5
    MODEL_TYPE    = "DNN (Keras)"
    MODEL_VERSION = model_metrics.get("model_version", "1.0.0")

# SHAP explainer — TreeExplainer sobre XGBoost
shap_explainer = shap.TreeExplainer(xgb_model)

logger.info("Modelos carregados | tipo=%s | versão=%s | threshold=%.2f | calibrado=%s",
            MODEL_TYPE, MODEL_VERSION, THRESHOLD, calibrated_model is not None)

# ──────────────────────────────────────────────
# Rate Limiter
# ──────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ──────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────

_integrity_status = {"status": "não_verificado"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_audit_table()
    init_users_table()
    ensure_default_admin()
    logger.info("Banco de auditoria inicializado: %s", settings.audit_db_path)

    # Verifica integridade dos artefatos
    global _integrity_status
    _integrity_status = verify_model_integrity(MODEL_DIR)
    if _integrity_status["status"] not in ("ok", "sem_referencia"):
        logger.error("INTEGRIDADE DOS MODELOS COMPROMETIDA — verifique os artefatos!")

    yield
    # Shutdown
    logger.info("API encerrada.")


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description=(
        "Prediz o risco de readmissão hospitalar em 30 dias usando Ensemble (DNN + XGBoost).\n\n"
        "**Autenticação:** Bearer JWT. Obtenha um token em `POST /token`.\n\n"
        "**Roles:** `viewer` (leitura), `clinician` (predições), `admin` (auditoria)."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

if settings.is_production:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts_list)


# ──────────────────────────────────────────────
# Schemas
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

    model_config = {
        "json_schema_extra": {
            "example": {
                "age_numeric": 72, "gender": "Female", "diag_primary": "Circulatory",
                "time_in_hospital": 5, "num_medications": 18, "num_procedures": 2,
                "num_diagnoses": 8, "num_lab_procedures": 45, "number_outpatient": 0,
                "number_emergency": 1, "number_inpatient": 2, "hba1c_result": ">8",
                "glucose_serum_test": ">200", "insulin": "Up",
                "change_medications": "Ch", "diabetes_medication": "Yes"
            }
        }
    }


class PredictionResponse(BaseModel):
    readmission_probability: float
    risk_level: str
    prediction: int
    recommendation: str
    model_auc: float
    threshold_used: float
    calibrated: bool = False


class ExplainResponse(BaseModel):
    readmission_probability: float
    prediction: int
    threshold_used: float
    feature_contributions: Dict[str, float]
    top_risk_factors: List[str]
    top_protective_factors: List[str]
    calibrated: bool = False


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    role: Literal["admin", "clinician", "viewer"]
    full_name: str = ""


# ──────────────────────────────────────────────
# Helper de inferência
# ──────────────────────────────────────────────

def _predict_proba(X: np.ndarray) -> tuple[float, bool]:
    """
    Retorna (probabilidade, calibrado).
    Se modelo calibrado disponível e ativado, usa-o; senão, usa ensemble bruto.
    """
    if calibrated_model is not None:
        prob = float(calibrated_model.predict_proba(X)[:, 1][0])
        return prob, True

    X_scaled = scaler.transform(X)
    prob_dnn  = float(dnn_model.predict(X_scaled, verbose=0)[0][0])
    prob_xgb  = float(xgb_model.predict_proba(X)[:, 1][0])
    return DNN_WEIGHT * prob_dnn + XGB_WEIGHT * prob_xgb, False


# ──────────────────────────────────────────────
# Endpoints públicos (sem autenticação)
# ──────────────────────────────────────────────

@app.post("/token", response_model=Token, tags=["Autenticação"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Obtém token JWT. Use as credenciais no header: Authorization: Bearer <token>"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuário ou senha incorretos.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return Token(
        access_token=token,
        token_type="bearer",
        role=user["role"],
        expires_in=settings.access_token_expire_minutes * 60,
    )


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "model": MODEL_TYPE,
        "version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "calibrated": calibrated_model is not None,
        "metrics": model_metrics,
    }


@app.get("/health", tags=["Health"])
def health():
    """Health check detalhado — inclui integridade e métricas recentes."""
    audit_ok = False
    predictions_24h = 0
    try:
        from api.audit import _get_db, get_predictions_last_n
        conn = _get_db()
        conn.execute("SELECT 1").fetchone()
        predictions_24h = conn.execute(
            "SELECT COUNT(*) FROM prediction_log WHERE timestamp >= datetime('now', '-1 day')"
        ).fetchone()[0]
        conn.close()
        audit_ok = True
    except Exception:
        pass

    update_realtime_gauges(settings.audit_db_path)

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "calibrated": calibrated_model is not None,
        "model_integrity": _integrity_status.get("status", "não_verificado"),
        "audit_db_accessible": audit_ok,
        "predictions_last_24h": predictions_24h,
        "threshold": THRESHOLD,
    }


@app.get("/metrics", tags=["Monitoramento"], include_in_schema=False)
def prometheus_metrics():
    """Expõe métricas no formato Prometheus (para scraping)."""
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ──────────────────────────────────────────────
# Endpoints clínicos (viewer+)
# ──────────────────────────────────────────────

@app.get("/model-info", tags=["Modelo"],
         dependencies=[Depends(require_role(["admin", "clinician", "viewer"]))])
def model_info():
    """Retorna informações sobre o modelo em produção."""
    return {
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "dnn_weight": DNN_WEIGHT,
        "xgb_weight": XGB_WEIGHT,
        "threshold": THRESHOLD,
        "calibrated": calibrated_model is not None,
        "features": feature_cols,
        "metrics": model_metrics,
        "target": "readmitted_30days (binário: 0=Não, 1=Sim)",
    }


# ──────────────────────────────────────────────
# Endpoints clínicos (clinician+)
# ──────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse, tags=["Predição"])
@limiter.limit(settings.rate_limit_predict)
async def predict(
    request: Request,
    patient: PatientInput,
    current_user: dict = Depends(require_role(["admin", "clinician"])),
):
    """
    Retorna a probabilidade de readmissão hospitalar em 30 dias.

    **Atenção:** Esta predição é um auxílio à decisão clínica.
    Não substitui o julgamento do profissional de saúde responsável.
    Toda predição é registrada conforme LGPD Art. 11, II, f.
    """
    try:
        X = build_feature_vector(patient.model_dump(), encoders)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        t0 = time.time()
        prob, is_calibrated = _predict_proba(X)
        latency = time.time() - t0
        PREDICTION_LATENCY.observe(latency)
    except Exception as exc:
        logger.error("Erro na inferência: %s", exc)
        raise HTTPException(status_code=500, detail="Erro interno ao processar a predição.")

    risk, recommendation = classify_risk(prob)
    PREDICTIONS_TOTAL.labels(risk_level=risk, endpoint="/predict").inc()

    result = {
        "readmission_probability": round(prob, 4),
        "risk_level": risk,
        "prediction": int(prob >= THRESHOLD),
        "recommendation": recommendation,
    }

    log_prediction(
        user_id=current_user["username"],
        patient_data=patient.model_dump(),
        prediction_result=result,
        endpoint="/predict",
        ip_address=request.client.host if request.client else "unknown",
        model_version=MODEL_VERSION,
        threshold=THRESHOLD,
    )

    logger.info("Predição | user=%s | risco=%s | prob=%.4f | latência=%.3fs",
                current_user["username"], risk, prob, latency)

    return PredictionResponse(
        **result,
        model_auc=model_metrics["roc_auc"],
        threshold_used=THRESHOLD,
        calibrated=is_calibrated,
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Explicabilidade"])
@limiter.limit(settings.rate_limit_explain)
async def explain(
    request: Request,
    patient: PatientInput,
    current_user: dict = Depends(require_role(["admin", "clinician"])),
):
    """
    Retorna a contribuição de cada feature via SHAP values
    (TreeExplainer sobre o componente XGBoost do ensemble).
    Valores positivos aumentam o risco; negativos reduzem.
    """
    try:
        X = build_feature_vector(patient.model_dump(), encoders)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        t0_shap = time.time()
        prob, is_calibrated = _predict_proba(X)
        sv = shap_explainer.shap_values(X)[0]
        SHAP_LATENCY.observe(time.time() - t0_shap)
    except Exception as exc:
        logger.error("Erro no SHAP: %s", exc)
        raise HTTPException(status_code=500, detail="Erro ao calcular explicabilidade.")

    contributions = {feat: round(float(v), 6) for feat, v in zip(feature_cols, sv)}
    sorted_contribs = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))
    top_risk       = [f for f, v in sorted_contribs.items() if v > 0][:5]
    top_protective = [f for f, v in sorted_contribs.items() if v < 0][:5]

    result = {
        "readmission_probability": round(prob, 4),
        "prediction": int(prob >= THRESHOLD),
    }

    log_prediction(
        user_id=current_user["username"],
        patient_data=patient.model_dump(),
        prediction_result={**result, "risk_level": classify_risk(prob)[0]},
        endpoint="/explain",
        ip_address=request.client.host if request.client else "unknown",
        model_version=MODEL_VERSION,
        threshold=THRESHOLD,
    )

    logger.info("SHAP | user=%s | prob=%.4f | top_risk=%s",
                current_user["username"], prob, top_risk[:3])

    return ExplainResponse(
        **result,
        threshold_used=THRESHOLD,
        feature_contributions=sorted_contribs,
        top_risk_factors=top_risk,
        top_protective_factors=top_protective,
        calibrated=is_calibrated,
    )


@app.post("/predict-batch", response_model=List[PredictionResponse], tags=["Predição"])
@limiter.limit(settings.rate_limit_batch)
async def predict_batch(
    request: Request,
    patients: List[PatientInput],
    current_user: dict = Depends(require_role(["admin", "clinician"])),
):
    """
    Recebe uma lista de pacientes e retorna a predição para cada um.
    Limite: 5 requisições/minuto por IP.
    """
    if not patients:
        raise HTTPException(status_code=422, detail="Lista de pacientes não pode ser vazia.")

    results = []
    for i, patient in enumerate(patients):
        try:
            X = build_feature_vector(patient.model_dump(), encoders)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Paciente {i}: {exc}")

        try:
            prob, is_calibrated = _predict_proba(X)
        except Exception as exc:
            logger.error("Erro na inferência do paciente %d: %s", i, exc)
            raise HTTPException(status_code=500, detail=f"Erro ao processar paciente {i}.")

        risk, recommendation = classify_risk(prob)
        result = {
            "readmission_probability": round(prob, 4),
            "risk_level": risk,
            "prediction": int(prob >= THRESHOLD),
            "recommendation": recommendation,
        }
        log_prediction(
            user_id=current_user["username"],
            patient_data=patient.model_dump(),
            prediction_result=result,
            endpoint="/predict-batch",
            ip_address=request.client.host if request.client else "unknown",
            model_version=MODEL_VERSION,
            threshold=THRESHOLD,
        )
        results.append(PredictionResponse(
            **result,
            model_auc=model_metrics["roc_auc"],
            threshold_used=THRESHOLD,
            calibrated=is_calibrated,
        ))

    logger.info("Batch | user=%s | %d pacientes | alto_risco=%d",
                current_user["username"], len(results),
                sum(1 for r in results if r.risk_level == "Alto"))
    return results


# ──────────────────────────────────────────────
# Endpoints administrativos (admin only)
# ──────────────────────────────────────────────

@app.get("/audit/summary", tags=["Auditoria"],
         dependencies=[Depends(require_role(["admin"]))])
def audit_summary(days: int = 30):
    """Resumo estatístico das predições dos últimos N dias (admin only)."""
    return get_audit_summary(days=days)


@app.get("/audit/export", tags=["Auditoria"],
         dependencies=[Depends(require_role(["admin"]))])
def audit_export(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Exporta registros de auditoria para CSV (admin only).
    Formato de data: YYYY-MM-DD. Retorna caminho do arquivo gerado.
    """
    output_path = f"./audit/export_{start_date or 'inicio'}_{end_date or 'fim'}.csv"
    count = export_audit_csv(output_path, start_date, end_date)
    return {"exported_rows": count, "file": output_path}


@app.post("/admin/users", tags=["Administração"],
          dependencies=[Depends(require_role(["admin"]))])
def create_user_endpoint(user_data: UserCreate):
    """Cria novo usuário no sistema (admin only)."""
    from api.auth import create_user, _get_db, init_users_table
    conn = _get_db()
    init_users_table(conn)
    try:
        create_user(
            user_data.username, user_data.password,
            user_data.role, user_data.full_name, conn
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    finally:
        conn.close()
    return {"message": f"Usuário '{user_data.username}' criado com role '{user_data.role}'."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
