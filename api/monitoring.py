"""
Monitoramento da API clínica.

Funcionalidades:
  1. Métricas Prometheus (GET /metrics): contadores, histogramas, gauges
  2. PSI (Population Stability Index): detecção de deriva de dados
  3. Integridade dos artefatos: verificação SHA-256 no startup e /health
  4. Resumo de predições recentes para o /health detalhado
"""

import hashlib
import json
import logging
import os
from typing import Optional

import numpy as np
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Métricas Prometheus
# ──────────────────────────────────────────────

PREDICTIONS_TOTAL = Counter(
    "readmission_predictions_total",
    "Total de predições realizadas",
    ["risk_level", "endpoint"]
)

PREDICTION_LATENCY = Histogram(
    "readmission_prediction_duration_seconds",
    "Latência das predições (segundos)",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

SHAP_LATENCY = Histogram(
    "readmission_shap_duration_seconds",
    "Latência dos cálculos SHAP (segundos)",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

HIGH_RISK_RATIO = Gauge(
    "readmission_high_risk_ratio_last_100",
    "Proporção de predições Alto Risco nas últimas 100"
)

MODEL_INTEGRITY = Gauge(
    "readmission_model_integrity",
    "Integridade do artefato (1=OK, 0=divergente)",
    ["artifact"]
)

AUDIT_DB_ACCESSIBLE = Gauge(
    "readmission_audit_db_accessible",
    "Banco de auditoria acessível (1=sim, 0=não)"
)

PREDICTIONS_LAST_24H = Gauge(
    "readmission_predictions_last_24h",
    "Total de predições nas últimas 24 horas"
)

# ──────────────────────────────────────────────
# Integridade dos artefatos
# ──────────────────────────────────────────────

CHECKSUM_FILE = "checksums.sha256"

MONITORED_ARTIFACTS = [
    "best_model.keras",
    "best_model_xgb.pkl",
    "scaler.pkl",
    "encoders.pkl",
    "feature_cols.pkl",
]


def _sha256_file(path: str) -> str:
    """Calcula SHA-256 de um arquivo em chunks (suporte a arquivos grandes)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return ""


def generate_checksums(model_dir: str) -> dict:
    """Gera arquivo de checksums para os artefatos do modelo."""
    checksums = {}
    for artifact in MONITORED_ARTIFACTS:
        path = os.path.join(model_dir, artifact)
        if os.path.exists(path):
            checksums[artifact] = _sha256_file(path)
            logger.info("SHA-256 [%s]: %s", artifact, checksums[artifact][:16] + "...")

    checksum_path = os.path.join(model_dir, CHECKSUM_FILE)
    with open(checksum_path, "w") as f:
        for name, digest in checksums.items():
            f.write(f"{digest}  {name}\n")
    logger.info("Checksums gerados: %s", checksum_path)
    return checksums


def verify_model_integrity(model_dir: str) -> dict:
    """
    Verifica integridade dos artefatos comparando SHA-256 com checksums.sha256.
    Atualiza métricas Prometheus.

    Returns:
        dict com status de cada artefato e status geral
    """
    checksum_path = os.path.join(model_dir, CHECKSUM_FILE)
    results = {}

    if not os.path.exists(checksum_path):
        logger.warning("Arquivo de checksums não encontrado: %s", checksum_path)
        logger.warning("Execute model/compute_baseline.py para gerar checksums.")
        for artifact in MONITORED_ARTIFACTS:
            MODEL_INTEGRITY.labels(artifact=artifact).set(1)  # assume OK se não há referência
        return {"status": "sem_referencia", "detalhes": {}}

    # Carrega checksums de referência
    reference = {}
    with open(checksum_path) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("  ", 1)
                if len(parts) == 2:
                    reference[parts[1]] = parts[0]

    # Verifica cada artefato
    all_ok = True
    for artifact in MONITORED_ARTIFACTS:
        path = os.path.join(model_dir, artifact)
        if artifact not in reference:
            results[artifact] = "sem_referencia"
            MODEL_INTEGRITY.labels(artifact=artifact).set(1)
            continue

        current = _sha256_file(path)
        ok = current == reference[artifact]
        results[artifact] = "ok" if ok else "DIVERGENTE"
        MODEL_INTEGRITY.labels(artifact=artifact).set(1 if ok else 0)

        if not ok:
            all_ok = False
            logger.error("INTEGRIDADE COMPROMETIDA: %s (esperado: %s, atual: %s)",
                         artifact, reference[artifact][:16], current[:16])

    overall = "ok" if all_ok else "COMPROMETIDA"
    if all_ok:
        logger.info("Integridade dos modelos: OK")
    else:
        logger.error("ALERTA: Integridade dos modelos comprometida!")

    return {"status": overall, "detalhes": results}


# ──────────────────────────────────────────────
# PSI — Population Stability Index
# ──────────────────────────────────────────────

def compute_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Calcula o PSI entre distribuição de referência e atual.

    PSI < 0.1  → estável
    PSI 0.1-0.2 → ligeira deriva, monitorar
    PSI > 0.2  → deriva significativa, investigar

    Referência: Siddiqi, N. (2006). Credit Risk Scorecards.
    """
    eps = 1e-6
    # Define bins baseados na distribuição baseline
    bin_edges = np.percentile(baseline, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)  # remove duplicatas

    baseline_counts = np.histogram(baseline, bins=bin_edges)[0]
    current_counts  = np.histogram(current, bins=bin_edges)[0]

    baseline_pct = (baseline_counts + eps) / (len(baseline) + eps * len(baseline_counts))
    current_pct  = (current_counts + eps)  / (len(current)  + eps * len(current_counts))

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def check_prediction_drift(
    recent_probs: list[float],
    model_dir: str,
    threshold: float = 0.2,
) -> dict:
    """
    Verifica deriva na distribuição das probabilidades preditas.
    Compara as N predições mais recentes contra a distribuição baseline.
    """
    baseline_path = os.path.join(model_dir, "baseline_stats.json")
    if not os.path.exists(baseline_path) or not recent_probs:
        return {"status": "sem_baseline", "psi": None}

    with open(baseline_path) as f:
        baseline_stats = json.load(f)

    baseline_probs = np.array(baseline_stats.get("prediction_distribution", []))
    if len(baseline_probs) < 50:
        return {"status": "baseline_insuficiente", "psi": None}

    psi = compute_psi(baseline_probs, np.array(recent_probs))
    status = "ok" if psi < 0.1 else ("monitorar" if psi < threshold else "ALERTA")

    if status == "ALERTA":
        logger.warning("PSI de predições: %.4f — deriva significativa detectada!", psi)

    return {
        "psi": round(psi, 4),
        "status": status,
        "n_recentes": len(recent_probs),
        "n_baseline": len(baseline_probs),
    }


# ──────────────────────────────────────────────
# Atualização de gauges a partir do banco de auditoria
# ──────────────────────────────────────────────

def update_realtime_gauges(audit_db_path: str) -> None:
    """
    Lê do banco de auditoria e atualiza gauges Prometheus.
    Chamado periodicamente ou em cada request de /health.
    """
    import sqlite3
    try:
        conn = sqlite3.connect(audit_db_path, check_same_thread=False)

        # Proporção de alto risco nas últimas 100 predições
        rows = conn.execute(
            "SELECT risk_level FROM prediction_log ORDER BY id DESC LIMIT 100"
        ).fetchall()
        if rows:
            high_risk_count = sum(1 for r in rows if r[0] == "Alto")
            HIGH_RISK_RATIO.set(high_risk_count / len(rows))

        # Predições nas últimas 24h
        count_24h = conn.execute(
            "SELECT COUNT(*) FROM prediction_log WHERE timestamp >= datetime('now', '-1 day')"
        ).fetchone()[0]
        PREDICTIONS_LAST_24H.set(count_24h)

        AUDIT_DB_ACCESSIBLE.set(1)
        conn.close()
    except Exception as exc:
        logger.warning("Falha ao atualizar gauges de auditoria: %s", exc)
        AUDIT_DB_ACCESSIBLE.set(0)
