"""
Módulo de auditoria LGPD para a API clínica.

Registra cada predição em SQLite SEM armazenar dados pessoais identificáveis (PII).
O paciente é representado por um hash SHA-256 das features de entrada,
permitindo rastrear re-submissões sem expor identidade.

Base legal LGPD: Art. 11, II, f — tutela da saúde, em procedimento
realizado por profissionais da área da saúde ou por entidades sanitárias.

Conforme CFM e legislação vigente, os registros são retidos por
AUDIT_LOG_RETENTION_DAYS dias (padrão: 7300 ≈ 20 anos).
"""

import csv
import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from api.config import settings

logger = logging.getLogger(__name__)

LEGAL_BASIS = "Legítimo interesse — tutela da saúde (LGPD Art. 11, II, f)"


# ──────────────────────────────────────────────
# Init
# ──────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    db_path = settings.audit_db_path
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_audit_table(conn: Optional[sqlite3.Connection] = None) -> None:
    """Cria tabelas de auditoria se não existirem."""
    close = conn is None
    if conn is None:
        conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT NOT NULL,
            user_id             TEXT NOT NULL,
            patient_hash        TEXT NOT NULL,
            endpoint            TEXT NOT NULL,
            prediction          INTEGER NOT NULL,
            probability         REAL NOT NULL,
            risk_level          TEXT NOT NULL,
            confidence_band     TEXT NOT NULL,
            feature_hash        TEXT NOT NULL,
            model_version       TEXT NOT NULL,
            ip_address_hash     TEXT NOT NULL,
            legal_basis         TEXT NOT NULL,
            deletion_scheduled  TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_prediction_log_timestamp
            ON prediction_log (timestamp);

        CREATE INDEX IF NOT EXISTS idx_prediction_log_user
            ON prediction_log (user_id);
    """)
    conn.commit()
    if close:
        conn.close()


# ──────────────────────────────────────────────
# Helpers internos
# ──────────────────────────────────────────────

def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _confidence_band(prob: float, threshold: float) -> str:
    """Classifica a distância da probabilidade em relação ao threshold."""
    dist = abs(prob - threshold)
    if dist < 0.05:
        return "limítrofe"   # dentro de ±5% do threshold — maior incerteza clínica
    elif dist < 0.15:
        return "moderada"
    return "alta"


def _schedule_deletion(retention_days: int) -> str:
    from datetime import timedelta
    return (
        datetime.now(timezone.utc) + timedelta(days=retention_days)
    ).isoformat()


# ──────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────

def log_prediction(
    user_id: str,
    patient_data: dict,
    prediction_result: dict,
    endpoint: str,
    ip_address: str,
    model_version: str,
    threshold: float,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[int]:
    """
    Registra uma predição na tabela de auditoria.

    Retorna o id do registro inserido, ou None em caso de falha.
    Falhas de auditoria NÃO propagam exceção para não bloquear o atendimento clínico.
    """
    try:
        close = conn is None
        if conn is None:
            conn = _get_db()
            init_audit_table(conn)

        # Hash das features de entrada — sem PII
        patient_hash = _sha256(json.dumps(patient_data, sort_keys=True))
        feature_hash = _sha256(str(sorted(patient_data.items())))
        ip_hash = _sha256(ip_address or "unknown")

        prob = prediction_result.get("readmission_probability", 0.0)
        confidence = _confidence_band(prob, threshold)

        cursor = conn.execute(
            """INSERT INTO prediction_log
               (timestamp, user_id, patient_hash, endpoint, prediction, probability,
                risk_level, confidence_band, feature_hash, model_version,
                ip_address_hash, legal_basis, deletion_scheduled)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                user_id,
                patient_hash,
                endpoint,
                prediction_result.get("prediction", -1),
                prob,
                prediction_result.get("risk_level", ""),
                confidence,
                feature_hash,
                model_version,
                ip_hash,
                LEGAL_BASIS,
                _schedule_deletion(settings.audit_log_retention_days),
            ),
        )
        conn.commit()
        log_id = cursor.lastrowid

        if close:
            conn.close()

        return log_id

    except Exception as exc:
        logger.error("Falha na auditoria (não bloqueia predição): %s", exc)
        return None


def get_audit_summary(days: int = 30, conn: Optional[sqlite3.Connection] = None) -> dict:
    """Retorna resumo estatístico dos últimos N dias."""
    close = conn is None
    if conn is None:
        conn = _get_db()

    since = datetime.now(timezone.utc).isoformat()[:10]  # simplificado para data

    rows = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN risk_level = 'Alto' THEN 1 ELSE 0 END) as alto_risco,
            SUM(CASE WHEN risk_level = 'Moderado' THEN 1 ELSE 0 END) as moderado,
            SUM(CASE WHEN risk_level = 'Baixo' THEN 1 ELSE 0 END) as baixo_risco,
            SUM(CASE WHEN confidence_band = 'limítrofe' THEN 1 ELSE 0 END) as limitrofe,
            AVG(probability) as prob_media,
            COUNT(DISTINCT user_id) as usuarios_ativos
        FROM prediction_log
        WHERE timestamp >= date('now', ?)
    """, (f"-{days} days",)).fetchone()

    if close:
        conn.close()

    return {
        "periodo_dias": days,
        "total_predicoes": rows["total"] or 0,
        "distribuicao_risco": {
            "alto": rows["alto_risco"] or 0,
            "moderado": rows["moderado"] or 0,
            "baixo": rows["baixo_risco"] or 0,
        },
        "predicoes_limitrofes": rows["limitrofe"] or 0,
        "probabilidade_media": round(rows["prob_media"] or 0.0, 4),
        "usuarios_ativos": rows["usuarios_ativos"] or 0,
    }


def get_predictions_last_n(n: int = 100, conn: Optional[sqlite3.Connection] = None) -> list:
    """Retorna as últimas N predições para monitoramento de deriva."""
    close = conn is None
    if conn is None:
        conn = _get_db()
    rows = conn.execute(
        "SELECT probability, risk_level FROM prediction_log ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    if close:
        conn.close()
    return [dict(r) for r in rows]


def export_audit_csv(
    output_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """
    Exporta registros de auditoria para CSV.
    Retorna o número de linhas exportadas.
    Campos PII (ip_address_hash, patient_hash) são incluídos como hashes — não identificáveis.
    """
    close = conn is None
    if conn is None:
        conn = _get_db()

    query = "SELECT * FROM prediction_log WHERE 1=1"
    params = []
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date + "T23:59:59")

    rows = conn.execute(query, params).fetchall()
    if close:
        conn.close()

    if not rows:
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows([dict(r) for r in rows])

    return len(rows)
