# Guia de Deploy — Hospital Readmission ML

**Versão:** 1.0 | **Ambiente:** Produção Clínica | **Plataforma:** Linux/Docker

---

## Pré-requisitos

- Docker Engine ≥ 24.0
- Docker Compose ≥ 2.20
- 4 GB RAM (mínimo — TensorFlow + XGBoost em memória)
- 10 GB disco (modelos + logs de auditoria)
- Certificado TLS (obrigatório para produção)

---

## 1. Configuração Inicial

### 1.1 Clone e configure

```bash
git clone https://github.com/user/hospital-readmission-ml.git
cd hospital-readmission-ml

# Copia o arquivo de configuração
cp .env.example .env
```

### 1.2 Gere a SECRET_KEY

```bash
# Gera chave segura de 32 bytes (64 caracteres hex)
python -c "import secrets; print(secrets.token_hex(32))"
```

Copie a saída para o campo `SECRET_KEY` no `.env`.

### 1.3 Edite o .env

```bash
nano .env
```

Campos obrigatórios para produção:
```
SECRET_KEY=<chave-gerada-acima>
ENVIRONMENT=production
TRUSTED_HOSTS=seu-dominio.com.br,www.seu-dominio.com.br
ALLOWED_ORIGINS=https://seu-dominio.com.br
AUDIT_LOG_RETENTION_DAYS=7300
```

---

## 2. Preparação dos Modelos

### 2.1 Treine os modelos (se necessário)

```bash
# Com dados sintéticos:
python data/generate_data.py
python model/train.py
python model/train_xgboost.py
python model/train_ensemble.py

# Com dados reais (UCI dataset — requer download manual):
# python data/process_real_data.py
# python model/train.py  (mesmos comandos)
```

### 2.2 Calibre o modelo

```bash
python model/calibrate.py
# Verifica se Brier score diminuiu
```

Ative no `.env`:
```
USE_CALIBRATED_MODEL=true
```

### 2.3 Execute análise de fairness

```bash
python model/fairness.py
# Verifique model/fairness_report.json — não deve haver alertas
```

### 2.4 Gere checksums e baseline

```bash
python model/compute_baseline.py
# Cria model/checksums.sha256 e model/baseline_stats.json
```

---

## 3. Deploy com Docker Compose

### 3.1 Build e start

```bash
docker compose build
docker compose up -d
```

### 3.2 Verifique o health check

```bash
curl http://localhost:8000/health
# Esperado: {"status": "healthy", "model_integrity": "ok", ...}
```

### 3.3 Crie o primeiro usuário admin

```bash
# Com a API rodando:
curl -X POST http://localhost:8000/token \
  -d "username=admin&password=admin123"

# Obtenha o token e crie usuários clínicos:
TOKEN=$(curl -s -X POST http://localhost:8000/token \
  -d "username=admin&password=admin123" | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -X POST http://localhost:8000/admin/users \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"username":"dr_silva","password":"senha_segura_456","role":"clinician","full_name":"Dr. João Silva"}'
```

> ⚠ **IMPORTANTE:** Altere a senha do usuário `admin` imediatamente após o primeiro login em produção!

---

## 4. Configuração HTTPS (nginx reverse proxy)

### 4.1 Arquivo nginx.conf

```nginx
server {
    listen 443 ssl;
    server_name seu-dominio.com.br;

    ssl_certificate     /etc/ssl/certs/seu-dominio.crt;
    ssl_certificate_key /etc/ssl/private/seu-dominio.key;
    ssl_protocols       TLSv1.2 TLSv1.3;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;  # para SHAP
    }
}

server {
    listen 80;
    server_name seu-dominio.com.br;
    return 301 https://$server_name$request_uri;
}
```

---

## 5. Monitoramento

### 5.1 Prometheus + Grafana

Acesse: `http://localhost:9090` (Prometheus)

Métricas disponíveis:
- `readmission_predictions_total` — total por risk_level e endpoint
- `readmission_prediction_duration_seconds` — latência
- `readmission_high_risk_ratio_last_100` — proporção de alto risco
- `readmission_model_integrity` — integridade dos artefatos (1=OK)
- `readmission_audit_db_accessible` — disponibilidade do banco

### 5.2 Logs de auditoria

```bash
# Resumo dos últimos 30 dias
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8000/audit/summary?days=30

# Exportar para análise
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  "http://localhost:8000/audit/export?start_date=2026-01-01&end_date=2026-03-31"
```

---

## 6. Backup

```bash
# Backup diário do banco de auditoria (crontab)
0 2 * * * tar -czf /backup/audit_$(date +%Y%m%d).tar.gz /app/audit/audit.db

# Backup dos modelos (após re-treinamento)
cp -r model/ /backup/models_$(date +%Y%m%d)/
```

---

## 7. Re-treinamento

Quando re-treinar (drift de dados, novos dados, etc.):

1. Treine novos modelos em branch separado
2. Execute calibração e fairness
3. Execute compute_baseline (novos checksums)
4. Teste com `pytest tests/` — todos devem passar
5. Swap de modelos: copie novos artefatos para `model/`
6. Reinicie a API: `docker compose restart api`
7. Verifique: `curl /health` → `model_integrity: ok`

---

## 8. Resolução de Problemas

| Problema | Diagnóstico | Solução |
|---|---|---|
| API não sobe | `docker compose logs api` | Verifique MODEL_DIR e arquivos .keras/.pkl |
| `model_integrity: COMPROMETIDA` | Checksums divergentes | Re-generate checksums ou restaure backup |
| `audit_db_accessible: false` | Permissões no volume | `chown -R appuser /app/audit` |
| Latência alta (> 2s) | SHAP em CPU | Normal — considere cache de SHAP |
| 429 Too Many Requests | Rate limit atingido | Aumente RATE_LIMIT_PREDICT no .env |
| JWT expirado | Token vencido | Re-autentique via POST /token |
