# ============================================================
# Dockerfile — API de Predição de Readmissão Hospitalar
# Build multi-stage para imagem de produção enxuta e segura
# ============================================================

# ── Estágio 1: Builder ──────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Instala dependências do sistema (necessárias para compilar alguns pacotes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements primeiro (aproveita cache de camadas)
COPY requirements.txt .

# Instala dependências em diretório isolado
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Estágio 2: Produção ─────────────────────────────────────
FROM python:3.11-slim AS production

LABEL maintainer="hospital-readmission-ml"
LABEL description="API de predição de readmissão hospitalar — uso clínico"

# Cria usuário não-root para segurança
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copia dependências instaladas do estágio builder
COPY --from=builder /install /usr/local

# Copia apenas o código necessário (sem dados de treino, .git, etc.)
COPY api/       ./api/
COPY utils/     ./utils/
COPY model/     ./model/
COPY gunicorn.conf.py .

# Cria diretório de auditoria com permissões corretas
RUN mkdir -p /app/audit && chown -R appuser:appgroup /app

# Muda para usuário não-root
USER appuser

# Expõe porta da API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Comando de inicialização (gunicorn + uvicorn workers)
CMD ["gunicorn", "api.main:app", \
     "--config", "gunicorn.conf.py", \
     "--bind", "0.0.0.0:8000"]
