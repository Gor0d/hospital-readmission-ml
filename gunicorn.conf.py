"""
Configuração do Gunicorn para produção.
Otimizado para modelos TensorFlow (carregamento pesado) e SHAP (latência variável).
"""

import multiprocessing

# Workers: 2 para modelos ML pesados em memória (evita OOM com TF)
# Fórmula padrão seria 2*CPU+1, mas TF ocupa ~1GB por worker
workers = 2
worker_class = "uvicorn.workers.UvicornWorker"

# Timeouts: SHAP pode levar até 2-3s; startup do TF até 30s
timeout = 120
graceful_timeout = 30
keepalive = 5

# Binding
bind = "0.0.0.0:8000"

# Logging
accesslog = "-"   # stdout
errorlog = "-"    # stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s %(f)s %(a)s'

# Segurança
limit_request_line = 4094
limit_request_fields = 100

# Reload apenas em desenvolvimento
reload = False
