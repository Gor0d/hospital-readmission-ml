"""
Configuração centralizada da aplicação via variáveis de ambiente.
Lê do arquivo .env automaticamente (em desenvolvimento).
Em produção, use variáveis de ambiente do sistema ou secrets manager.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Autenticação
    secret_key: str = Field(default="dev-insecure-key-change-in-production-32chars")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Auditoria / LGPD
    audit_db_path: str = "./audit/audit.db"
    audit_log_retention_days: int = 7300  # ~20 anos (CFM)

    # Modelo
    model_dir: str = "./model"
    use_calibrated_model: bool = False

    # API
    allowed_origins: str = "http://localhost:8501,http://localhost:8000"
    environment: str = "development"
    trusted_hosts: str = "localhost,127.0.0.1"

    # Rate limiting
    rate_limit_predict: str = "30/minute"
    rate_limit_batch: str = "5/minute"
    rate_limit_explain: str = "20/minute"

    # Monitoramento
    psi_alert_threshold: float = 0.2
    min_subgroup_auc: float = 0.60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def trusted_hosts_list(self) -> list[str]:
        return [h.strip() for h in self.trusted_hosts.split(",") if h.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


settings = Settings()
