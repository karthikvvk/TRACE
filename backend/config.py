import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = True

    # Database
    db_path: Path = BASE_DIR / "backend" / "db" / "friday.db"

    # Privacy
    use_presidio: bool = False          # Set True once Presidio models are downloaded
    pseudonym_salt: str = "friday-dev-salt-change-me"

    # Extension origin (CORS)
    extension_origin: str = "*"         # Lock down to extension ID in production

    # Proactive engine
    check_interval_minutes: int = 10
    overdue_threshold_minutes: int = 5

    # Local LLM (Phase 4)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    use_llm: bool = False


settings = Settings()
