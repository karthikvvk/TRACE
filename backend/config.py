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

    # Browser bridge
    extension_secret: str = "change-me-in-production"  # Shared secret for WS handshake
    allow_script_execution: bool = False                # Gate browser_execute_script tool

    # ── Colab Brain Offload (optional module — remove backend/bridge/ to disable) ──
    colab_secret: str = "colab-change-me"   # COLAB_SECRET — shared with the Colab notebook
    colab_mode: bool = False                # COLAB_MODE=true — route /chat through Colab brain

    # MCP (Model Context Protocol) — exposes tools over /mcp (Streamable HTTP)
    mcp_enabled: bool = True            # MCP_ENABLED=false to disable the /mcp endpoint

    # Proactive engine
    check_interval_minutes: int = 10
    overdue_threshold_minutes: int = 5

    # Gemini
    # gemini_api_key: str = ""                            # GEMINI_API_KEY env var
    # gemini_model: str = "gemini-2.5-flash-lite"

    # Local LLM — works with any OpenAI-compatible server (LM Studio, Ollama, etc.)
    # LM Studio default: http://localhost:1234/v1
    # Ollama default:    http://localhost:11434/v1
    ollama_base_url: str = "http://localhost:1234/v1"
    ollama_model: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    use_llm: bool = True


settings = Settings()
