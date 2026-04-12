"""Application settings loaded from environment variables or a .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pydantic settings model for judge and evaluation configuration."""

    model_config = SettingsConfigDict(
        env_file=Path(".env"),
        env_file_encoding="utf-8",
    )

    OPEN_ROUTER_URL: str = "https://openrouter.ai/api/v1"
    OPEN_ROUTER_API_KEY: str

    JUDGE_MODEL_NAME: str

    RAGAS_MAX_WORKERS: int = 3
    DEEPEVAL_MAX_CONCURRENT: int = 1
    DEEPEVAL_BATCH_SIZE: int = 100

    HTTP_MAX_RETRIES: int = 3
    HTTP_TIMEOUT: int = 1


settings = Settings()  # type: ignore[call-arg]
