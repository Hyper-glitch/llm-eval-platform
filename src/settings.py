"""Application settings loaded from environment variables or a .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pydantic settings model for judge and evaluation configuration."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
    )

    OPEN_ROUTER_URL: str = "https://openrouter.ai/api/v1"
    OPEN_ROUTER_API_KEY: str

    JUDGE_MODEL_NAME: str

    RAGAS_MAX_WORKERS: int = 3
    DEEPEVAL_MAX_CONCURRENT: int = 1
    DEEPEVAL_BATCH_SIZE: int = 100
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.0

    HTTP_MAX_RETRIES: int = 3
    HTTP_TIMEOUT: int = 1
    HTTP_RETRY_MIN_WAIT: int = 1
    HTTP_RETRY_MAX_WAIT: int = 3

    LANGFUSE_BASE_URL: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str


settings = Settings()  # type: ignore[call-arg]
