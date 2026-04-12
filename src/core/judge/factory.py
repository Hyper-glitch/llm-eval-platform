"""Factory for creating judge instances from application settings."""

import logging

from openai import AsyncOpenAI, OpenAI
from ragas.llms import InstructorBaseRagasLLM, llm_factory

from core.judge.model import DeepEvalJudge
from settings import Settings

logger = logging.getLogger(__name__)


def create_judges(settings: Settings) -> tuple[DeepEvalJudge, InstructorBaseRagasLLM]:
    """Instantiate and return a DeepEvalJudge and a Ragas LLM from application settings."""
    logger.info("Initializing judges and evaluator...")
    sync_client = OpenAI(
        base_url=settings.OPEN_ROUTER_URL,
        api_key=settings.OPEN_ROUTER_API_KEY,
        max_retries=settings.HTTP_MAX_RETRIES,
        timeout=settings.HTTP_TIMEOUT,
    )
    async_client = AsyncOpenAI(
        base_url=settings.OPEN_ROUTER_URL,
        api_key=settings.OPEN_ROUTER_API_KEY,
        max_retries=settings.HTTP_MAX_RETRIES,
        timeout=settings.HTTP_TIMEOUT,
    )
    judge = DeepEvalJudge(
        model_name=settings.JUDGE_MODEL_NAME,
        sync_client=sync_client,
        async_client=async_client,
    )
    return judge, llm_factory(settings.JUDGE_MODEL_NAME, client=async_client)
