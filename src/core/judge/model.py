"""DeepEval LLM adapter backed by an OpenAI-compatible endpoint."""

import logging
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import APIError, AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.judge.utils import build_messages, fallback_result, max_tokens_for_attempt, parse_response
from settings import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
STEPS_FALLBACK = ("Check correctness", "Check hallucination", "Check tool usage")

_RETRY_ON = (APIError,)


class _ParseError(Exception):
    """Raised when response parsing fails; triggers a retry."""


class DeepEvalJudge(DeepEvalBaseLLM):
    """DeepEval LLM adapter backed by an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_name: str,
        sync_client: OpenAI,
        async_client: AsyncOpenAI,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._sync_client = sync_client
        self._async_client = async_client
        self._extra_body = extra_body
        super().__init__(model_name)

    def load_model(self, *args: Any, **kwargs: Any) -> "DeepEvalJudge":
        """Return self as the loaded model (required by the DeepEval base class)."""
        return self

    def get_model_name(self) -> str:
        """Return the model name string (required by the DeepEval base class)."""
        return self._model_name

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any) -> Any:
        """Generate a synchronous response, optionally validating against a Pydantic schema."""
        if schema is not None and schema.__name__ == "Steps":
            return schema(steps=list(STEPS_FALLBACK))
        try:
            return self._generate_sync(build_messages(prompt, schema), schema)
        except RetryError as exc:
            return fallback_result(schema, str(exc))

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any) -> Any:
        """Generate an asynchronous response, optionally validating against a Pydantic schema."""
        if schema is not None and schema.__name__ == "Steps":
            return schema(steps=list(STEPS_FALLBACK))
        try:
            return await self._generate_async(build_messages(prompt, schema), schema)
        except RetryError as exc:
            return fallback_result(schema, str(exc))

    def _generate_sync(
        self, messages: list[ChatCompletionMessageParam], schema: type[BaseModel] | None
    ) -> Any:
        for attempt in Retrying(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((*_RETRY_ON, _ParseError)),
        ):
            with attempt:
                attempt_num = attempt.retry_state.attempt_number - 1
                response = self._sync_client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    max_tokens=max_tokens_for_attempt(schema, attempt_num),
                    temperature=settings.LLM_TEMPERATURE,
                    extra_body=self._extra_body,
                )
                result, error = parse_response(
                    response.choices[0].message.content,
                    response.choices[0].finish_reason,
                    schema,
                )
                if result is None:
                    raise _ParseError(error)
                return result

    async def _generate_async(
        self, messages: list[ChatCompletionMessageParam], schema: type[BaseModel] | None
    ) -> Any:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((*_RETRY_ON, _ParseError)),
        ):
            with attempt:
                attempt_num = attempt.retry_state.attempt_number - 1
                response = await self._async_client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    max_tokens=max_tokens_for_attempt(schema, attempt_num),
                    temperature=settings.LLM_TEMPERATURE,
                    extra_body=self._extra_body,
                )
                result, error = parse_response(
                    response.choices[0].message.content,
                    response.choices[0].finish_reason,
                    schema,
                )
                if result is None:
                    raise _ParseError(error)
                return result
