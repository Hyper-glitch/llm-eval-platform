"""Helper utilities for the judge layer: message building and response parsing."""

import json
import logging
from typing import Any

from langchain_core.utils.json import parse_json_markdown
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from settings import settings

_Schema = type[BaseModel] | None

logger = logging.getLogger(__name__)


def build_messages(prompt: str, schema: _Schema) -> list[ChatCompletionMessageParam]:
    """Build the messages list for a chat completion request.

    Prepends a system message with the JSON schema when schema is provided.
    """
    if schema is None:
        return [{"role": "user", "content": prompt}]

    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    return [
        {"role": "system", "content": f"Return ONLY valid JSON matching schema:\n{schema_json}"},
        {"role": "user", "content": prompt},
    ]


def max_tokens_for_attempt(schema: _Schema, attempt: int) -> int:
    """Return the max_tokens cap for a retry attempt, doubling each time up to 8192."""
    base = 4096 if schema is not None else settings.LLM_MAX_TOKENS
    return int(min(base * (2**attempt), 8192))


def parse_response(
    content: str | None, finish_reason: str | None, schema: _Schema
) -> tuple[Any | None, str | None]:
    """Parse and validate a raw completion response.

    Returns (result, None) on success or (None, error_message) on failure.
    """
    if not content or finish_reason == "length":
        return None, f"Empty or truncated response: finish_reason={finish_reason}"
    if schema is None:
        return content, None

    try:
        return schema.model_validate(parse_json_markdown(content)), None
    except Exception as error:
        return None, f"No valid JSON for {schema.__name__}: {error}"


def fallback_result(schema: _Schema, error: str | None) -> Any:
    """Return a safe fallback value when all retries fail.

    Returns an empty model instance if a schema is provided, otherwise an empty string.
    """
    logger.warning("DeepEval judge fallback: %s", error)
    return schema.model_validate({}) if schema is not None else ""
