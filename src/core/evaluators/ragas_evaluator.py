"""Ragas evaluator for multi-turn agent dialogues."""

import ast
import asyncio
from typing import Any, TypeAlias

import pandas as pd
from ragas import MultiTurnSample
from ragas.llms import InstructorBaseRagasLLM
from ragas.metrics.collections import AgentGoalAccuracyWithoutReference, ToolCallAccuracy

from core.criteria import DEFAULT_TOPICS, REFERENCE_TOPICS
from core.message_utils import build_ragas_messages, build_reference_tool_calls

_RagasMetric: TypeAlias = ToolCallAccuracy | AgentGoalAccuracyWithoutReference


class RagasEvaluator:
    """Evaluates agent dialogues using Ragas multi-turn metrics."""

    def __init__(self, llm: InstructorBaseRagasLLM, max_concurrent: int = 1) -> None:
        self._llm = llm
        self._metrics = self._build_metrics()
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate(self, samples: list[MultiTurnSample]) -> pd.DataFrame:
        """Score a list of MultiTurnSamples and return the results as a DataFrame."""
        rows = [await self._score_sample(sample) for sample in samples]
        return pd.DataFrame(rows)

    @staticmethod
    def build_case(row: pd.Series) -> MultiTurnSample:
        """Build a Ragas MultiTurnSample from a DataFrame row."""
        raw_messages = row.get("messages_ragas_dicts") or []
        scenario = row.get("scenario", "")
        topics = REFERENCE_TOPICS.get(scenario, DEFAULT_TOPICS)

        messages = build_ragas_messages(raw_messages)

        raw_tools = row.get("expected_tools") or []
        if isinstance(raw_tools, str):
            try:
                raw_tools = ast.literal_eval(raw_tools)
            except (ValueError, SyntaxError):
                raw_tools = []
        reference_tool_calls = build_reference_tool_calls(raw_tools)

        return MultiTurnSample(
            user_input=messages,
            reference_topics=topics,
            reference_tool_calls=reference_tool_calls,
        )

    async def _score_sample(self, sample: MultiTurnSample) -> dict[str, Any]:
        """Score a single sample across all configured metrics."""
        row = {}
        for metric in self._metrics:
            row[metric.name] = await self._score_metric(metric, sample)
        return row

    async def _score_metric(self, metric: _RagasMetric, sample: MultiTurnSample) -> Any:
        """Run a single metric against a sample and normalize the result to a scalar."""
        async with self._semaphore:
            if isinstance(metric, ToolCallAccuracy):
                result = await metric.ascore(
                    user_input=sample.user_input,
                    reference_tool_calls=sample.reference_tool_calls or [],
                )
            elif hasattr(metric, "ascore"):
                result = await metric.ascore(user_input=sample.user_input)
            else:
                return None

            if hasattr(result, "value"):
                return result.value
            if isinstance(result, dict):
                return result.get("score")
            return result

    def _build_metrics(self) -> list[_RagasMetric]:
        """Instantiate Ragas metrics used for evaluation."""
        return [
            ToolCallAccuracy(name="tool_call_accuracy", llm=self._llm),
            AgentGoalAccuracyWithoutReference(name="agent_goal_accuracy", llm=self._llm),
        ]
