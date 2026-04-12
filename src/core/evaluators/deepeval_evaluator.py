"""DeepEval evaluator for conversational agent dialogues."""

import logging
from typing import Any

from deepeval.evaluate import AsyncConfig, CacheConfig, ErrorConfig, evaluate
from deepeval.metrics import (
    BaseConversationalMetric,
    ConversationalGEval,
    ConversationCompletenessMetric,
    RoleAdherenceMetric,
)
from deepeval.test_case import ConversationalTestCase, TurnParams
import pandas as pd

from config import EvalConfig
from core.judge.model import DeepEvalJudge
from core.message_utils import build_deepeval_turns

logger = logging.getLogger(__name__)

DEFAULT_EVALUATION_PARAMS = [TurnParams.CONTENT]
GEVAL_PARAMS = {
    "hallucination": [TurnParams.CONTENT, TurnParams.EXPECTED_OUTCOME, TurnParams.TOOLS_CALLED],
    "tool_truthfulness": [TurnParams.CONTENT, TurnParams.EXPECTED_OUTCOME, TurnParams.TOOLS_CALLED],
}


class DeepevalEvaluator:
    """Evaluates agent dialogues using DeepEval conversational metrics."""

    def __init__(
        self,
        model: DeepEvalJudge,
        config: EvalConfig,
        max_concurrent: int = 10,
        batch_size: int = 100,
    ) -> None:
        self._model = model
        self._config = config
        self._max_concurrent = max_concurrent
        self._batch_size = batch_size
        self._metrics = self._build_metrics()

    def evaluate(self, cases: list[ConversationalTestCase]) -> pd.DataFrame:
        """Run all metrics on the given test cases in batches and return per-metric scores."""
        evaluates = []
        n_batches = (len(cases) + self._batch_size - 1) // self._batch_size

        for batch_idx in range(n_batches):
            batch = cases[batch_idx * self._batch_size : (batch_idx + 1) * self._batch_size]
            logger.info("DeepEval batch %d/%d (%d cases)", batch_idx + 1, n_batches, len(batch))
            batch_results = evaluate(
                test_cases=batch,
                metrics=self._metrics,
                async_config=AsyncConfig(run_async=False, max_concurrent=self._max_concurrent),
                error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=True),
                cache_config=CacheConfig(write_cache=True, use_cache=True),
            )
            evaluates.extend(batch_results.test_results)

        logger.info("DeepEval returned %d test_results (submitted %d)", len(evaluates), len(cases))
        return pd.DataFrame(_metrics_rows(evaluates))

    def build_case(self, row: pd.Series) -> ConversationalTestCase:
        """Build a DeepEval ConversationalTestCase from a DataFrame row."""
        turns = build_deepeval_turns(row["messages"])
        if not turns:
            ticket_id = row["ticket_id"]
            raise ValueError(f"No dialogue turns parsed for ticket_id={ticket_id}")

        return ConversationalTestCase(
            name=row["ticket_id"],
            turns=turns,
            chatbot_role=self._config.chatbot_role,
            scenario=self._config.scenario,
            user_description=self._config.user_description,
            expected_outcome=self._config.expected_outcome,
        )

    def _build_metrics(self) -> list[BaseConversationalMetric]:
        """Build the metric list from config flags and criteria."""
        metrics: list[BaseConversationalMetric] = []

        if self._config.run_role_adherence:
            metrics.append(RoleAdherenceMetric(model=self._model))
        if self._config.run_conversation_completeness:
            metrics.append(ConversationCompletenessMetric(model=self._model))

        for name, criteria in self._config.geval_criteria.items():
            metrics.append(
                ConversationalGEval(
                    name=name,
                    model=self._model,
                    criteria=criteria,
                    evaluation_params=GEVAL_PARAMS.get(name, DEFAULT_EVALUATION_PARAMS),
                )
            )

        for criterion in self._config.tone_criteria:
            metrics.append(
                ConversationalGEval(
                    name=criterion["id"],
                    model=self._model,
                    criteria=criterion["description"],
                )
            )

        return metrics


def _metrics_rows(test_results: Any) -> list[dict[str, Any]]:
    """Convert DeepEval test results to a list of row dicts for DataFrame construction."""
    rows = []
    for test_result in test_results:
        row: dict[str, Any] = {"ticket_id": test_result.name}
        for metric_data in test_result.metrics_data:
            metric_name = metric_data.name.lower().replace(" ", "_")
            row[f"{metric_name}_score"] = (
                round(metric_data.score, 3) if metric_data.score is not None else None
            )
            row[f"{metric_name}_reason"] = metric_data.reason
        rows.append(row)
    return rows
