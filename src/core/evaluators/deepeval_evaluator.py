"""DeepEval evaluator for conversational agent dialogues."""

import json
import logging
from pathlib import Path
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

from core.criteria import GEVAL_CRITERIA
from core.judge.model import DeepEvalJudge
from core.message_utils import build_deepeval_turns
from core.prompts import CUSTOMER_CHATBOT_ROLE, EXPECTED_OUTCOME, SCENARIO, USER_DESCRIPTION

logger = logging.getLogger(__name__)

ALLOWED_TONE_IDS = frozenset({"short_simple_phrases", "no_excessive_emotion"})
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
        granular_tone_path: Path | None,
        max_concurrent: int = 10,
        batch_size: int = 100,
    ) -> None:
        self._model = model
        self._max_concurrent = max_concurrent
        self._batch_size = batch_size
        self._metrics = self._build_metrics(self._load_tone_criteria(granular_tone_path))

    def evaluate(self, cases: list[ConversationalTestCase]) -> pd.DataFrame:
        """
        Run all metrics on the given test cases in batches and
        return per-metric scores as a DataFrame.
        """
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

    @staticmethod
    def build_case(row: pd.Series) -> ConversationalTestCase:
        """Build a DeepEval ConversationalTestCase from a DataFrame row."""
        turns = build_deepeval_turns(row.get("messages_ragas_dicts") or [])
        if not turns:
            ticket_id = row.get("ticket_id", "<unknown>")
            raise ValueError(f"No dialogue turns parsed for ticket_id={ticket_id}")

        return ConversationalTestCase(
            name=row["ticket_id"],
            turns=turns,
            chatbot_role=CUSTOMER_CHATBOT_ROLE,
            scenario=SCENARIO,
            user_description=USER_DESCRIPTION,
            expected_outcome=EXPECTED_OUTCOME,
        )

    @staticmethod
    def _load_tone_criteria(path: Path | None) -> list[dict[str, Any]]:
        """Load tone-of-voice criteria from a JSON file, filtering to allowed IDs."""
        if path is None:
            return []
        with open(path, "r", encoding="utf-8") as file_obj:
            criteria = json.load(file_obj)
        return [item for item in criteria if item.get("id") in ALLOWED_TONE_IDS]

    def _build_metrics(self, tone_criteria: list[dict[str, Any]]) -> list[BaseConversationalMetric]:
        """Construct the full list of DeepEval metrics, including any tone-of-voice criteria."""
        metrics: list[BaseConversationalMetric] = [
            RoleAdherenceMetric(model=self._model),
            ConversationCompletenessMetric(model=self._model),
        ]
        metrics.extend(_build_geval_metrics(self._model))
        metrics.extend(
            ConversationalGEval(
                name=criterion["id"],
                model=self._model,
                criteria=criterion["description"],
            )
            for criterion in tone_criteria
        )
        return metrics


def _build_geval_metrics(model: DeepEvalJudge) -> list[ConversationalGEval]:
    """Build ConversationalGEval metrics from GEVAL_CRITERIA definitions."""
    return [
        ConversationalGEval(
            name=name,
            model=model,
            criteria=criteria,
            evaluation_params=GEVAL_PARAMS.get(name, DEFAULT_EVALUATION_PARAMS),
        )
        for name, criteria in GEVAL_CRITERIA.items()
    ]


def _metrics_rows(test_results: Any) -> list[dict[str, Any]]:
    """Convert DeepEval test results to a list of row dicts for DataFrame construction."""
    rows = []
    for test_result in test_results:
        row = {"ticket_id": test_result.name}
        for metric_data in test_result.metrics_data:
            metric_name = metric_data.name.lower().replace(" ", "_")
            row[f"{metric_name}_score"] = (
                round(metric_data.score, 3) if metric_data.score is not None else None
            )
            row[f"{metric_name}_reason"] = metric_data.reason
        rows.append(row)

    return rows
