"""Langfuse exporter — pushes eval results as trace scores and dataset experiments."""

import logging
from typing import Any, cast

from langfuse import Langfuse
from langfuse.experiment import Evaluation, EvaluatorFunction, ExperimentItem, LocalExperimentItem
import pandas as pd

from settings import settings

logger = logging.getLogger(__name__)

_SCORE_SUFFIX = "_score"
_REASON_SUFFIX = "_reason"


class LangfuseExporter:
    """Pushes evaluation results to Langfuse.

    Holds a single Langfuse client and exposes two operations:
    - push_scores_to_traces: attach scores to existing production traces by trace_id
    - push_experiment: register results as a dataset experiment run
    """

    def __init__(self) -> None:
        self._client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_BASE_URL,
        )

    def push_scores_to_traces(
        self,
        scores_df: pd.DataFrame,
        trace_id_col: str = "trace_id",
    ) -> int:
        """Attach eval scores to existing Langfuse traces. Returns number of scores pushed."""
        if trace_id_col not in scores_df.columns:
            logger.warning("Column %r not found — skipping trace score push", trace_id_col)
            return 0

        score_cols = self._score_columns(scores_df)
        pushed = 0

        for _, row in scores_df.iterrows():
            trace_id = row.get(trace_id_col)

            for metric in score_cols:
                value = row.get(metric)
                if pd.isna(value):
                    continue

                self._client.create_score(
                    trace_id=str(trace_id),
                    name=self._metric_name(metric),
                    value=float(value),
                    comment=self._reason(row, metric),
                )
                pushed += 1

        self._client.flush()
        logger.info("Pushed %d scores to Langfuse traces", pushed)
        return pushed

    def push_experiment(
        self,
        source_df: pd.DataFrame,
        scores_df: pd.DataFrame,
        dataset_name: str,
        run_name: str,
    ) -> None:
        """Register eval results as a Langfuse dataset experiment run."""
        score_cols = self._score_columns(scores_df)
        if not score_cols:
            logger.warning("No score columns found — nothing to push")
            return

        scores_indexed = (
            scores_df.set_index("ticket_id") if "ticket_id" in scores_df.columns else scores_df
        )
        items = self._build_items(source_df)
        evaluators = [self._make_evaluator(m, scores_indexed) for m in score_cols]

        self._client.run_experiment(
            name=dataset_name,
            run_name=run_name,
            data=items,
            task=self._identity_task,
            evaluators=evaluators,
        )
        logger.info("Experiment run %r pushed to dataset %r", run_name, dataset_name)

    @staticmethod
    def _build_items(source_df: pd.DataFrame) -> list[LocalExperimentItem]:
        return [
            {
                "input": {
                    "messages": row.get("messages", []),
                    "ticket_id": row.get("ticket_id", ""),
                },
                "expected_output": {"tools": row.get("expected_tools", [])},
                "metadata": {
                    "scenario": str(row.get("scenario", "")),
                    "ticket_id": str(row.get("ticket_id", "")),
                },
            }
            for _, row in source_df.iterrows()
        ]

    def _make_evaluator(self, metric: str, scores_indexed: pd.DataFrame) -> EvaluatorFunction:
        """
        Output is the ticket_id returned by _identity_task; input/expected_output/metadata
        are passed by Langfuse but not needed for score lookup.
        """

        def evaluator(
            *,
            output: Any,
            input: Any = None,
            expected_output: Any = None,
            metadata: dict[str, Any] | None = None,
            **_: Any,
        ) -> Evaluation | None:
            ticket_id = output
            if ticket_id not in scores_indexed.index:
                return None

            value = scores_indexed.at[ticket_id, metric]
            if pd.isna(value):
                return None

            return Evaluation(
                name=self._metric_name(metric),
                value=float(value),
                comment=self._reason(scores_indexed.loc[ticket_id], metric),
            )

        return cast(EvaluatorFunction, evaluator)

    @staticmethod
    def _identity_task(*, item: ExperimentItem, **_: Any) -> Any:
        return item["metadata"]["ticket_id"]  # type: ignore[index]

    @staticmethod
    def _score_columns(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c.endswith(_SCORE_SUFFIX)]

    @staticmethod
    def _metric_name(col: str) -> str:
        return col.removesuffix(_SCORE_SUFFIX)

    @classmethod
    def _reason(cls, row: pd.Series, metric: str) -> str | None:
        reason_col = cls._metric_name(metric) + _REASON_SUFFIX
        if reason_col not in row.index:
            return None

        value = row.get(reason_col)
        return str(value) if value and not pd.isna(value) else None
