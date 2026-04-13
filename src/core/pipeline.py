"""Agent evaluation orchestrator."""

import asyncio
import logging
from pathlib import Path

import pandas as pd
from ragas import MultiTurnSample
from tqdm.asyncio import tqdm

from core.evaluators.base import Evaluator

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Orchestrates evaluation using DeepEval and Ragas."""

    def __init__(self, deepeval: Evaluator, ragas: Evaluator) -> None:
        self._deepeval = deepeval
        self._ragas = ragas

    async def run(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        skip_ragas: bool = False,
        skip_deepeval: bool = False,
    ) -> None:
        """Run complete evaluation pipeline."""
        logger.info("Starting evaluation pipeline...")
        result_df = await self._evaluate(df, skip_ragas=skip_ragas, skip_deepeval=skip_deepeval)
        logger.info("Evaluation completed successfully!")

        result_df.to_csv(output_dir / "scores.csv", index=False)
        logger.info(f"Results saved to {output_dir}")

    async def _evaluate(
        self, df: pd.DataFrame, skip_ragas: bool = False, skip_deepeval: bool = False
    ) -> pd.DataFrame:
        """Evaluate using DeepEval and optionally Ragas."""
        if skip_deepeval:
            logger.info("Skipping DeepEval (--skip_deepeval flag set)")
            deepeval_df = pd.DataFrame()
        else:
            logger.info("Running DeepEval...")
            deepeval_cases = [self._deepeval.build_case(row) for _, row in df.iterrows()]
            loop = asyncio.get_event_loop()
            deepeval_df = await loop.run_in_executor(None, self._deepeval.evaluate, deepeval_cases)

        if skip_ragas:
            logger.info("Skipping Ragas (--skip_ragas flag set)")
            ragas_df = pd.DataFrame()
        else:
            logger.info("Running Ragas...")
            ragas_samples = [self._ragas.build_case(row) for _, row in df.iterrows()]
            ragas_df = await self._evaluate_ragas_with_progress(ragas_samples)

        return self._combine_results(df, deepeval_df, ragas_df)

    async def _evaluate_ragas_with_progress(self, samples: list[MultiTurnSample]) -> pd.DataFrame:
        """Evaluate Ragas samples with progress bar."""
        if not samples:
            return pd.DataFrame()

        results = []
        with tqdm(total=len(samples), desc="Ragas evaluation") as pbar:
            batch_size = max(1, len(samples) // 10)
            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]
                try:
                    batch_results = await self._ragas.evaluate(batch)
                    results.append(batch_results)
                except Exception as exc:
                    logger.error("Ragas batch %d failed, skipping: %s", i, exc)
                finally:
                    pbar.update(len(batch))

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    @staticmethod
    def _combine_results(
        source_df: pd.DataFrame, deepeval_df: pd.DataFrame, ragas_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine source data with evaluation results."""
        result = pd.concat(
            [
                source_df.reset_index(drop=True),
                deepeval_df.reset_index(drop=True),
                ragas_df.reset_index(drop=True),
            ],
            axis=1,
        )
        return result.loc[:, ~result.columns.duplicated()]
