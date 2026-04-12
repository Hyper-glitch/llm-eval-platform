"""Runs the evaluation pipeline in a background thread with its own event loop."""

import asyncio
from pathlib import Path
import threading

import pandas as pd

from config import EvalConfig
from core.evaluators.deepeval_evaluator import DeepevalEvaluator
from core.evaluators.ragas_evaluator import RagasEvaluator
from core.judge.factory import create_judges
from core.pipeline import EvaluationPipeline
from settings import settings


def run_evaluation(
    df: pd.DataFrame,
    config: EvalConfig,
    output_dir: Path,
    skip_ragas: bool,
    skip_deepeval: bool,
) -> Exception | None:
    """Execute the full pipeline in a new thread. Returns None on success, Exception on failure."""
    error: Exception | None = None

    def _thread() -> None:
        nonlocal error
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            deepeval_judge, ragas_llm = create_judges(settings)
            pipeline = EvaluationPipeline(
                deepeval=DeepevalEvaluator(
                    model=deepeval_judge,
                    config=config,
                    max_concurrent=settings.DEEPEVAL_MAX_CONCURRENT,
                    batch_size=settings.DEEPEVAL_BATCH_SIZE,
                ),
                ragas=RagasEvaluator(
                    llm=ragas_llm,
                    config=config,
                    max_concurrent=settings.RAGAS_MAX_WORKERS,
                ),
            )
            loop.run_until_complete(
                pipeline.run(
                    df=df,
                    output_dir=output_dir,
                    skip_ragas=skip_ragas,
                    skip_deepeval=skip_deepeval,
                )
            )
        except Exception as exc:
            error = exc
        finally:
            loop.close()

    t = threading.Thread(target=_thread)
    t.start()
    t.join()
    return error
