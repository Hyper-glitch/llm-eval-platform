"""Main entry point for agent evaluation."""

import argparse
import asyncio
import logging
from pathlib import Path

from core.data_loader import load_eval_df
from core.evaluators.deepeval_evaluator import DeepevalEvaluator
from core.evaluators.ragas_evaluator import RagasEvaluator
from core.judge.factory import create_judges
from core.pipeline import EvaluationPipeline
from settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main_async(args: argparse.Namespace) -> None:
    """Main async function for evaluation."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    deepeval_judge, ragas_llm = create_judges(settings)
    pipeline = EvaluationPipeline(
        deepeval=DeepevalEvaluator(
            model=deepeval_judge,
            granular_tone_path=args.tone,
            max_concurrent=settings.DEEPEVAL_MAX_CONCURRENT,
            batch_size=settings.DEEPEVAL_BATCH_SIZE,
        ),
        ragas=RagasEvaluator(llm=ragas_llm, max_concurrent=settings.RAGAS_MAX_WORKERS),
    )
    df = load_eval_df(args.csv, nrows=args.nrows)

    try:
        await pipeline.run(
            df=df,
            output_dir=args.output_dir,
            skip_ragas=args.skip_ragas,
            skip_deepeval=args.skip_deepeval,
        )
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}", exc_info=True)
        raise exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics for agent evaluation")
    parser.add_argument("--csv", required=True, type=Path, help="Path to use-cases file in CSV or Parquet format")
    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to process (None = all)")
    parser.add_argument("--tone", type=Path, default=None, help="JSON file with tone of voice criteria")
    parser.add_argument("--output_dir", type=Path, default=Path("../agent_eval_outputs"), help="Output directory")
    parser.add_argument("--skip_ragas", action="store_true", help="Skip Ragas evaluation")
    parser.add_argument("--skip_deepeval", action="store_true", help="Skip DeepEval evaluation")

    asyncio.run(main_async(parser.parse_args()))
