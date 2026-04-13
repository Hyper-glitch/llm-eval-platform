"""Shared evaluator protocol for type-safe pipeline wiring."""

from typing import Any, Protocol

import pandas as pd


class Evaluator(Protocol):
    """Structural interface implemented by DeepevalEvaluator and RagasEvaluator."""

    def evaluate(self, cases: list[Any]) -> Any:
        """Run all metrics on the given cases and return results."""
        ...

    def build_case(self, row: pd.Series) -> Any:
        """Build an evaluator-specific test case from a DataFrame row."""
        ...
