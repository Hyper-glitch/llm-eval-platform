"""Shared constants and UI utilities for the eval platform."""

from pathlib import Path

import pandas as pd
import streamlit as st

OUTPUT_ROOT = Path(__file__).parent.parent.parent / "agent_eval_outputs"
_SCORE_SUFFIX = "_score"


def score_columns(df: pd.DataFrame) -> list[str]:
    """Return columns whose names end with '_score'."""
    return [c for c in df.columns if c.endswith(_SCORE_SUFFIX)]


def mean_scores(df: pd.DataFrame) -> "pd.Series[float]":
    """Return mean of every score column, rounded to 3 decimals."""
    return df[score_columns(df)].mean().round(3)


def recent_runs() -> list[Path]:
    """Return output run directories sorted by modification time (newest first)."""
    if not OUTPUT_ROOT.exists():
        return []
    return sorted(
        [p for p in OUTPUT_ROOT.iterdir() if (p / "scores.csv").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def render_scores_table(df: pd.DataFrame) -> None:
    """Render a summary table of mean/std/min/max for every score column."""
    cols = score_columns(df)
    if not cols:
        st.warning("В датасете нет колонок со скорами.")
        return
    summary = df[cols].agg(["mean", "std", "min", "max"]).T.round(3)
    summary.index = [c.replace(_SCORE_SUFFIX, "") for c in summary.index]
    st.dataframe(summary, use_container_width=True)
