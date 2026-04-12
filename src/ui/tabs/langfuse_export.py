"""Langfuse export section, composed into ResultsTab."""

import pandas as pd
import streamlit as st

from core.langfuse.exporter import LangfuseExporter
from settings import settings
from ui.helpers import OUTPUT_ROOT


class LangfuseExport:
    def __init__(self) -> None:
        self._exporter = LangfuseExporter()

    def render(self, scores_df: pd.DataFrame, run_name: str) -> None:
        st.divider()
        with st.expander("📤 Экспорт в Langfuse", expanded=False):
            st.caption(f"Хост: `{settings.LANGFUSE_BASE_URL}`")

            has_trace_id = "trace_id" in scores_df.columns
            col_a, col_b = st.columns(2)
            with col_a:
                push_traces = st.checkbox(
                    "Скоры на трейсы",
                    value=has_trace_id,
                    disabled=not has_trace_id,
                    help="Требует колонку `trace_id` в датасете",
                )
            with col_b:
                push_exp = st.checkbox("Создать experiment run", value=True)

            dataset_name = st.text_input("Название датасета в Langfuse", value="llm-eval")
            exp_run_name = st.text_input("Название запуска", value=run_name)

            if not st.button("Экспортировать", type="primary"):
                return

            source_df = self._load_source_df(run_name)

            with st.spinner("Экспорт в Langfuse..."):
                if push_traces and has_trace_id:
                    count = self._exporter.push_scores_to_traces(scores_df)
                    st.success(f"Pushed {count} scores на трейсы")

                if push_exp:
                    if source_df is None:
                        st.warning("Исходный датасет не найден — experiment run пропущен.")
                    else:
                        self._exporter.push_experiment(
                            source_df, scores_df, dataset_name, exp_run_name
                        )
                        st.success(
                            f"Experiment run **{exp_run_name}** → датасет **{dataset_name}**"
                        )

    @staticmethod
    def _load_source_df(run_name: str) -> pd.DataFrame | None:
        source_path = OUTPUT_ROOT / run_name / "source.csv"
        if source_path.exists():
            return pd.read_csv(source_path)

        return st.session_state.get("last_source")
