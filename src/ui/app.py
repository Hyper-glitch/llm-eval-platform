"""Streamlit entry point."""

import streamlit as st

from ui.tabs.compare import CompareTab
from ui.tabs.import_tab import ImportTab
from ui.tabs.results import ResultsTab
from ui.tabs.run import RunTab


class EvalApp:
    def __init__(self) -> None:
        self._run = RunTab()
        self._results = ResultsTab()
        self._compare = CompareTab()
        self._import = ImportTab()

    def run(self) -> None:
        st.set_page_config(page_title="Agent Eval", layout="wide")
        st.title("Agent Evaluation Platform")

        tab_run, tab_results, tab_compare, tab_import = st.tabs(
            ["▶ Запуск", "📊 Результаты", "🔍 Сравнение", "📥 Импорт"]
        )

        with tab_run:
            self._run.render()
        with tab_results:
            self._results.render()
        with tab_compare:
            self._compare.render()
        with tab_import:
            self._import.render()
