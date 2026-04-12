"""Streamlit entry point."""

import streamlit as st

from ui.tabs.compare import CompareTab
from ui.tabs.results import ResultsTab
from ui.tabs.run import RunTab


class EvalApp:
    def __init__(self) -> None:
        self._run = RunTab()
        self._results = ResultsTab()
        self._compare = CompareTab()

    def run(self) -> None:
        st.set_page_config(page_title="Agent Eval", layout="wide")
        st.title("Agent Evaluation Platform")

        tab_run, tab_results, tab_compare = st.tabs(["▶ Запуск", "📊 Результаты", "🔍 Сравнение"])

        with tab_run:
            self._run.render()
        with tab_results:
            self._results.render()
        with tab_compare:
            self._compare.render()
