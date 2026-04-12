"""Entry point for the Agent Evaluation Platform."""

import streamlit as st

from ui.tabs import compare, results, run

st.set_page_config(page_title="Agent Eval", layout="wide")
st.title("Agent Evaluation Platform")

tab_run, tab_results, tab_compare = st.tabs(["▶ Запуск", "📊 Результаты", "🔍 Сравнение"])

with tab_run:
    run.render()

with tab_results:
    results.render()

with tab_compare:
    compare.render()
