"""Tab: Results Viewer."""

import pandas as pd
import streamlit as st

from ui.helpers import recent_runs, render_scores_table, score_columns


def render() -> None:
    st.subheader("Просмотр результатов")

    source = st.radio(
        "Источник",
        ["Последний запуск", "Выбрать из истории", "Загрузить файл"],
        horizontal=True,
    )

    df: pd.DataFrame | None = None

    if source == "Последний запуск":
        df = st.session_state.get("last_result")
        if df is None:
            st.info("Запустите оценку на вкладке «Запуск».")

    elif source == "Выбрать из истории":
        runs = recent_runs()
        if not runs:
            st.info("Нет сохранённых запусков.")
        else:
            options = {p.name: p for p in runs}
            selected = st.selectbox("Запуск", list(options.keys()))
            if selected:
                df = pd.read_csv(options[selected] / "scores.csv")

    else:
        up = st.file_uploader("scores.csv", type=["csv"], key="results_upload")
        if up:
            df = pd.read_csv(up)

    if df is None:
        return

    st.subheader("Сводка метрик")
    render_scores_table(df)

    cols = score_columns(df)
    if cols and "scenario" in df.columns:
        st.subheader("По сценариям")
        pivot = df.groupby("scenario")[cols].mean().round(3)
        pivot.columns = [c.replace("_score", "") for c in pivot.columns]
        st.dataframe(pivot, use_container_width=True)

    st.subheader("Все строки")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Скачать CSV",
        data=df.to_csv(index=False).encode(),
        file_name="scores.csv",
        mime="text/csv",
    )
