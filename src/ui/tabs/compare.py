"""Tab: Compare two evaluation runs."""

import pandas as pd
import streamlit as st

from ui.helpers import mean_scores, recent_runs


def _load_run(side: str) -> tuple[pd.DataFrame | None, str]:
    """Render source selector for one side, return (dataframe, label)."""
    label = st.text_input(f"Название {side}", value=f"LLM {side}")
    source = st.radio(f"Источник {side}", ["История", "Файл"], key=f"src_{side}", horizontal=True)

    if source == "История":
        runs = recent_runs()
        if not runs:
            st.info("Нет сохранённых запусков.")
            return None, label
        options = {p.name: p for p in runs}
        selected = st.selectbox(f"Запуск {side}", list(options.keys()), key=f"sel_{side}")
        return pd.read_csv(options[selected] / "scores.csv"), label

    up = st.file_uploader(f"scores_{side}.csv", type=["csv"], key=f"up_{side}")
    return (pd.read_csv(up) if up else None), label


def render() -> None:
    st.subheader("Сравнение двух запусков")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Запуск A**")
        df_a, label_a = _load_run("A")
    with col_b:
        st.markdown("**Запуск B**")
        df_b, label_b = _load_run("B")

    if df_a is None or df_b is None:
        if df_a is not None or df_b is not None:
            st.info("Загрузите оба датасета для сравнения.")
        return

    st.divider()
    comparison = pd.concat(
        [mean_scores(df_a).rename(label_a), mean_scores(df_b).rename(label_b)],
        axis=1,
    ).dropna(how="all")
    comparison.index = [i.replace("_score", "") for i in comparison.index]
    comparison["Δ (B − A)"] = (comparison[label_b] - comparison[label_a]).round(3)
    comparison["Лучше"] = comparison["Δ (B − A)"].apply(
        lambda d: f"✅ {label_b}" if d > 0.01 else (f"✅ {label_a}" if d < -0.01 else "≈ Равно")
    )

    st.subheader("Сравнение метрик")
    st.dataframe(
        comparison.style.background_gradient(subset=["Δ (B − A)"], cmap="RdYlGn", vmin=-1, vmax=1),
        use_container_width=True,
    )
    st.caption(f"Строк в A: {len(df_a)} · Строк в B: {len(df_b)}")
