"""Tab: Import traces from Langfuse."""

from datetime import datetime, timedelta

import streamlit as st

from core.langfuse.fetcher import LangfuseFetcher, TraceFilters
from settings import settings


class ImportTab:
    def __init__(self) -> None:
        self._fetcher = LangfuseFetcher()

    def render(self) -> None:
        st.subheader("Импорт трейсов из Langfuse")
        st.caption(f"Хост: `{settings.LANGFUSE_BASE_URL}`")
        filters = self._render_filters()

        if not st.button("Загрузить трейсы", type="primary"):
            return

        with st.spinner("Fetching traces..."):
            try:
                df = self._fetcher.fetch(filters)
            except Exception as exc:
                st.error(f"Ошибка при запросе к Langfuse: {exc}")
                return

        if df.empty:
            st.info("Трейсы по заданным фильтрам не найдены.")
            return

        st.success(f"Загружено {len(df)} трейсов")
        st.dataframe(
            df.drop(columns=["messages", "expected_tools"], errors="ignore"),
            use_container_width=True,
        )

        if st.button("Использовать как датасет для оценки", type="secondary"):
            st.session_state["imported_dataset"] = df
            st.success(
                "Датасет сохранён. Перейди на вкладку **▶ Запуск** и выбери «Импорт из Langfuse»."
            )

    @staticmethod
    def _render_filters() -> TraceFilters:
        col_date, col_filters = st.columns([1, 1])

        with col_date:
            st.markdown("**Период**")
            from_date = st.date_input("От", value=datetime.now() - timedelta(days=7))
            to_date = st.date_input("До", value=datetime.now())

        with col_filters:
            st.markdown("**Фильтры**")
            name = st.text_input("Имя трейса (trace name)", placeholder="my-agent")
            tags_raw = st.text_input("Теги (через запятую)", placeholder="production, v2")
            user_id = st.text_input("User ID", placeholder="необязательно")
            limit = st.number_input("Макс. трейсов", min_value=1, max_value=500, value=50, step=10)

        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

        return TraceFilters(
            from_date=datetime.combine(from_date, datetime.min.time()),
            to_date=datetime.combine(to_date, datetime.max.time()),
            name=name or None,
            tags=tags,
            user_id=user_id or None,
            limit=int(limit),
        )
