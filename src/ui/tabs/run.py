"""Tab: Run Evaluation."""

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from config import EvalConfig
from core.criteria import GEVAL_CRITERIA
from core.data_loader import load_eval_df
from core.prompts import CUSTOMER_CHATBOT_ROLE, EXPECTED_OUTCOME, SCENARIO, USER_DESCRIPTION
from ui.helpers import OUTPUT_ROOT, render_scores_table
from ui.runner import run_evaluation


def render() -> None:
    col_data, col_context, col_metrics = st.columns([1, 1.4, 1.4])

    with col_data:
        st.subheader("Датасет")
        uploaded = st.file_uploader("CSV файл", type=["csv"], key="run_upload")
        nrows = st.number_input("Макс. строк (0 = все)", min_value=0, value=0, step=10)
        run_name = st.text_input("Имя запуска", value="run_1")

        st.subheader("Запуск")
        skip_deepeval = st.checkbox("Пропустить DeepEval")
        skip_ragas = st.checkbox("Пропустить Ragas")

    with col_context:
        st.subheader("Контекст агента")
        chatbot_role = st.text_area("Роль агента", value=CUSTOMER_CHATBOT_ROLE, height=130)
        scenario = st.text_area("Сценарий", value=SCENARIO, height=80)
        user_description = st.text_area("Описание пользователя", value=USER_DESCRIPTION, height=80)
        expected_outcome = st.text_area("Ожидаемый результат", value=EXPECTED_OUTCOME, height=80)

    with col_metrics:
        st.subheader("Метрики")

        st.markdown("**DeepEval — встроенные**")
        run_role = st.checkbox("Role Adherence", value=True)
        run_completeness = st.checkbox("Conversation Completeness", value=True)

        st.markdown("**DeepEval — GEval критерии**")
        geval_criteria: dict[str, str] = {}
        for metric_name, default_text in GEVAL_CRITERIA.items():
            enabled = st.checkbox(metric_name, value=True, key=f"geval_check_{metric_name}")
            if enabled:
                text = st.text_area(
                    f"Критерий: {metric_name}",
                    value=default_text,
                    height=70,
                    key=f"geval_text_{metric_name}",
                )
                geval_criteria[metric_name] = text

        st.markdown("**Ragas**")
        run_tool_acc = st.checkbox("Tool Call Accuracy", value=True)
        run_goal_acc = st.checkbox("Agent Goal Accuracy", value=True)

    st.divider()
    if st.button("🚀 Запустить оценку", type="primary", use_container_width=True):
        if not uploaded:
            st.error("Загрузите CSV файл.")
            st.stop()

        config = EvalConfig(
            chatbot_role=chatbot_role,
            scenario=scenario,
            user_description=user_description,
            expected_outcome=expected_outcome,
            geval_criteria=geval_criteria,
            run_role_adherence=run_role,
            run_conversation_completeness=run_completeness,
            run_tool_call_accuracy=run_tool_acc,
            run_agent_goal_accuracy=run_goal_acc,
        )

        output_dir = OUTPUT_ROOT / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        nrows_val: int | None = int(nrows) if nrows > 0 else None

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        try:
            df = load_eval_df(tmp_path, nrows=nrows_val)
        except Exception as exc:
            st.error(f"Ошибка загрузки датасета: {exc}")
            st.stop()
        finally:
            tmp_path.unlink(missing_ok=True)

        with st.spinner(f"Оценка {len(df)} сэмплов…"):
            err = run_evaluation(df, config, output_dir, skip_ragas, skip_deepeval)

        if err:
            st.error(f"Ошибка: {err}")
        else:
            st.success(f"Готово! Результаты в `{output_dir}`")
            result_df = pd.read_csv(output_dir / "scores.csv")
            st.session_state["last_result"] = result_df
            st.session_state["last_run_name"] = run_name

            st.subheader("Сводка метрик")
            render_scores_table(result_df)

            st.subheader("Полные результаты")
            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                "Скачать scores.csv",
                data=result_df.to_csv(index=False).encode(),
                file_name=f"{run_name}_scores.csv",
                mime="text/csv",
            )
