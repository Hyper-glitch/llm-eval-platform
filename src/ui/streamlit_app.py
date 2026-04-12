import json
from pathlib import Path
import subprocess
from typing import Any

import streamlit as st

st.set_page_config(page_title="Agent Eval UI", layout="wide")
st.title("🧪 Agent Evaluation Control Panel")

st.sidebar.header("📂 Dataset")

csv_path = st.sidebar.text_input("CSV / Parquet path")
output_dir = st.sidebar.text_input("Output dir", "agent_eval_outputs")

st.sidebar.header("📊 DeepEval Metrics")
deepeval_metrics = st.sidebar.multiselect(
    "Select DeepEval metrics",
    [
        "role_adherence",
        "turn_relevancy",
        "knowledge_retention",
        "conversation_completeness",
        "geval",
    ],
    default=["role_adherence", "turn_relevancy"],
)

tone_eval = st.sidebar.checkbox("Enable tone-of-voice eval", value=False)

st.sidebar.header("📊 Ragas")
use_ragas = st.sidebar.checkbox("Enable Ragas evaluation", value=True)


st.sidebar.header("⚙️ Runtime params")

batch_size = st.sidebar.slider("Batch size", 10, 500, 100)
max_concurrent = st.sidebar.slider("Max concurrency", 1, 50, 5)
nrows = st.sidebar.number_input("N rows", value=0, step=10)


st.header("🧠 Evaluation context")
chatbot_role = st.text_area(
    "Chatbot role",
    value="Customer support assistant for delivery service",
)

scenario = st.text_area(
    "Scenario",
    value="User wants to cancel an order",
)

user_description = st.text_area(
    "User description",
    value="User is frustrated and wants fast cancellation",
)

expected_outcome = st.text_area(
    "Expected outcome",
    value="Assistant correctly guides cancellation flow and uses tools properly",
)

run = st.button("🚀 Run Evaluation")


def build_config() -> dict[str, Any]:
    return {
        "deepeval_metrics": deepeval_metrics,
        "ragas": use_ragas,
        "tone_eval": tone_eval,
        "batch_size": batch_size,
        "max_concurrent": max_concurrent,
        "nrows": nrows if nrows > 0 else None,
        "chatbot_role": chatbot_role,
        "scenario": scenario,
        "user_description": user_description,
        "expected_outcome": expected_outcome,
    }


if run:
    if not csv_path:
        st.error("Please provide dataset path")
        st.stop()

    config = build_config()

    st.subheader("📦 Active Config")
    st.json(config)

    config_path = Path(output_dir) / "eval_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))

    st.info("Running evaluation...")

    cmd = [
        "python",
        "main.py",
        "--csv",
        csv_path,
        "--output_dir",
        output_dir,
    ]

    if config["nrows"]:
        cmd += ["--nrows", str(config["nrows"])]

    if not use_ragas:
        cmd += ["--skip_ragas"]

    if not deepeval_metrics:
        cmd += ["--skip_deepeval"]

    if tone_eval:
        cmd += ["--tone", "tone_criteria.json"]

    subprocess.run(cmd)
    st.success("Done!")


st.divider()
st.header("📊 Results Viewer")

if Path(output_dir).exists():
    files = list(Path(output_dir).glob("*"))

    st.write("Output files:")
    st.write([f.name for f in files])

    parquet_files = list(Path(output_dir).glob("*.parquet"))
    if parquet_files:
        df = st.dataframe(Path(parquet_files[0]).read_bytes())
