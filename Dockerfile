FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

COPY src/ ./src/

EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "src/ui/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
