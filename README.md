# LLM Eval Platform

Платформа для оценки качества диалогов LLM-агентов с поддержкой метрик DeepEval и Ragas.

## Требования

- Python 3.13+
- [Poetry](https://python-poetry.org/)

## Установка

```bash
git clone <repo>
cd llm-eval-platform
poetry install
```

## Переменные окружения

Создай файл `src/.env`:

```env
# Обязательные
OPEN_ROUTER_API_KEY=sk-or-...       # API-ключ OpenRouter
JUDGE_MODEL_NAME=openai/gpt-4o      # Модель-судья (любая через OpenRouter)

# Опциональные (со значениями по умолчанию)
OPEN_ROUTER_URL=https://openrouter.ai/api/v1
RAGAS_MAX_WORKERS=3
DEEPEVAL_MAX_CONCURRENT=1
DEEPEVAL_BATCH_SIZE=100
LLM_MAX_TOKENS=1024
LLM_TEMPERATURE=0.0
HTTP_MAX_RETRIES=3
HTTP_TIMEOUT=1
HTTP_RETRY_MIN_WAIT=1
HTTP_RETRY_MAX_WAIT=3
```

## Запуск

```bash
PYTHONPATH=src poetry run streamlit run src/ui/app.py
```

Приложение откроется на [http://localhost:8501](http://localhost:8501).

## Формат датасета

CSV-файл с колонками:

| Колонка | Тип | Описание |
|---|---|---|
| `ticket_id` | str | Уникальный ID диалога |
| `scenario` | str | Сценарий (`cancel_order`, `change_delivery`, `log_topics`) |
| `messages` | JSON (str) | Список сообщений диалога |
| `expected_tools` | JSON (str) | Эталонные вызовы инструментов (для ToolCallAccuracy) |

### Формат `messages`

```json
[
  {"role": "user", "content": "отмени заказ"},
  {"role": "assistant", "content": "", "tool_calls": [{"name": "get_reasons", "args": {}}]},
  {"role": "tool", "content": "{\"status\": \"OK\"}", "name": "get_reasons"},
  {"role": "assistant", "content": "Выберите причину отмены..."}
]
```

### Формат `expected_tools`

```json
[
  {"name": "get_reasons", "args": {}},
  {"name": "confirm_cancellation", "args": {}}
]
```

Пример датасета: [`dataset/dataset-v1.csv`](dataset/dataset-v1.csv).

## Интерфейс

### Вкладка «Запуск»

1. Загрузи CSV-файл
2. Укажи имя запуска (используется как имя папки с результатами)
3. При необходимости отредактируй контекст агента и критерии оценки
4. Нажми **Запустить оценку**

Результаты сохраняются в `agent_eval_outputs/<имя_запуска>/scores.csv`.

### Вкладка «Результаты»

Просмотр результатов: из последнего запуска, из истории или из загруженного файла.  
Показывает сводную таблицу mean / std / min / max по каждой метрике и разбивку по сценариям.

### Вкладка «Сравнение»

Сравнение двух запусков (или двух CSV) по всем метрикам с дельтой B − A.

## Метрики

| Метрика | Фреймворк | Описание |
|---|---|---|
| `role_adherence` | DeepEval | Соблюдение роли агента |
| `conversation_completeness` | DeepEval | Полнота решения задачи пользователя |
| `hallucination` | DeepEval GEval | Галлюцинации относительно ответов инструментов |
| `tool_truthfulness` | DeepEval GEval | Агент не обещает то, чего нет в инструментах |
| `tool_call_accuracy` | Ragas | Точность вызовов инструментов (требует `expected_tools`) |
| `agent_goal_accuracy` | Ragas | Достигнута ли цель пользователя |