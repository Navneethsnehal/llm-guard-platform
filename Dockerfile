FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY data ./data
COPY ui ./ui

RUN pip install --no-cache-dir ".[ui]"

EXPOSE 8080 8501

CMD ["llm-guard", "serve", "--host", "0.0.0.0", "--port", "8080"]
