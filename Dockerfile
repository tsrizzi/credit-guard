FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv \
    && useradd --create-home --uid 10001 appuser

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY params.yaml ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PREDICTION_LOG=/app/logs/predictions.jsonl \
    PYTHONUNBUFFERED=1

RUN mkdir -p /app/logs /app/models && chown -R appuser /app/logs
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "credit_guard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
