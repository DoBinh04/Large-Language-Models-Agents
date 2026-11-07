# syntax=docker/dockerfile:1
# Hai target từ cùng một Dockerfile:
#   docker build --target api -t techviet-api .                                  # BM25 only, image nhỏ
#   docker build --target api --build-arg WITH_RAG_MODELS=1 -t techviet-api .   # + dense + reranker (torch CPU)
#   docker build --target ui  -t techviet-ui .
# Hoặc dùng docker compose up --build (xem docker-compose.yml).
# Router Small LLM (sft/dpo) cần GPU + transformers/peft: không đóng gói ở đây, dùng API_ROUTER=api hoặc rule.

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN useradd --create-home --uid 1000 app


# ---------------------------------------------------------------------------- API
FROM base AS api
ARG WITH_RAG_MODELS=0
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt \
 && if [ "$WITH_RAG_MODELS" = "1" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu torch \
     && pip install "sentence-transformers>=3.0"; \
    fi

COPY config.py ./
COPY support ./support
COPY api ./api
COPY data ./data
RUN mkdir -p results/traces /home/app/.cache && chown -R app:app /app /home/app/.cache

ENV LOG_FORMAT=json \
    LOG_LEVEL=INFO \
    HF_HOME=/home/app/.cache/huggingface \
    HOST=0.0.0.0 \
    PORT=8000
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1
CMD ["python", "-m", "api.main"]


# ---------------------------------------------------------------------------- UI
FROM base AS ui
COPY requirements-ui.txt .
RUN pip install -r requirements-ui.txt
COPY frontend ./frontend
ENV API_URL=http://api:8000
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)" || exit 1
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
