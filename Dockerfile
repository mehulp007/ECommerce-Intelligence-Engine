# API image. Data, reports, and models are mounted at run time (see compose.yaml).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EIE_PROJECT_ROOT=/app \
    HF_HOME=/cache/huggingface

WORKDIR /app

# CPU PyTorch keeps the image small; the API only encodes short search queries.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[data,search,api,copilot]"

RUN useradd --create-home app && mkdir -p /cache/huggingface && chown -R app /cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "ecommerce_intelligence.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
