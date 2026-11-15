# Image CPU de l'API (modèles téléchargés au premier appel, puis mis en cache dans le volume /models-cache)
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/models-cache PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY pyproject.toml ./
COPY src ./src
RUN pip install .

COPY configs ./configs
COPY data/kb ./data/kb
COPY models ./models

RUN useradd --create-home app && mkdir -p /models-cache && chown app /models-cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "support_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
