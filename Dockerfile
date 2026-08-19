FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml setup.py README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir . && mkdir -p /data/runs

ENV PORT=10000
ENV JARVISX_RUN_STORE=/data/runs
VOLUME ["/data"]

EXPOSE 10000

CMD ["sh", "-c", "uvicorn jarvisx.api:app --host 0.0.0.0 --port ${PORT:-10000}"]
