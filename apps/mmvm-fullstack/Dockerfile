FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MMVM_DB=/data/mmvm.sqlite3 \
    MMVM_FRONTEND=/app/apps/mmvm-fullstack/index.html \
    MMVM_WORKER_INTERVAL=0.05

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin mmvm \
    && mkdir -p /app /data \
    && chown -R mmvm:mmvm /app /data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY --chown=mmvm:mmvm apps/mmvm-fullstack ./apps/mmvm-fullstack

USER mmvm
EXPOSE 8080
VOLUME ["/data"]

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=4 \
  CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["uvicorn", "jarvisx.mmvm_api:app", "--host", "0.0.0.0", "--port", "8080", "--no-server-header"]
