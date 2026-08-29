FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JARVISX_IDE_STATIC_DIR=/app/apps/dr-moagi-ide/static \
    JARVISX_IDE_DB=/app/state/dr-moagi-ide/ide.sqlite3 \
    JARVISX_STATE_DIR=/app/state/dr-moagi-os

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY apps/dr-moagi-ide ./apps/dr-moagi-ide
RUN pip install --no-cache-dir . && \
    mkdir -p /app/state/dr-moagi-ide /app/state/dr-moagi-os

EXPOSE 8000
VOLUME ["/app/state"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).read()" || exit 1

CMD ["python", "apps/dr-moagi-ide/server.py"]
