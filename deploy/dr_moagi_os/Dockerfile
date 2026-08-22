FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=10000 \
    JARVISX_STATE_DIR=/var/lib/jarvisx

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 jarvisx \
    && mkdir -p /var/lib/jarvisx \
    && chown -R jarvisx:jarvisx /var/lib/jarvisx /app

USER jarvisx

VOLUME ["/var/lib/jarvisx"]
EXPOSE 10000

HEALTHCHECK --interval=20s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT','10000') + '/healthz', timeout=2).read()"

CMD ["sh", "-c", "uvicorn jarvisx.dr_moagi_os_api:app --host 0.0.0.0 --port ${PORT:-10000}"]
