FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HYPERION_DATA_DIR=/var/lib/hyperion \
    HYPERION_MAX_OBSERVATIONS=50000 \
    HYPERION_REQUIRE_API_KEY=true \
    HYPERION_CONFIG_FILE=/etc/hyperion/config.json \
    HYPERION_MODEL_FILE=/etc/hyperion/model.json

WORKDIR /app

RUN groupadd --system hyperion \
    && useradd --system --gid hyperion --home-dir /nonexistent --shell /usr/sbin/nologin hyperion \
    && mkdir -p /var/lib/hyperion /etc/hyperion \
    && chown -R hyperion:hyperion /var/lib/hyperion

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs/hyperion.production.json /etc/hyperion/config.json
COPY configs/hyperion.model.v1.json /etc/hyperion/model.json

RUN python -m pip install .

USER hyperion
EXPOSE 8080
VOLUME ["/var/lib/hyperion"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"

CMD ["uvicorn", "jarvisx.hyperion_service:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers"]
