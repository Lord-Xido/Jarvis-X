FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DR_MOAGI_CLOUD_DATA_DIR=/var/lib/dr-moagi-cloud \
    DR_MOAGI_CLOUD_REQUIRE_API_KEY=true

WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 jarvisx \
    && mkdir -p /var/lib/dr-moagi-cloud \
    && chown -R jarvisx:jarvisx /var/lib/dr-moagi-cloud

USER jarvisx
EXPOSE 8080

CMD ["uvicorn", "jarvisx.dr_moagi_cloud_service:app", "--host", "0.0.0.0", "--port", "8080"]
