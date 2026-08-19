FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml setup.py README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn jarvisx.api:app --host 0.0.0.0 --port ${PORT:-10000}"]
