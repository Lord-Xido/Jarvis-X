FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY setup.py .
RUN pip install .

EXPOSE 10000

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port $PORT"]