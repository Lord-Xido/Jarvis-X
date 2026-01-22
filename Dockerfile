FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
COPY setup.py .
RUN pip install .
EXPOSE 8080 5000 9000
CMD ["jarvisx", "api"]
