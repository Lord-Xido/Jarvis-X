FROM python:3.12-slim
WORKDIR /app
COPY codec.py server.py ./
COPY static/ ./static/
RUN useradd -r -u 10001 jx
USER jx
ENV HOST=0.0.0.0 PORT=8080 PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "server.py"]
