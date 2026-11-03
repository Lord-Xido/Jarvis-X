FROM python:3.12-alpine
ENV PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /app
RUN adduser -D -u 10001 app
COPY engine.py demo.py server.py index.html ./
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz',timeout=3).read()" || exit 1
CMD ["python", "server.py"]
