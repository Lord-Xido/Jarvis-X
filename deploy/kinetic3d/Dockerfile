FROM debian:bookworm-slim AS native-build

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY native/kinetic3d_backend.cpp /src/kinetic3d_backend.cpp
RUN mkdir -p /out \
    && g++ -O3 -std=c++17 -fPIC -shared \
       /src/kinetic3d_backend.cpp \
       -o /out/libjarvisx_kinetic3d.so

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JARVISX_KINETIC3D_NATIVE_LIB=/opt/jarvisx-native/libjarvisx_kinetic3d.so

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY --from=native-build /out/libjarvisx_kinetic3d.so /opt/jarvisx-native/libjarvisx_kinetic3d.so
RUN python -m pip install --no-cache-dir .

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).read()"

CMD ["uvicorn", "jarvisx.kinetic3d_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
