# Batch worker image. No listening socket or exposed HTTP API.
FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates cmake g++ make libpng-dev zlib1g-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt cubepng_neural.cpp neural_mlp.hpp ./
RUN cmake -S . -B /tmp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
 && cmake --build /tmp/build --parallel 2 \
 && install -Dm755 /tmp/build/cubepng_neural /out/cubepng_neural
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libpng16-16 zlib1g libssl3 \
    && rm -rf /var/lib/apt/lists/* \
 && groupadd -g 65532 worker && useradd -u 65532 -g 65532 -M -s /usr/sbin/nologin worker
COPY --from=build /out/cubepng_neural /usr/local/bin/cubepng_neural
WORKDIR /data
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/cubepng_neural"]
CMD ["self-test"]
