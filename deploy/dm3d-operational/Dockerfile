# syntax=docker/dockerfile:1.7
FROM ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates cmake g++ libgomp1 ninja-build \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY cpp_runtime ./cpp_runtime
RUN cmake -S cpp_runtime -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DJARVISX_BUILD_GL_VISUALIZER=OFF \
      -DJARVISX_DM3D_ENABLE_OPENMP=ON \
 && cmake --build build --parallel --target jarvisx-dm3d-operational \
 && ./build/DrMoagi-DM3D-Operational --self-test

FROM ubuntu:24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive \
    OMP_PROC_BIND=spread \
    OMP_PLACES=cores
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/DrMoagi-DM3D-Operational /usr/local/bin/DrMoagi-DM3D-Operational
USER 65532:65532
HEALTHCHECK --interval=30s --timeout=3s --start-period=3s --retries=3 \
  CMD ["/usr/local/bin/DrMoagi-DM3D-Operational", "--health"]
ENTRYPOINT ["/usr/local/bin/DrMoagi-DM3D-Operational"]
CMD ["--side","1000","--brick","25","--iterations","4","--bricks","256"]
