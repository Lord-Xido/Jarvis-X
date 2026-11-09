# CPU-only runtime; pin upstream image digests before production promotion.
FROM gcc:14-bookworm AS build
WORKDIR /src
COPY src/jx3d_cloud.cpp /src/jx3d_cloud.cpp
RUN g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -pthread /src/jx3d_cloud.cpp -o /jx3d_cloud
FROM debian:bookworm-slim
RUN groupadd --gid 10001 jx && useradd --uid 10001 --gid jx --create-home jx && mkdir -p /data && chown jx:jx /data
USER jx
COPY --from=build /jx3d_cloud /usr/local/bin/jx3d_cloud
VOLUME ["/data"]
ENTRYPOINT ["/usr/local/bin/jx3d_cloud"]
CMD ["--depth","2","--steps","24","--workers","4","--out","/data","--events","sample"]
