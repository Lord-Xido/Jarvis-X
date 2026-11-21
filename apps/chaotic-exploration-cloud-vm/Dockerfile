FROM gcc:14 AS build
WORKDIR /build
COPY vm.cpp .
RUN g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror vm.cpp -o jarvisx_vm
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 HOST=0.0.0.0
WORKDIR /app
COPY --from=build /build/jarvisx_vm /app/jarvisx_vm
COPY server.py .
COPY static ./static
RUN useradd -r -u 10001 -s /usr/sbin/nologin vmuser && chown -R vmuser:vmuser /app
USER vmuser
EXPOSE 8080
CMD ["python", "server.py"]
