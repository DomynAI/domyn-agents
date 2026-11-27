# One image for the example's processes: the bank's services, `domyn serve`, the alerts
# and the console. Built from the repository's root (see docker-compose.yaml), so it runs
# the domyn of this checkout; Dockerfile.dockerignore keeps the rest of the repository out.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /src
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY domyn domyn
COPY docs/docs docs/docs
RUN pip install --no-cache-dir ".[kafka]"

WORKDIR /src/examples/fraud-alerts
COPY examples/fraud-alerts .
