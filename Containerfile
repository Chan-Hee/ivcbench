# Optional GPU-free reproduction image. Primary verification uses the local CPU path.
FROM python:3.13-slim
WORKDIR /ivcbench
RUN apt-get update \
    && apt-get install -y --no-install-recommends make build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-core.txt pyproject.toml ./
RUN python -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements-core.txt
COPY . .
RUN .venv/bin/pip install --no-cache-dir -e .
CMD ["make", "reproduce"]
