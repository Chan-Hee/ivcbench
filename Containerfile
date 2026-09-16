# Optional GPU-free reproduction image. Primary verification uses the local CPU path.
# 3.10 matches the verified environment stated in requirements-core.txt and REPRODUCE.md;
# the image used to pin 3.13, which is not the interpreter any of those checks ran on.
FROM python:3.10-slim
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
