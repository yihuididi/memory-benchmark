FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /bin/

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1

COPY pyproject.toml uv.lock README.md .python-version ./
COPY src/ ./src/
COPY configs/ ./configs/

# Space-separated optional dependency extras from pyproject.toml.
ARG BENCH_EXTRA=""
RUN set -eu; set -f; \
    set --; \
    for extra in $BENCH_EXTRA; do \
        set -- "$@" --extra "$extra"; \
    done; \
    uv sync --locked --no-dev "$@"
RUN mkdir -p data models results artifacts

ENTRYPOINT ["uv", "run", "--no-sync", "memory-bench"]
CMD ["run", "--config", "configs/demo.toml"]
