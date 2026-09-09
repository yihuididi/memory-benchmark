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

# An optional dependency extra can contain one or several backend SDKs.
ARG BENCH_EXTRA=""
RUN if [ -n "$BENCH_EXTRA" ]; then \
        uv sync --locked --no-dev --extra "$BENCH_EXTRA"; \
    else \
        uv sync --locked --no-dev; \
    fi
RUN mkdir -p data models results artifacts

ENTRYPOINT ["uv", "run", "--no-sync", "memory-bench"]
CMD ["run", "--config", "configs/demo.toml"]
