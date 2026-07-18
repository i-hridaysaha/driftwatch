FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY configs/ ./configs/
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "src/driftwatch/dashboard/app.py", "--server.address=0.0.0.0"]
