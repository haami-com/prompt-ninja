FROM node:22-alpine AS frontend-build

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.11.18 AS uv

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/backend/.venv/bin:$PATH" \
    FRONTEND_DIST=/app/frontend/dist \
    PORT=8080

COPY --from=uv /uv /uvx /bin/
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock backend/README.md backend/LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra server

COPY backend/prompt_ninja ./prompt_ninja
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra server

COPY --from=frontend-build --chown=appuser:appuser /build/frontend/dist /app/frontend/dist
RUN chown -R appuser:appuser /app/backend

USER appuser
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"

CMD ["sh", "-c", "exec uvicorn prompt_ninja.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
