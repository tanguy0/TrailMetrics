# Compute API image. Works unchanged on Railway and Render.
#
# The Streamlit app and the notebooks are not part of this image — it exists to
# serve api/ and the analytics under src/.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# glibc opens an arena per thread (up to 8x the core count) and each one fragments
# independently, so a threaded process holds far more than it uses and its RSS
# never comes back down — which is how a container gets killed on a request
# smaller than the one that actually caused the spike. Two arenas costs a little
# allocator contention and keeps the measured footprint near the live one.
ENV MALLOC_ARENA_MAX=2

# The two memory ceilings, both overridable per environment: how many CPU-bound
# handlers may run at once (api/main.py) and the byte budget for the render memo
# (api/memo.py). Raise them together with the container's memory, never ahead of
# it — 4 x 192 MB is sized for a 1 GB instance.
ENV MAX_CONCURRENT_REQUESTS=4 \
    MEMO_BUDGET_MB=192

# libgomp is required by XGBoost at runtime; curl is for the container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Dependencies ----------------------------------------------------------
# Copied on their own so a code change doesn't rebuild the (slow) scientific stack.
COPY requirements-api.txt ./
COPY requirements-compute.txt ./
RUN pip install --no-cache-dir -r requirements-compute.txt -r requirements-api.txt

# --- Application -----------------------------------------------------------
COPY pyproject.toml ./
COPY src ./src
COPY api ./api

# Run as a non-root user.
RUN useradd --create-home --uid 10001 trailmetrics \
    && chown -R trailmetrics:trailmetrics /app
USER trailmetrics

# Platforms inject PORT; 8000 is the local default.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" > /dev/null || exit 1

# One worker: each request may hold a multi-hundred-MB pandas frame and a fitted
# model, and the per-athlete caches are per process. Scale with replicas, and
# raise this only alongside the memory to back it.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 65"]
