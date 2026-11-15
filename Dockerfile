# Application image: Streamlit dashboard + the DuckDB warehouse it reads.
# Three stages:
#   base  - Python and the dependencies (shared by the two others, installed once)
#   data  - downloads the Météo-France files and builds the warehouse, figures and results
#   app   - the dashboard with only what it needs from "data" (no raw files: ~300 MB instead of ~600 MB)
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt


FROM base AS data
COPY config.yaml .
COPY meteo ./meteo
COPY sql ./sql
COPY templates ./templates
# The ERA5 comparison is versioned: rebuilding it would call the Open-Meteo API on every build
COPY reports/tables/era5_validation.csv ./reports/tables/
# A new DATA_VERSION (CI passes the date) downloads fresh data; the same value reuses the cached layer
ARG DATA_VERSION=dev
RUN echo "Data version: ${DATA_VERSION}" \
    && python -m meteo download \
    && python -m meteo build \
    && python -m meteo analyze \
    && python -m meteo report


FROM base AS app
# IDLE_PAUSE_MINUTES: an idle tab disconnects after that long (dashboard/idle.py), so the service can scale to zero
ENV PORT=8080 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    IDLE_PAUSE_MINUTES=20

# Caddy in front of Streamlit: compression, security headers, robots.txt, the "paused" page (docker/Caddyfile)
COPY --from=caddy:2.11.9-alpine /usr/bin/caddy /usr/bin/caddy

# Unprivileged user. The files stay owned by root: the app only reads them (the warehouse is opened read-only),
# and a `chown -R` would copy the whole warehouse into a new layer
RUN useradd --create-home --uid 10001 appuser

COPY config.yaml .
COPY .streamlit ./.streamlit
COPY meteo ./meteo
COPY sql ./sql
COPY dashboard ./dashboard
COPY samples ./samples
COPY --from=data /app/data/warehouse.duckdb /app/data/pipeline_stats.json ./data/
COPY --from=data /app/reports ./reports
COPY docker ./docker
USER appuser

# Cloud Run sets PORT (8080 by default); the same image runs anywhere with `docker run -p 8080:8080`
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8080\")}/_stcore/health', timeout=4)"

# Streamlit on 127.0.0.1:8501, Caddy on $PORT in front of it (docker/start.sh)
CMD ["bash", "docker/start.sh"]
