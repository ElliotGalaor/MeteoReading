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
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt


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
ENV PORT=8080 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY config.yaml .
COPY .streamlit ./.streamlit
COPY meteo ./meteo
COPY sql ./sql
COPY dashboard ./dashboard
COPY samples ./samples
COPY --from=data /app/data/warehouse.duckdb /app/data/pipeline_stats.json ./data/
COPY --from=data /app/reports ./reports

# Unprivileged user; the warehouse is only ever opened read-only
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

# Cloud Run sets PORT (8080 by default); the same image runs anywhere with `docker run -p 8080:8080`
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8080\")}/_stcore/health', timeout=4)"

CMD ["sh", "-c", "exec streamlit run dashboard/app.py --server.port=${PORT} --server.address=0.0.0.0"]
