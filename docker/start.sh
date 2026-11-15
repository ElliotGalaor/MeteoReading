#!/usr/bin/env bash
# Starts Streamlit (internal, port 8501) and Caddy in front of it (public, $PORT).
# The container stops as soon as either process stops, so that Cloud Run (or Docker) restarts it.
set -euo pipefail

streamlit run dashboard/app.py --server.port=8501 --server.address=127.0.0.1 &

# Caddy opens the public port only once Streamlit listens: the startup probe then waits for the app
until (exec 3<>/dev/tcp/127.0.0.1/8501) 2>/dev/null; do   # opens and closes a connection, sends nothing
    sleep 0.2
done

caddy run --config /app/docker/Caddyfile --adapter caddyfile &

trap 'kill -TERM $(jobs -p) 2>/dev/null' TERM INT
wait -n
