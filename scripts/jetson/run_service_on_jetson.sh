#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${JETSON_HOST:-nvidia@192.168.1.230}"
REMOTE_DIR="${JETSON_DIR:-/home/nvidia/GPT-SoVITS}"

"$SCRIPT_DIR/build_on_jetson.sh"

ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" \
  "cd '$REMOTE_DIR' && docker compose -f docker-compose.jetson.yaml up -d"

ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "\
  set -euo pipefail; \
  for attempt in \$(seq 1 60); do \
    if curl -fsS http://127.0.0.1:9880/healthz >/dev/null; then \
      curl -fsS http://127.0.0.1:9880/healthz; \
      exit 0; \
    fi; \
    sleep 5; \
  done; \
  docker logs --tail 200 gpt-sovits-jetson || true; \
  exit 1"
