#!/bin/bash

set -euo pipefail

REMOTE_HOST="${JETSON_HOST:-nvidia@192.168.1.230}"
REMOTE_DIR="${JETSON_DIR:-/home/nvidia/GPT-SoVITS}"

ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "\
  mkdir -p '$REMOTE_DIR/jetson/models' '$REMOTE_DIR/runtime' '$REMOTE_DIR/test-results' && \
  find '$REMOTE_DIR' -mindepth 1 -maxdepth 1 ! -name 'jetson' ! -name 'runtime' ! -name 'test-results' -exec rm -rf {} + && \
  find '$REMOTE_DIR/jetson' -mindepth 1 -maxdepth 1 ! -name 'models' ! -name 'cache' -exec rm -rf {} +"

tar \
  --exclude=.git \
  --exclude=.venv \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude=venv \
  --exclude=jetson/models \
  --exclude=runtime \
  --exclude=output \
  --exclude=test-results \
  -czf - . | ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "tar -xzf - -C '$REMOTE_DIR'"

echo "Synced repository to $REMOTE_HOST:$REMOTE_DIR"
