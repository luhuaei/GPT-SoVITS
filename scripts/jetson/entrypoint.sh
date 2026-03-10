#!/bin/bash

set -euo pipefail

PROJECT_DIR="/workspace/GPT-SoVITS"

mkdir -p "${GPT_SOVITS_RUNTIME_DIR:-/workspace/runtime}"

cd "$PROJECT_DIR"
exec "$@"
