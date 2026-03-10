#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${JETSON_HOST:-nvidia@192.168.1.230}"
REMOTE_DIR="${JETSON_DIR:-/home/nvidia/GPT-SoVITS}"

"$SCRIPT_DIR/sync_to_jetson.sh"

ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" \
  "test -f '$REMOTE_DIR/jetson/models/assets/open_jtalk_dic_utf_8-1.11.tar.gz' && \
   test -d '$REMOTE_DIR/jetson/models/pretrained_models' && \
   test -d '$REMOTE_DIR/jetson/models/G2PWModel' && \
   test -d '$REMOTE_DIR/jetson/models/nltk_data' && \
   cd '$REMOTE_DIR' && docker build -f Dockerfile.jetson -t gpt-sovits-jetson:local ."
