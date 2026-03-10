#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${JETSON_HOST:-nvidia@192.168.1.230}"
REMOTE_DIR="${JETSON_DIR:-/home/nvidia/GPT-SoVITS}"
REMOTE_REF_AUDIO_PATH="$TEST_REF_AUDIO_PATH"
CONTAINER_REF_AUDIO_PATH="${TEST_REF_AUDIO_PATH}"

: "${TEST_REF_AUDIO_PATH:?TEST_REF_AUDIO_PATH is required}"
: "${TEST_PROMPT_TEXT:?TEST_PROMPT_TEXT is required}"

"$SCRIPT_DIR/run_service_on_jetson.sh"

ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "\
  cd '$REMOTE_DIR' && \
  mkdir -p test-results && \
  rm -f test-results/tegrastats.log && \
  docker exec gpt-sovits-jetson bash -lc \
    'mkdir -p /workspace/test-results && \
     rm -rf /workspace/test-results/smoke /workspace/test-results/longform /workspace/test-results/concurrency && \
     rm -f /workspace/test-results/report.md'"

if [ -f "$TEST_REF_AUDIO_PATH" ]; then
  REMOTE_REF_AUDIO_PATH="$REMOTE_DIR/runtime/test-assets/$(basename "$TEST_REF_AUDIO_PATH")"
  CONTAINER_REF_AUDIO_PATH="/workspace/runtime/test-assets/$(basename "$TEST_REF_AUDIO_PATH")"
  ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR/runtime/test-assets'"
  scp -o StrictHostKeyChecking=no "$TEST_REF_AUDIO_PATH" "$REMOTE_HOST:$REMOTE_REF_AUDIO_PATH" >/dev/null
elif ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "test -f '$TEST_REF_AUDIO_PATH'"; then
  REMOTE_REF_AUDIO_PATH="$REMOTE_DIR/runtime/test-assets/$(basename "$TEST_REF_AUDIO_PATH")"
  CONTAINER_REF_AUDIO_PATH="/workspace/runtime/test-assets/$(basename "$TEST_REF_AUDIO_PATH")"
  ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR/runtime/test-assets' && cp '$TEST_REF_AUDIO_PATH' '$REMOTE_REF_AUDIO_PATH'"
fi

printf -v CONTAINER_REF_AUDIO_Q "%q" "$CONTAINER_REF_AUDIO_PATH"
printf -v TEST_PROMPT_TEXT_Q "%q" "$TEST_PROMPT_TEXT"
printf -v TEST_PROMPT_LANG_Q "%q" "${TEST_PROMPT_LANG:-zh}"

ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "\
  mkdir -p '$REMOTE_DIR/test-results' && \
  REMOTE_UID=\$(id -u) && \
  REMOTE_GID=\$(id -g) && \
  cd '$REMOTE_DIR' && \
  tegrastats --interval 1000 --logfile '$REMOTE_DIR/test-results/tegrastats.log' >/dev/null 2>&1 & \
  echo \$! > '$REMOTE_DIR/test-results/tegrastats.pid' && \
  docker exec \
    -e TTS_BASE_URL='http://127.0.0.1:9880' \
    -e ASR_BASE_URL='http://192.168.1.230:10001' \
    -e TEST_OUTPUT_DIR='/workspace/test-results' \
    -e TEST_REF_AUDIO_PATH=$CONTAINER_REF_AUDIO_Q \
    -e TEST_PROMPT_TEXT=$TEST_PROMPT_TEXT_Q \
    -e TEST_PROMPT_LANG=$TEST_PROMPT_LANG_Q \
    gpt-sovits-jetson \
    bash -lc \"python3 -m pytest \
      tests/unit/test_text_tools.py \
      tests/integration/test_service_contract.py \
      tests/integration/test_openai_speech.py \
      tests/integration/test_longform.py \
      --run-integration -q && \
      python3 -m pytest tests/integration/test_concurrency.py --run-integration -q\" ; \
  status=\$?; \
  docker exec gpt-sovits-jetson chown -R \$REMOTE_UID:\$REMOTE_GID /workspace/test-results /workspace/runtime >/dev/null 2>&1 || true; \
  if [ -f '$REMOTE_DIR/test-results/tegrastats.pid' ]; then kill \$(cat '$REMOTE_DIR/test-results/tegrastats.pid') || true; fi; \
  exit \$status"
