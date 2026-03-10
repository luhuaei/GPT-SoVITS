#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${JETSON_HOST:-nvidia@192.168.1.230}"
REMOTE_DIR="${JETSON_DIR:-/home/nvidia/GPT-SoVITS}"
PREPARE_MODELS="${PREPARE_MODELS:-true}"
MODEL_SOURCE="${MODEL_SOURCE:-ModelScope}"

"$SCRIPT_DIR/sync_to_jetson.sh"

if [ "$PREPARE_MODELS" = "true" ]; then
  MODEL_SOURCE="$MODEL_SOURCE" "$SCRIPT_DIR/prepare_models_on_jetson.sh"
fi

REF_JSON=$(ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "cd '$REMOTE_DIR' && python3 scripts/jetson/resolve_test_reference.py")
TEST_REF_AUDIO_PATH=$(python3 - <<'PY' "$REF_JSON"
import json, sys
print(json.loads(sys.argv[1])["ref_audio_path"])
PY
)
TEST_PROMPT_TEXT=$(python3 - <<'PY' "$REF_JSON"
import json, sys
print(json.loads(sys.argv[1])["prompt_text"])
PY
)
TEST_PROMPT_LANG=$(python3 - <<'PY' "$REF_JSON"
import json, sys
print(json.loads(sys.argv[1])["prompt_lang"])
PY
)

export TEST_REF_AUDIO_PATH
export TEST_PROMPT_TEXT
export TEST_PROMPT_LANG

"$SCRIPT_DIR/run_tests_on_jetson.sh"

ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "cd '$REMOTE_DIR' && python3 scripts/jetson/generate_report.py test-results >/tmp/gsv-report-path.txt && cat /tmp/gsv-report-path.txt && echo '---REPORT---' && cat test-results/report.md"
