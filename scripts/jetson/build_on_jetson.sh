#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${JETSON_HOST:-nvidia@192.168.1.230}"
REMOTE_DIR="${JETSON_DIR:-/home/nvidia/GPT-SoVITS}"

"$SCRIPT_DIR/sync_to_jetson.sh"

ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" \
  "test -f '$REMOTE_DIR/jetson/models/assets/open_jtalk_dic_utf_8-1.11.tar.gz' && \
   test -f '$REMOTE_DIR/jetson/models/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt' && \
   test -f '$REMOTE_DIR/jetson/models/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth' && \
   test -f '$REMOTE_DIR/jetson/models/pretrained_models/fast_langdetect/lid.176.bin' && \
   test -d '$REMOTE_DIR/jetson/models/pretrained_models/chinese-hubert-base' && \
   test -d '$REMOTE_DIR/jetson/models/pretrained_models/chinese-roberta-wwm-ext-large' && \
   test -d '$REMOTE_DIR/jetson/models/G2PWModel' && \
   test -d '$REMOTE_DIR/jetson/models/nltk_data/corpora/cmudict' && \
   test -f '$REMOTE_DIR/jetson/models/nltk_data/corpora/cmudict.zip' && \
   test -f '$REMOTE_DIR/jetson/models/nltk_data/taggers/averaged_perceptron_tagger.zip' && \
   test -d '$REMOTE_DIR/jetson/models/nltk_data/taggers/averaged_perceptron_tagger_eng' && \
   cd '$REMOTE_DIR' && DOCKER_BUILDKIT=1 docker build -f Dockerfile.jetson -t gpt-sovits-jetson:local ."
